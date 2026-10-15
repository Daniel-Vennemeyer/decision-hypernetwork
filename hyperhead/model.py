"""HyperHead: one Qwen backbone that writes a decision head for itself.

Two passes share the same frozen weights:

  hypernetwork pass  (LoRA ON, trainable)   spec prompt  -> per-option vectors w_k, bias b_k,
                                                            pooling queries q_1..q_m
  decision pass      (LoRA OFF, no grad)    state / joint prompt -> hidden states H

The generated head reads H with the queries (attention pooling over a few layers) and scores
every option: logit_k = <w_k, [u_1..u_m]> / sqrt(m r) + b_k.

mode="joint": the decision pass sees state + question + options (the SII-Decision prompt) and
              the generated logits are added to the frozen answer-slot readout, so training
              starts exactly at the base model's no-thinking distribution (p0).
mode="split": the decision pass sees only the state. Heads depend only on the spec and the
              state pass only on the state, so one state pass serves any number of cached heads.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import data


def _mlp(d_in, d_hidden, d_out, zero_last=False):
    last = nn.Linear(d_hidden, d_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(), last)


class HeadGenerator(nn.Module):
    """Maps spec-pass embeddings to a decision head; also holds the shared state reader."""

    def __init__(self, d_model, n_read, rank, queries):
        super().__init__()
        self.rank, self.queries = rank, queries
        # parameter-free norm (the Linear absorbs the affine part); an affine nn.LayerNorm here
        # produced spurious inf bias gradients on MPS.
        self.reader = nn.Sequential(nn.LayerNorm(d_model * n_read, elementwise_affine=False),
                                    nn.Linear(d_model * n_read, rank))
        self.option = _mlp(d_model, d_model, queries * rank, zero_last=True)
        self.bias = _mlp(d_model, d_model // 4, 1, zero_last=True)
        self.query = _mlp(d_model, d_model, queries * rank)

    def forward(self, option_emb, spec_emb):
        b, k, _ = option_emb.shape
        return dict(w=self.option(option_emb),                                     # [B, K, m*r]
                    b=self.bias(option_emb).squeeze(-1),                            # [B, K]
                    q=self.query(spec_emb).view(b, self.queries, self.rank))        # [B, m, r]

    def apply(self, head, states, mask):
        z = self.reader(states)                                                     # [B, T, r]
        scores = torch.einsum('bmr,btr->bmt', head['q'], z) / math.sqrt(self.rank)
        scores = scores.masked_fill(~mask[:, None, :], float('-inf'))
        pooled = torch.einsum('bmt,btr->bmr', scores.softmax(-1), z).flatten(1)     # [B, m*r]
        return torch.einsum('bkx,bx->bk', head['w'], pooled) / math.sqrt(pooled.shape[-1]) + head['b']


class HyperHead(nn.Module):
    def __init__(self, base='Qwen/Qwen3.5-4B', mode='joint', rank=256, queries=4,
                 read_layers=(0.5, 0.75, 1.0), lora_r=16, lora_alpha=32, max_tokens=4096,
                 dtype=torch.bfloat16, revision=None, gradient_checkpointing=True):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert mode in ('joint', 'split')
        self.config = dict(base=base, mode=mode, rank=rank, queries=queries,
                           read_layers=list(read_layers), lora_r=lora_r, lora_alpha=lora_alpha,
                           max_tokens=max_tokens, revision=revision)
        self.mode, self.max_tokens = mode, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        self.tokenizer.padding_side = 'right'
        backbone = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=dtype)
        backbone.requires_grad_(False)
        n_layers = backbone.config.num_hidden_layers
        # fractions of depth -> hidden_states indices (index 0 is the embedding output)
        self.read_layers = sorted({max(1, round(f * n_layers)) for f in read_layers})
        self.labels = data.option_labels(self.tokenizer)
        self.register_buffer('label_ids', torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(l))[0] for l in self.labels]),
            persistent=False)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            backbone.enable_input_require_grads()
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                          target_modules='all-linear', task_type='CAUSAL_LM')
        self.lm = get_peft_model(backbone, lora)
        self.generator = HeadGenerator(backbone.config.hidden_size, len(self.read_layers), rank, queries)

    # ------------------------------------------------------------------ helpers
    @property
    def device(self):
        return self.generator.reader[1].weight.device

    def _text_model(self):
        return self.lm.get_base_model().model

    def _encode(self, texts, offsets=False):
        enc = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                             max_length=self.max_tokens, return_offsets_mapping=offsets,
                             add_special_tokens=False)
        return enc

    @staticmethod
    def _last(hidden, mask):
        return hidden[torch.arange(len(hidden), device=hidden.device), mask.sum(-1) - 1]

    # ------------------------------------------------------------------ passes
    def generate_heads(self, rows):
        """Hypernetwork pass (LoRA on). Depends only on question + options, so it can be cached."""
        texts, spans = zip(*(data.spec_prompt(self.tokenizer, r) for r in rows))
        enc = self._encode(list(texts), offsets=True)
        offsets = enc.pop('offset_mapping')
        enc = enc.to(self.device)
        hidden = self._text_model()(**enc, use_cache=False).last_hidden_state.float()
        k_max = max(len(s) for s in spans)
        pool = torch.zeros(len(rows), k_max, hidden.shape[1])
        for i, row_spans in enumerate(spans):
            start, end = offsets[i, :, 0], offsets[i, :, 1]
            for k, (a, b) in enumerate(row_spans):
                inside = (start >= a) & (end <= b) & (end > start)
                if not inside.any():  # option truncated away: fall back to the last token
                    inside[enc['attention_mask'][i].sum() - 1] = True
                pool[i, k] = inside.float() / inside.sum()
        option_emb = torch.einsum('bkt,btd->bkd', pool.to(hidden), hidden)
        spec_emb = self._last(hidden, enc['attention_mask'])
        return self.generator(option_emb, spec_emb)

    def _clip(self, row, reserve=768):
        """Shortens an over-long state so the prompt's answer cue is never truncated away."""
        ids = self.tokenizer.encode(data.text(row['state']), add_special_tokens=False)
        budget = self.max_tokens - reserve
        if len(ids) <= budget:
            return row
        return row | {'state': self.tokenizer.decode(ids[:budget]) + ' ...'}

    @torch.no_grad()
    def read_states(self, rows):
        """Decision pass with the original weights (LoRA off). Returns reader inputs and, in joint
        mode, the frozen answer-slot logits over the lettered options."""
        rows = [self._clip(r) for r in rows]
        if self.mode == 'joint':
            texts = [data.joint_prompt(self.tokenizer, r, self.labels) for r in rows]
        else:
            texts = [data.state_prompt(self.tokenizer, r) for r in rows]
        enc = self._encode(texts).to(self.device)
        with self.lm.disable_adapter():
            out = self._text_model()(**enc, use_cache=False, output_hidden_states=True)
        mask = enc['attention_mask'].bool()
        # padded positions can hold huge values (linear-attention layers); they would overflow
        # the reader's LayerNorm statistics and its gradients, so zero them.
        states = torch.cat([out.hidden_states[i] for i in self.read_layers], -1).float()
        states = states.masked_fill(~mask[..., None], 0.0)
        base = None
        if self.mode == 'joint':
            k_max = max(len(r['options']) for r in rows)
            weight = self.lm.get_base_model().get_output_embeddings().weight[self.label_ids[:k_max]]
            base = F.linear(self._last(out.last_hidden_state, mask), weight).float()
        return dict(states=states, mask=mask, base=base)

    def decide(self, head, read):
        logits = self.generator.apply(head, read['states'], read['mask'])
        if read['base'] is not None:
            logits = logits + read['base']
        return logits

    def forward(self, rows):
        read = self.read_states(rows)
        logits = self.decide(self.generate_heads(rows), read)
        n = torch.tensor([len(r['options']) for r in rows], device=logits.device)
        invalid = torch.arange(logits.shape[-1], device=logits.device)[None] >= n[:, None]
        return logits.masked_fill(invalid, -1e9), read['base']

    # ------------------------------------------------------------------ io
    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items()
                if k.startswith('generator.') or 'lora_' in k}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state(), path / 'hyperhead.pt')
        (path / 'config.json').write_text(json.dumps(self.config, indent=2))

    @classmethod
    def load(cls, path, **overrides):
        path = Path(path)
        config = json.loads((path / 'config.json').read_text()) | overrides
        model = cls(**config, gradient_checkpointing=False)
        missing = model.load_state_dict(torch.load(path / 'hyperhead.pt', map_location='cpu'), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        return model
