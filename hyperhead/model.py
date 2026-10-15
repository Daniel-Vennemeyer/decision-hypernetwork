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


def _map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


def prefetch(prepare, batches):
    """Yields prepare(b) for each b, preparing the next batch in a background thread."""
    from concurrent.futures import ThreadPoolExecutor
    batches = iter(batches)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = next(batches, None)
        if first is None:
            return
        pending = pool.submit(prepare, first)
        for upcoming in batches:
            ready = pending.result()
            pending = pool.submit(prepare, upcoming)
            yield ready
        yield pending.result()


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

    def _clip(self, row, reserve=768):
        """Shortens an over-long state so the prompt's answer cue is never truncated away."""
        ids = self.tokenizer.encode(data.text(row['state']), add_special_tokens=False)
        budget = self.max_tokens - reserve
        if len(ids) <= budget:
            return row
        return row | {'state': self.tokenizer.decode(ids[:budget]) + ' ...'}

    # ------------------------------------------------------------------ CPU side
    def token_lengths(self, rows, chunk=2048):
        """(spec_lens, decision_lens): prompt lengths in tokens for both passes, capped at
        max_tokens. Used to build micro-batches by token budget."""
        spec, decision = [], []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            spec_texts = [data.spec_prompt(self.tokenizer, r)[0] for r in part]
            if self.mode == 'joint':
                decision_texts = [data.joint_prompt(self.tokenizer, r, self.labels) for r in part]
            else:
                decision_texts = [data.state_prompt(self.tokenizer, r) for r in part]
            for texts, out in ((spec_texts, spec), (decision_texts, decision)):
                ids = self.tokenizer(texts, add_special_tokens=False)['input_ids']
                out.extend(min(len(x), self.max_tokens) for x in ids)
        return spec, decision

    def prepare(self, rows, spec=True, decision=True):
        """All CPU work for a batch: prompts, tokenization, option-pooling masks, targets.
        Thread-safe with respect to the GPU passes, so the next batch can be prepared in a
        background thread while the current one runs (see prefetch()). spec / decision select
        which pass to prepare (evaluation prepares them separately to cache heads per spec)."""
        from .objective import targets

        k_max = max(len(r['options']) for r in rows)
        batch = dict(n_options=torch.tensor([len(r['options']) for r in rows]), k_max=k_max,
                     targets=targets(rows, k_max))
        if spec:
            texts, spans = zip(*(data.spec_prompt(self.tokenizer, r) for r in rows))
            enc = self._encode(list(texts), offsets=True)
            offsets = enc.pop('offset_mapping')
            lengths = enc['attention_mask'].sum(-1)
            pool = torch.zeros(len(rows), k_max, enc['input_ids'].shape[1])
            for i, row_spans in enumerate(spans):
                start, end = offsets[i, :, 0], offsets[i, :, 1]
                for k, (a, b) in enumerate(row_spans):
                    inside = (start >= a) & (end <= b) & (end > start)
                    if not inside.any():  # option truncated away: fall back to the last token
                        inside[lengths[i] - 1] = True
                    pool[i, k] = inside.float() / inside.sum()
            batch.update(spec=dict(enc), pool=pool)
        if decision:
            clipped = [self._clip(r) for r in rows]
            if self.mode == 'joint':
                texts = [data.joint_prompt(self.tokenizer, r, self.labels) for r in clipped]
            else:
                texts = [data.state_prompt(self.tokenizer, r) for r in clipped]
            batch.update(decision=dict(self._encode(texts)))
        if torch.cuda.is_available():  # pinned memory -> asynchronous host-to-device copies
            batch = _map_tensors(batch, lambda t: t.pin_memory())
        return batch

    def to_device(self, batch):
        return _map_tensors(batch, lambda t: t.to(self.device, non_blocking=True))

    # ------------------------------------------------------------------ GPU passes
    def generate_heads(self, batch):
        """Hypernetwork pass (LoRA on). Depends only on question + options, so it can be cached."""
        spec = batch['spec']
        hidden = self._text_model()(**spec, use_cache=False).last_hidden_state.float()
        option_emb = torch.einsum('bkt,btd->bkd', batch['pool'], hidden)
        spec_emb = self._last(hidden, spec['attention_mask'])
        return self.generator(option_emb, spec_emb)

    @torch.no_grad()
    def read_states(self, batch):
        """Decision pass with the original weights (LoRA off). Returns reader inputs and, in joint
        mode, the frozen answer-slot logits over the lettered options. Only the read layers are
        kept (forward hooks), not all hidden states."""
        enc = batch['decision']
        model = self._text_model()
        n_layers = len(model.layers)
        captured = {}
        hooks = [model.layers[i - 1].register_forward_hook(
                     lambda _m, _i, out, i=i: captured.__setitem__(i, out[0] if isinstance(out, tuple) else out))
                 for i in self.read_layers if i < n_layers]
        try:
            with self.lm.disable_adapter():
                out = model(**enc, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        captured[n_layers] = out.last_hidden_state  # == hidden_states[n_layers] (post-norm)
        mask = enc['attention_mask'].bool()
        # zero padded positions so they cannot disturb the reader's normalisation
        states = torch.cat([captured[i] for i in self.read_layers], -1).float()
        states = states.masked_fill(~mask[..., None], 0.0)
        base = None
        if self.mode == 'joint':
            weight = self.lm.get_base_model().get_output_embeddings().weight[self.label_ids[:batch['k_max']]]
            base = F.linear(self._last(out.last_hidden_state, mask), weight).float()
        return dict(states=states, mask=mask, base=base)

    def decide(self, head, read):
        logits = self.generator.apply(head, read['states'], read['mask'])
        if read['base'] is not None:
            logits = logits + read['base']
        return logits

    def mask_options(self, logits, n_options):
        invalid = torch.arange(logits.shape[-1], device=logits.device)[None] >= n_options[:, None]
        return logits.masked_fill(invalid, -1e9)

    def forward(self, batch):
        """batch: rows (list of dicts) or the output of prepare()."""
        if isinstance(batch, list):
            batch = self.prepare(batch)
        batch = self.to_device(batch)
        read = self.read_states(batch)
        logits = self.decide(self.generate_heads(batch), read)
        return self.mask_options(logits, batch['n_options']), batch

    def head_is_zero(self):
        """True while the generator's output layers are still at their zero initialisation: every
        generated logit is then exactly 0, so the hypernetwork pass can be skipped (joint mode then
        equals the frozen answer-slot readout)."""
        layers = (self.generator.option[-1], self.generator.bias[-1])
        return all(not p.any() for layer in layers for p in (layer.weight, layer.bias))

    @staticmethod
    def split_heads(head, n_options):
        """Batched head -> one head per row (cacheable per spec)."""
        return [dict(w=head['w'][i, :n], b=head['b'][i, :n], q=head['q'][i])
                for i, n in enumerate(n_options.tolist())]

    @staticmethod
    def stack_heads(heads, k_max):
        """Per-row heads -> a batched head padded to k_max options (padding is masked later)."""
        w = torch.zeros(len(heads), k_max, heads[0]['w'].shape[-1], device=heads[0]['w'].device)
        b = torch.zeros(len(heads), k_max, device=w.device)
        for i, h in enumerate(heads):
            w[i, :len(h['b'])] = h['w']
            b[i, :len(h['b'])] = h['b']
        return dict(w=w, b=b, q=torch.stack([h['q'] for h in heads]))

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
