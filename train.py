"""Train a HyperHead on SII-Decision teacher distributions.

    python train.py --base Qwen/Qwen3.5-4B --mode joint --out runs/joint-4b
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from evaluate import predict
from hyperhead import data, metrics
from hyperhead.model import HyperHead, prefetch
from hyperhead.objective import distillation_loss


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='Qwen/Qwen3.5-4B')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--mode', choices=['joint', 'split'], default='joint')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rank', type=int, default=256)
    ap.add_argument('--queries', type=int, default=4)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--lr-lora', type=float, default=5e-5)
    ap.add_argument('--lr-head', type=float, default=2e-4)
    ap.add_argument('--warmup', type=float, default=0.03)
    ap.add_argument('--kappa', type=float, default=0.05)
    ap.add_argument('--anchor', choices=['auto', 'self', 'data', 'none'], default='auto',
                    help="KL anchor / gain reference: the model's own frozen readout (self, joint "
                         "mode only), the dataset's Qwen3.5-4B readout (data), or none (kappa 0). "
                         "auto: self in joint mode; in split mode data for Qwen3.5-4B, else none")
    ap.add_argument('--gold-weight', type=float, default=0.0, help='mix gold labels into the teacher')
    ap.add_argument('--train-rows', type=int, default=None)
    ap.add_argument('--eval-rows', type=int, default=1000)
    ap.add_argument('--eval-per-source', type=int, default=None,
                    help='in-training eval on a fixed stratified subset (N rows per source)')
    ap.add_argument('--eval-every', type=int, default=250,
                    help='optimizer steps; 0 = no evaluation during training (checkpoint = last)')
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--max-spec-tokens', type=int, default=8192,
                    help='micro-batch cap: rows x longest spec prompt (hypernetwork pass, with grads)')
    ap.add_argument('--max-decision-tokens', type=int, default=16384,
                    help='micro-batch cap: rows x longest decision prompt (no grads)')
    ap.add_argument('--no-grad-ckpt', action='store_true',
                    help='keep activations instead of recomputing them: faster, more memory')
    ap.add_argument('--max-steps', type=int, default=None, help='stop early, e.g. for a timing run')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True  # the head generator runs in fp32
        from hyperhead.kernels import linear_attention_backend
        # fla carries the cost; causal-conv1d is optional (torch conv1d fallback is cheap)
        if not linear_attention_backend()['fla']:
            print('WARNING: flash-linear-attention not found; Qwen3.5 linear-attention layers will use '
                  'the slow torch fallback (see scripts/setup_cuda.sh).', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(args), indent=2))

    anchor = args.anchor
    if anchor == 'auto':
        anchor = 'self' if args.mode == 'joint' else ('data' if 'Qwen3.5-4B' in args.base else 'none')
    if anchor == 'self' and args.mode != 'joint':
        raise SystemExit('--anchor self needs joint mode (split mode has no frozen readout)')
    if anchor == 'none':
        args.kappa = 0.0
    print(f'anchor: {anchor}' + ('' if anchor != 'none' else
          " (gain is then measured against the dataset's Qwen3.5-4B readout)"), flush=True)
    model = HyperHead(args.base, mode=args.mode, rank=args.rank, queries=args.queries,
                      lora_r=args.lora_r, max_tokens=args.max_tokens, revision=args.revision,
                      gradient_checkpointing=not args.no_grad_ckpt).to(device)
    train_rows = data.read_jsonl(Path(args.data_dir) / 'train.jsonl', args.train_rows, args.seed)
    if args.eval_per_source:
        val_rows = data.per_source_sample(data.read_jsonl(Path(args.data_dir) / 'validation.jsonl'),
                                          args.eval_per_source)
    else:
        val_rows = data.read_jsonl(Path(args.data_dir) / 'validation.jsonl', args.eval_rows, args.seed)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    head = list(model.generator.parameters())
    print(f'trainable: lora {sum(p.numel() for p in lora)/1e6:.1f}M  head {sum(p.numel() for p in head)/1e6:.1f}M'
          f'  train {len(train_rows)}  val {len(val_rows)}  device {device}', flush=True)

    optimizer = torch.optim.AdamW([{'params': lora, 'lr': args.lr_lora},
                                   {'params': head, 'lr': args.lr_head}], weight_decay=0.0)
    # micro-batches by padded token count, so memory stays bounded whatever the row mix
    started = time.time()
    spec_lens, decision_lens = model.token_lengths(train_rows)
    print(f'tokenized {len(train_rows)} rows in {time.time() - started:.0f}s', flush=True)
    batches = []
    for epoch in range(math.ceil(args.epochs)):
        batches += data.token_batches(spec_lens, decision_lens, args.batch, args.max_spec_tokens,
                                      args.max_decision_tokens, seed=args.seed + epoch)
    keep, seen = [], 0
    for b in batches:  # fractional epochs
        if seen >= args.epochs * len(train_rows):
            break
        keep.append(b)
        seen += len(b)
    batches = keep
    # run the most memory-hungry micro-batch first, so an out-of-memory error shows up in the
    # first seconds (e.g. in the timing stage) rather than hours into training
    padded = lambda b: (len(b) * max(spec_lens[j] for j in b), len(b) * max(decision_lens[j] for j in b))
    biggest = max(range(len(batches)), key=lambda k: padded(batches[k]))
    batches.insert(0, batches.pop(biggest))
    sizes = [len(b) for b in batches]
    print(f'{len(batches)} micro-batches, mean {sum(sizes) / len(sizes):.1f} rows; first (largest) has '
          f'{padded(batches[0])[0]} padded spec / {padded(batches[0])[1]} decision tokens', flush=True)
    total_steps = max(1, len(batches) // args.accum)
    warmup = max(1, int(args.warmup * total_steps))
    schedule = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    eval_seconds = 0.0

    def run_eval(step):
        nonlocal eval_seconds
        started = time.time()
        model.eval()
        result = metrics.evaluate(val_rows, predict(model, val_rows, args.batch * 2))
        model.train()
        eval_seconds += time.time() - started
        summary = {k: result[k] for k in ('score', 'accuracy', 'ece', 'brier', 'nll')}
        print(f'[eval step {step}] ' + '  '.join(f'{k} {v:.4f}' for k, v in summary.items()), flush=True)
        with (out / 'eval.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **summary, 'capabilities': result['capabilities']}) + '\n')
        return result['score']

    evaluating = args.eval_every > 0
    best = run_eval(0) if evaluating else None
    if evaluating:
        model.save(out / 'best')
    model.train()
    step, t0, eval_seconds, running, rows_done = 0, time.time(), 0.0, {}, 0
    batch_rows = ([train_rows[j] for j in b] for b in batches)
    for i, batch in enumerate(prefetch(model.prepare, batch_rows)):
        logits, batch = model(batch)
        p0 = batch['base'].softmax(-1) if anchor == 'self' else None
        loss, parts = distillation_loss(logits, batch['targets'], kappa=args.kappa,
                                        gold_weight=args.gold_weight, p0=p0)
        (loss / args.accum).backward()
        rows_done += len(batch['n_options'])
        for k, v in (parts | {'loss': loss.detach()}).items():  # stays on the GPU: no sync per step
            running[k] = running.get(k, 0.0) + v
        if (i + 1) % args.accum:
            continue
        torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            n = args.log_every * args.accum
            train_seconds = time.time() - t0 - eval_seconds  # training time only
            per_step = train_seconds / step
            memory = (f' peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB'
                      if device.type == 'cuda' else '')
            print(f'step {step}/{total_steps} ' + ' '.join(f'{k} {float(v)/n:.4f}' for k, v in running.items())
                  + f' lr {scheduler.get_last_lr()[0]:.2e} {per_step:.2f}s/step'
                  + f' {rows_done / train_seconds:.1f} rows/s'
                  + f' eta {(total_steps - step) * per_step / 3600:.2f}h{memory}', flush=True)
            running = {}
        if args.max_steps and step >= args.max_steps:
            break
        if evaluating and (step % args.eval_every == 0 or step == total_steps):
            score = run_eval(step)
            if score > best:
                best = score
                model.save(out / 'best')
    model.save(out / 'last')
    if not evaluating:  # no validation during training: the final checkpoint is the one to use
        model.save(out / 'best')
        print('done (no in-training evaluation; best = last)')
    else:
        print(f'done. best val score {best:.2f}')


if __name__ == '__main__':
    main()
