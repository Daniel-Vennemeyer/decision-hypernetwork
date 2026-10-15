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
from hyperhead.model import HyperHead
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
    ap.add_argument('--gold-weight', type=float, default=0.0, help='mix gold labels into the teacher')
    ap.add_argument('--train-rows', type=int, default=None)
    ap.add_argument('--eval-rows', type=int, default=1000)
    ap.add_argument('--eval-every', type=int, default=250, help='optimizer steps')
    ap.add_argument('--log-every', type=int, default=10)
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

    model = HyperHead(args.base, mode=args.mode, rank=args.rank, queries=args.queries,
                      lora_r=args.lora_r, max_tokens=args.max_tokens, revision=args.revision).to(device)
    train_rows = data.read_jsonl(Path(args.data_dir) / 'train.jsonl', args.train_rows, args.seed)
    val_rows = data.read_jsonl(Path(args.data_dir) / 'validation.jsonl', args.eval_rows, args.seed)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    head = list(model.generator.parameters())
    print(f'trainable: lora {sum(p.numel() for p in lora)/1e6:.1f}M  head {sum(p.numel() for p in head)/1e6:.1f}M'
          f'  train {len(train_rows)}  val {len(val_rows)}  device {device}', flush=True)

    optimizer = torch.optim.AdamW([{'params': lora, 'lr': args.lr_lora},
                                   {'params': head, 'lr': args.lr_head}], weight_decay=0.0)
    length = lambda r: len(r['state']) + len(r['question']) + sum(len(o['desc']) for o in r['options'])
    batches = []
    for epoch in range(math.ceil(args.epochs)):
        batches += data.length_batches(train_rows, length, args.batch, args.seed + epoch)
    batches = batches[:int(args.epochs * len(train_rows) / args.batch)]
    total_steps = max(1, len(batches) // args.accum)
    warmup = max(1, int(args.warmup * total_steps))
    schedule = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def run_eval(step):
        model.eval()
        result = metrics.evaluate(val_rows, predict(model, val_rows, args.batch * 2))
        model.train()
        summary = {k: result[k] for k in ('score', 'accuracy', 'ece', 'brier', 'nll')}
        print(f'[eval step {step}] ' + '  '.join(f'{k} {v:.4f}' for k, v in summary.items()), flush=True)
        with (out / 'eval.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **summary, 'capabilities': result['capabilities']}) + '\n')
        return result['score']

    best = run_eval(0)
    model.save(out / 'best')
    model.train()
    step, t0, running = 0, time.time(), {}
    for i, batch in enumerate(batches):
        rows = [train_rows[j] for j in batch]
        logits, _ = model(rows)
        loss, parts = distillation_loss(logits, rows, kappa=args.kappa, gold_weight=args.gold_weight)
        (loss / args.accum).backward()
        for k, v in (parts | {'loss': loss.item()}).items():
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
            per_step = (time.time() - t0) / step
            memory = (f' peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB'
                      if device.type == 'cuda' else '')
            print(f'step {step}/{total_steps} ' + ' '.join(f'{k} {v/n:.4f}' for k, v in running.items())
                  + f' lr {scheduler.get_last_lr()[0]:.2e} {per_step:.2f}s/step'
                  + f' eta {(total_steps - step) * per_step / 3600:.2f}h{memory}', flush=True)
            running = {}
        if args.max_steps and step >= args.max_steps:
            break
        if step % args.eval_every == 0 or step == total_steps:
            score = run_eval(step)
            if score > best:
                best = score
                model.save(out / 'best')
    model.save(out / 'last')
    print(f'done. best val score {best:.2f}')


if __name__ == '__main__':
    main()
