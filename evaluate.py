"""Evaluate a HyperHead checkpoint (or the untrained base readout) on SII validation or JEVal.

    python evaluate.py --checkpoint runs/joint-4b/best --dataset jeval
    python evaluate.py --base Qwen/Qwen3.5-4B --dataset jeval        # frozen answer-slot baseline
"""
import argparse
import json
from pathlib import Path

import torch

from hyperhead import data, metrics


@torch.no_grad()
def predict(model, rows, batch_size=8, max_padded_chars=64000):
    # characters as a length proxy: ~1 token/char for Chinese, ~4 chars/token for English
    length = lambda r: len(data.text(r['state'])) + sum(len(o['desc']) for o in r['options'])
    from hyperhead.model import prefetch

    probs = [None] * len(rows)
    order = data.length_batches(rows, length, batch_size, 0, shuffle=False, max_padded=max_padded_chars)
    prepared = prefetch(model.prepare, ([rows[i] for i in b] for b in order))
    for batch, ready in zip(order, prepared):  # next batch is tokenized while this one runs
        logits, _ = model(ready)
        for i, p in zip(batch, logits.float().softmax(-1).tolist()):
            probs[i] = p[:len(rows[i]['options'])]
    return probs


def main():
    from hyperhead.model import HyperHead
    from train import pick_device

    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', help='directory written by HyperHead.save')
    ap.add_argument('--base', default='Qwen/Qwen3.5-4B', help='used without --checkpoint')
    ap.add_argument('--dataset', choices=['val', 'jeval'], default='val')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=None,
                    help='input budget; JEVal inputs go up to 20,480 tokens (default: checkpoint value / 4096)')
    ap.add_argument('--max-padded-chars', type=int, default=64000, help='caps batch x longest input')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = pick_device(args.device)
    overrides = {'max_tokens': args.max_tokens} if args.max_tokens else {}
    if args.checkpoint:
        model = HyperHead.load(args.checkpoint, **overrides)
    else:  # zero-initialised head in joint mode == the frozen answer-slot readout
        model = HyperHead(args.base, mode='joint', gradient_checkpointing=False, **overrides)
    model.to(device).eval()

    if args.dataset == 'jeval':
        import pandas as pd
        rows = data.convert_jeval(pd.read_parquet(Path(args.data_dir) / 'jeval_test.parquet'))
        if args.limit:
            import random
            rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    else:
        rows = data.read_jsonl(Path(args.data_dir) / 'validation.jsonl', args.limit, 0)

    result = metrics.evaluate(rows, predict(model, rows, args.batch, args.max_padded_chars))
    print(json.dumps({k: result[k] for k in ('n', 'score', 'accuracy', 'ece', 'brier', 'nll', 'capabilities')},
                     indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
