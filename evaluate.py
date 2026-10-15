"""Evaluate a HyperHead checkpoint (or the untrained base readout) on SII validation or JEVal.

    python evaluate.py --checkpoint runs/joint-4b/best --dataset jeval
    python evaluate.py --base Qwen/Qwen3.5-4B --dataset jeval        # frozen answer-slot baseline
"""
import argparse
import json
import time
from pathlib import Path

import torch

from hyperhead import data, metrics


def spec_key(row):
    """Rows with the same key share a decision head (same type, question and options)."""
    return (row['type'], data.text(row['question']),
            json.dumps(row['options'], ensure_ascii=False, sort_keys=True))


@torch.no_grad()
def predict(model, rows, batch_size=8, max_spec_tokens=32768, max_decision_tokens=65536,
            verbose=False):
    """Probabilities per row, in row order.

    1. Hypernetwork pass once per distinct spec (question + options); heads are cached.
       Skipped when the head is still all-zero in joint mode (untrained baseline).
    2. Decision pass once per row, reusing the cached heads.
    Micro-batches are bounded by padded token counts, and the next batch is tokenized in a
    background thread while the current one runs."""
    from hyperhead.model import prefetch

    spec_lens, decision_lens = model.token_lengths(rows)
    heads = None
    if not (model.mode == 'joint' and model.head_is_zero()):
        first = {}
        for i, row in enumerate(rows):
            first.setdefault(spec_key(row), i)
        unique = list(first.values())
        order = data.token_batches([spec_lens[i] for i in unique], [0] * len(unique), batch_size,
                                   max_spec_tokens, 1, shuffle=False)
        batches = ([rows[unique[j]] for j in b] for b in order)
        cache = {}
        for b, ready in zip(order, prefetch(lambda rs: model.prepare(rs, decision=False), batches)):
            ready = model.to_device(ready)
            for j, head in zip(b, model.split_heads(model.generate_heads(ready), ready['n_options'])):
                cache[spec_key(rows[unique[j]])] = head
        heads = [cache[spec_key(row)] for row in rows]
    if verbose:
        print(f'{len(rows)} rows: hypernetwork pass ' + (
            'skipped (untrained head = frozen readout)' if heads is None
            else f'run once per distinct spec ({len(cache)} specs)')
            + f'; decision pass {sum(decision_lens) / 1e6:.2f}M tokens', flush=True)

    order = data.token_batches([0] * len(rows), decision_lens, batch_size, 1, max_decision_tokens,
                               shuffle=False)
    probs = [None] * len(rows)
    batches = ([rows[i] for i in b] for b in order)
    for b, ready in zip(order, prefetch(lambda rs: model.prepare(rs, spec=False), batches)):
        ready = model.to_device(ready)
        read = model.read_states(ready)
        if heads is None:
            logits = read['base']
        else:
            logits = model.decide(model.stack_heads([heads[i] for i in b], ready['k_max']), read)
        logits = model.mask_options(logits, ready['n_options'])
        for i, p in zip(b, logits.float().softmax(-1).tolist()):
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
    ap.add_argument('--max-decision-tokens', type=int, default=65536,
                    help='micro-batch cap: rows x longest decision prompt')
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

    started = time.time()
    probs = predict(model, rows, args.batch, max_decision_tokens=args.max_decision_tokens, verbose=True)
    print(f'predicted {len(rows)} rows in {time.time() - started:.0f}s', flush=True)
    result = metrics.evaluate(rows, probs)
    print(json.dumps({k: result[k] for k in ('n', 'score', 'accuracy', 'ece', 'brier', 'nll', 'capabilities')},
                     indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
