"""Paired comparison of two evaluations on the same rows (uses the .preds.jsonl files that
evaluate.py writes next to each result).

    python compare.py runs/base-qwen3.5-4b-val.json runs/joint-qwen3.5-4b-val.json

Accuracy is over hard-labelled choice/yes-no rows (as in the score); NLL over all rows. Reports the
paired difference B - A with a 95% bootstrap interval (resampling rows), an exact McNemar test on
the answers that changed, and the sources that moved most."""
import argparse
import collections
import json
import math
import random
from pathlib import Path


def load(path):
    path = Path(path)
    preds = path.with_name(path.stem + '.preds.jsonl')
    with preds.open() as f:
        return {r['id']: r for r in map(json.loads, f)}


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def mcnemar_p(b_only, a_only):
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a', help='result JSON of the reference run')
    ap.add_argument('b', help='result JSON of the run to compare')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--top', type=int, default=10, help='sources to list')
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    if len(ids) < max(len(a), len(b)):
        print(f'note: comparing the {len(ids)} rows present in both ({len(a)} / {len(b)})')
    hard, nll, by_source = [], [], collections.defaultdict(lambda: [0, 0, 0])
    for i in ids:
        ra, rb = a[i], b[i]
        gold = ra['gold']
        nll.append(-sum(y * math.log(max(p, 1e-30)) for y, p in zip(gold, rb['probs']))
                   + sum(y * math.log(max(p, 1e-30)) for y, p in zip(gold, ra['probs'])))
        if ra['type'] == 'score' or max(gold) < 1 - 1e-6:
            continue
        g = argmax(gold)
        ca, cb = int(argmax(ra['probs']) == g), int(argmax(rb['probs']) == g)
        hard.append((ca, cb))
        s = by_source[ra['source']]
        s[0] += 1; s[1] += ca; s[2] += cb

    rng = random.Random(0)

    def interval(values):
        means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(args.bootstrap))
        return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]

    diffs = [cb - ca for ca, cb in hard]
    acc_a = 100 * sum(ca for ca, _ in hard) / len(hard)
    acc_b = 100 * sum(cb for _, cb in hard) / len(hard)
    lo, hi = interval(diffs)
    b_only = sum(1 for ca, cb in hard if cb and not ca)
    a_only = sum(1 for ca, cb in hard if ca and not cb)
    print(f'accuracy  A {acc_a:.2f}  B {acc_b:.2f}  diff {acc_b - acc_a:+.2f} pts  '
          f'95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  ({len(hard)} hard-label rows)')
    print(f'          answers changed: {b_only} now right, {a_only} now wrong; '
          f'McNemar p = {mcnemar_p(b_only, a_only):.3g}')
    lo, hi = interval(nll)
    print(f'NLL       diff {sum(nll) / len(nll):+.4f} per row (B - A, lower is better)  '
          f'95% CI [{lo:+.4f}, {hi:+.4f}]  ({len(nll)} rows)')
    print(f'\nsources with the largest accuracy change (n, A, B):')
    moved = sorted(by_source.items(), key=lambda kv: -abs(kv[1][2] - kv[1][1]))[:args.top]
    for source, (n, ca, cb) in moved:
        print(f'  {source:50s} n={n:4d}  {100 * ca / n:5.1f} -> {100 * cb / n:5.1f}  ({cb - ca:+d} rows)')


if __name__ == '__main__':
    main()
