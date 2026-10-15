"""Decision metrics: SII capability-macro score, accuracy, ECE, Brier, NLL, and ordinal
scale utilization (arXiv 2609.38827) for score tasks."""
import collections
import math


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def _effective_labels(indices):
    counts = collections.Counter(indices)
    total = sum(counts.values())
    return math.exp(-sum(c / total * math.log(c / total) for c in counts.values()))


def evaluate(rows, probs):
    """rows: decision rows with gold_target; probs: list of probability lists in row order."""
    per_source = collections.defaultdict(lambda: collections.defaultdict(list))
    capability = {}
    hard, bins, scores = [], [[0, 0.0, 0.0] for _ in range(10)], collections.defaultdict(lambda: ([], []))
    for row, p in zip(rows, probs):
        gold = row['gold_target']
        source = row.get('evaluation_source', row['source'])
        capability[source] = row.get('capability', source)
        bucket = per_source[source]
        bucket['nll'].append(-sum(y * math.log(max(q, 1e-30)) for y, q in zip(gold, p)))
        if row['type'] == 'score':
            value = row.get('gold_value')
            value = sum(i * y for i, y in enumerate(gold)) if value is None else value
            bucket['score_nmae'].append(abs(sum(i * q for i, q in enumerate(p)) - value) / (len(p) - 1))
            scores[source][0].append(_argmax(p))
            scores[source][1].append(_argmax(gold))
        elif row['type'] == 'noul' and max(gold) < 1 - 1e-6:
            bucket['soft_noul_mae'].append(abs(p[0] - gold[0]))
        else:
            correct = int(_argmax(p) == _argmax(gold))
            bucket['accuracy'].append(correct)
            confidence = max(p)
            b = bins[min(int(confidence * 10), 9)]
            b[0] += 1; b[1] += confidence; b[2] += correct
            hard.append((correct, sum((q - y) ** 2 for q, y in zip(p, gold))))

    sources = {}
    for source, bucket in per_source.items():
        means = {m: sum(v) / len(v) for m, v in bucket.items()}
        task = [v if m == 'accuracy' else 1 - v for m, v in means.items() if m != 'nll']
        sources[source] = dict(n=len(bucket['nll']), score=100 * sum(task) / len(task), **means)
    groups = collections.defaultdict(list)
    for source, value in sources.items():
        groups[capability[source]].append(value['score'])
    n_hard = max(len(hard), 1)
    return dict(
        n=len(rows),
        score=sum(sum(v) / len(v) for v in groups.values()) / len(groups),
        accuracy=100 * sum(c for c, _ in hard) / n_hard,
        brier=sum(b for _, b in hard) / n_hard,
        ece=sum(abs(b[1] - b[2]) for b in bins if b[0]) / n_hard,
        nll=sum(s['nll'] * s['n'] for s in sources.values()) / len(rows),
        scale_utilization={s: dict(pred_effective=_effective_labels(p), gold_effective=_effective_labels(g))
                           for s, (p, g) in scores.items()},
        capabilities={g: sum(v) / len(v) for g, v in groups.items()},
        sources=sources,
    )
