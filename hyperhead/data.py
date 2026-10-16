"""Decision rows, JEVal conversion and the three prompts used by HyperHead.

A decision row follows the SII-Decision schema:
    {id, type: choice|noul|score, state, question, options: [{key, desc}], gold_target,
     train_target?, p0?, source, capability?, evaluation_source?}

Three prompts are built from a row:
    spec prompt      - question + options only; read by the hypernetwork (LoRA on).
    joint prompt     - SII-Decision answer-slot prompt (state + question + lettered options).
    state prompt     - state only; the decision pass in the amortized "split" mode.
"""
import json
import random
import string

ANSWER_CUE = 'The correct answer is ('
STATE_CUE = 'The decision is'


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_jsonl(path, limit=None, seed=0):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def option_labels(tokenizer, count=255):
    candidates = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                                 for b in string.ascii_uppercase]
    labels = [c for c in candidates if len(tokenizer.encode(c, add_special_tokens=False)) == 1]
    return labels[:count]


def _chat(tokenizer, content):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def joint_prompt(tokenizer, row, labels):
    choices = '\n'.join(f'({labels[i]}) {o["key"]}: {o["desc"]}' for i, o in enumerate(row['options']))
    body = f'State:\n{text(row["state"])}\nQuestion: {text(row["question"])}\nOptions:\n{choices}'
    return _chat(tokenizer, body.rstrip()) + ANSWER_CUE


def state_prompt(tokenizer, row):
    return _chat(tokenizer, f'State:\n{text(row["state"])}') + STATE_CUE


def spec_prompt(tokenizer, row):
    """Returns the spec text and the character span of every option line.

    The spans let the hypernetwork mean-pool one embedding per option."""
    content = f'Decision type: {row["type"]}\nQuestion: {text(row["question"])}\nOptions:\n'
    spans = []
    for o in row['options']:
        line = f'- {o["key"]}: {o["desc"]}'
        spans.append((len(content), len(content) + len(line)))
        content += line + '\n'
    content += 'Build a decision head for this question.'
    full = _chat(tokenizer, content)
    offset = full.index(content)
    return full, [(a + offset, b + offset) for a, b in spans]


def convert_jeval(frame):
    """JEVal parquet rows -> decision rows with a one-hot gold_target."""
    rows = []
    for record in frame.to_dict('records'):
        (name, q), = json.loads(record['questions_json']).items()
        label = json.loads(record['label_json'])
        kind = q['type']
        criteria = q.get('criteria')
        if kind == 'noul':
            criteria = criteria or {'true': 'true', 'false': 'false'}
            options = [{'key': 'true', 'desc': criteria['true']},
                       {'key': 'false', 'desc': criteria['false']}]
            gold = 0 if int(label) == 1 else 1
        elif kind == 'score':
            options = [{'key': str(i), 'desc': c} for i, c in enumerate(criteria)]
            gold = int(label)
        else:
            # many label-set tasks (clinc150, ledgar, ...) give only keys: the key is the label
            options = [{'key': k, 'desc': k if v is None else text(v)} for k, v in criteria.items()]
            gold = [o['key'] for o in options].index(label)
        target = [0.0] * len(options)
        target[gold] = 1.0
        rows.append(dict(id=record['id'], type=kind, state=record['state'], question=q['instructions'],
                         options=options, gold_target=target, gold_value=gold if kind == 'score' else None,
                         source=record['source'], capability=record['source'],
                         evaluation_source=record['source']))
    return rows


def token_batches(spec_lens, decision_lens, max_rows, max_spec_tokens, max_decision_tokens,
                  seed=0, shuffle=True):
    """Micro-batches bounded by padded token counts (rows x longest row) for each pass.

    Memory is driven by the padded length of the hypernetwork (spec) pass, which carries
    gradients, so rows are sorted by spec length and then decision length. Batch order is
    shuffled. Returns lists of row indices."""
    order = sorted(range(len(spec_lens)), key=lambda i: (spec_lens[i], decision_lens[i]))
    batches, current, longest_decision = [], [], 0
    for i in order:
        n = len(current) + 1
        decision = max(longest_decision, decision_lens[i])
        if current and (n > max_rows or n * spec_lens[i] > max_spec_tokens
                        or n * decision > max_decision_tokens):
            batches.append(current)
            current, decision = [], decision_lens[i]
        current.append(i)
        longest_decision = decision
    if current:
        batches.append(current)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def per_source_sample(rows, per_source, seed=0):
    """Fixed stratified subset: at most per_source rows from every evaluation source, so a quick
    eval still covers every dataset and is identical across runs (paired comparisons)."""
    import collections
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row.get('evaluation_source', row['source'])].append(row)
    picked = []
    for source in sorted(groups):
        group = sorted(groups[source], key=lambda r: r['id'])
        picked += random.Random(f'{seed}:{source}').sample(group, min(per_source, len(group)))
    return picked
