"""Reasoning-to-readout distillation loss (SII-Decision, arXiv 2610.03935), batched.

    loss = w_type * [ CE(teacher, p) + 0.5 * 1[score] * ((E_p[level] - E_t[level]) / (K-1))^2 ]
           + kappa * KL(p0 || p)

teacher = averaged thinking-mode answer distribution of the same base model (train_target),
p0      = the base model's no-thinking readout, used as an anchor.
"""
import torch

TYPE_WEIGHTS = {'choice': 1.0, 'noul': 1.1666667, 'score': 0.8}


def _pad(rows, field, width):
    out = torch.zeros(len(rows), width)
    for i, row in enumerate(rows):
        values = row.get(field)
        if values is not None:
            out[i, :len(values)] = torch.tensor(values)
    return out


def targets(rows, width):
    """CPU tensors for one batch (built in HyperHead.prepare, moved to the GPU in one go)."""
    return dict(
        teacher=_pad(rows, 'train_target', width),
        gold=_pad(rows, 'gold_target', width),
        p0=_pad(rows, 'p0', width),
        n=torch.tensor([len(r['options']) for r in rows], dtype=torch.float),
        weight=torch.tensor([TYPE_WEIGHTS[r['type']] for r in rows]),
        is_score=torch.tensor([float(r['type'] == 'score') for r in rows]),
        hard_noul=torch.tensor([r['type'] == 'noul' and max(r['gold_target']) >= 1 - 1e-6 for r in rows]),
    )


def distillation_loss(logits, t, kappa=0.05, kappa_hard_noul=0.2, gold_weight=0.0):
    """t: targets(...) on the logits' device. Returns (loss, parts); parts are detached GPU
    tensors so logging does not force a GPU sync every step.

    parts['gain'] = CE(teacher, p0) - CE(teacher, p): how much closer to the teacher than the base
    model's own no-thinking readout, on the same rows. 0 at the start of joint training,
    positive = learning. Unlike the raw CE it does not swing with batch difficulty."""
    logp = logits.float().log_softmax(-1)
    teacher = t['teacher']
    if gold_weight:
        teacher = (1 - gold_weight) * teacher + gold_weight * t['gold']
    levels = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float)

    ce = -(teacher * logp).sum(-1)
    ordinal = 0.5 * t['is_score'] * (((logp.exp() - teacher) * levels).sum(-1)
                                     / (t['n'] - 1).clamp_min(1)).square()
    k = torch.where(t['hard_noul'], torch.full_like(ce, kappa_hard_noul), torch.full_like(ce, kappa))
    p0 = t['p0']
    kl = k * (p0 * (p0.clamp_min(1e-30).log() - logp)).sum(-1)
    total = t['weight'] * (ce + ordinal) + kl
    ce_p0 = -(teacher * p0.clamp_min(1e-30).log()).sum(-1)
    parts = dict(ce=ce.mean(), ordinal=ordinal.mean(), kl=kl.mean(), gain=(ce_p0 - ce).mean())
    return total.mean(), {name: value.detach() for name, value in parts.items()}
