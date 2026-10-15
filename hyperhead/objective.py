"""Reasoning-to-readout distillation loss (SII-Decision, arXiv 2610.03935), batched.

    loss = w_type * [ CE(teacher, p) + 0.5 * 1[score] * ((E_p[level] - E_t[level]) / (K-1))^2 ]
           + kappa * KL(p0 || p)

teacher = averaged thinking-mode answer distribution of the same base model (train_target),
p0      = the base model's no-thinking readout, used as an anchor.
"""
import torch

TYPE_WEIGHTS = {'choice': 1.0, 'noul': 1.1666667, 'score': 0.8}


def _pad(rows, field, width, device):
    out = torch.zeros(len(rows), width, device=device)
    for i, row in enumerate(rows):
        values = row.get(field)
        if values is not None:
            out[i, :len(values)] = torch.tensor(values, device=device)
    return out


def distillation_loss(logits, rows, kappa=0.05, kappa_hard_noul=0.2, gold_weight=0.0):
    device, width = logits.device, logits.shape[-1]
    logp = logits.float().log_softmax(-1)
    teacher = _pad(rows, 'train_target', width, device)
    if gold_weight:
        teacher = (1 - gold_weight) * teacher + gold_weight * _pad(rows, 'gold_target', width, device)
    p0 = _pad(rows, 'p0', width, device)
    n = torch.tensor([len(r['options']) for r in rows], device=device, dtype=torch.float)
    weight = torch.tensor([TYPE_WEIGHTS[r['type']] for r in rows], device=device)
    is_score = torch.tensor([r['type'] == 'score' for r in rows], device=device, dtype=torch.float)
    levels = torch.arange(width, device=device, dtype=torch.float)

    ce = -(teacher * logp).sum(-1)
    ordinal = 0.5 * is_score * (((logp.exp() - teacher) * levels).sum(-1) / (n - 1).clamp_min(1)).square()
    k = torch.tensor([kappa_hard_noul if r['type'] == 'noul' and max(r['gold_target']) >= 1 - 1e-6
                      else kappa for r in rows], device=device)
    kl = k * (p0 * (p0.clamp_min(1e-30).log() - logp)).sum(-1)
    total = weight * (ce + ordinal) + kl
    return total.mean(), dict(ce=ce.mean().item(), ordinal=ordinal.mean().item(), kl=kl.mean().item())
