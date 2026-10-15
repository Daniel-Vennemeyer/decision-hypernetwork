"""Checks that torch sees the GPU, that Qwen3.5's linear-attention layers will use the
flash-linear-attention kernel, and that the kernel runs forward + backward and matches
transformers' torch reference. Exit code 0 on success, 1 on failure.

    python scripts/check_gpu.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import transformers
    print('torch', torch.__version__, '| built for CUDA', torch.version.cuda,
          '| transformers', transformers.__version__)
    if not torch.cuda.is_available():
        print('FAIL: CUDA not visible to torch (driver too old for this torch build?)')
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}, {props.total_memory / 2**30:.0f} GiB, sm_{props.major}{props.minor}')

    from hyperhead.kernels import linear_attention_backend
    backend = linear_attention_backend()
    print('Qwen3.5 gated delta rule kernel:', backend['delta_rule'])
    print('causal-conv1d:', 'yes' if backend['causal_conv1d'] else 'no (torch conv1d fallback)')
    if not backend['fla']:
        print('FAIL: Qwen3.5 layers would not use flash-linear-attention; training would be very slow')
        return 1

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    torch.manual_seed(0)
    B, T, H, D = 2, 200, 4, 64
    q, k, v = (torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3))
    g = -torch.rand(B, T, H, device='cuda').mul(0.5)
    beta = torch.rand(B, T, H, device='cuda', dtype=torch.bfloat16).sigmoid()
    try:
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        out.float().pow(2).sum().backward()
        torch.cuda.synchronize()
    except Exception as error:  # Triton compile / launch failures surface here
        print(f'FAIL: fla kernel did not run: {type(error).__name__}: {error}')
        return 1
    ref, _ = torch_chunk_gated_delta_rule(q.detach(), k.detach(), v.detach(), g=g, beta=beta,
                                          use_qk_l2norm_in_kernel=True)
    err = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    grads_ok = all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    print(f'fla gated delta rule: relative max error vs torch reference {err:.2e}, '
          f'gradients finite: {grads_ok}')
    if err > 5e-2 or not grads_ok:
        print('FAIL: fla kernel output looks wrong on this GPU')
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
