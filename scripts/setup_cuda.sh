#!/usr/bin/env bash
# One-time setup on a single-GPU Linux CUDA machine: environment, fast Qwen3.5 kernels, data, weights.
# Everything is written under $WORK (default /data/jiang/vennemdp/decision-hypernetwork, see env.sh).
#   bash scripts/setup_cuda.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "WORK=$WORK"

REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a   # Qwen3.5-4B revision used by SII-Decision

# 1. environment ------------------------------------------------------------------------------
#    transformers 5.x (needed for Qwen3.5) requires Python >= 3.10. Use $PY if given, else the
#    first python3.12/3.11/3.10 on PATH, else build Python 3.12 with conda under $WORK/.python.
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }
if [ -z "${PY:-}" ]; then
  for candidate in "$WORK/.python/bin/python" python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && py_ok "$candidate"; then PY=$candidate; break; fi
  done
fi
if [ -z "${PY:-}" ]; then
  command -v conda >/dev/null || { echo "Need Python >= 3.10: set PY=/path/to/python3.1x"; exit 1; }
  echo "No Python >= 3.10 found; creating one with conda in $WORK/.python"
  conda create -y -p "$WORK/.python" python=3.12
  PY=$WORK/.python/bin/python
fi
py_ok "$PY" || { echo "$PY is older than 3.10"; exit 1; }
echo "using $PY ($("$PY" --version))"

# an environment built by an earlier run on an older Python is rebuilt
if [ -x "$VENV/bin/python" ] && ! py_ok "$VENV/bin/python"; then
  echo "removing $VENV ($("$VENV/bin/python" --version 2>&1)); rebuilding with $PY"
  rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then "$PY" -m venv "$VENV"; fi
source "$VENV/bin/activate"
pip install -U pip wheel packaging ninja
pip install "torch>=2.7"                       # PyPI Linux wheels ship with CUDA 12.x
pip install -r requirements.txt huggingface_hub

# 2. fast kernels for Qwen3.5's Gated DeltaNet layers -------------------------------------------
#    Required: flash-linear-attention (Triton, pure pip). It replaces the expensive part, the
#    chunked gated delta rule.
#    Optional: causal-conv1d. It only replaces a short depthwise conv; without it transformers
#    uses torch's conv1d at a small cost. It has prebuilt wheels only for some torch/CUDA
#    combinations and otherwise compiles, which fails when the system nvcc differs from torch's
#    CUDA (e.g. nvcc 12.0 vs a CUDA 13.0 torch). So it is best-effort; INSTALL_CAUSAL_CONV1D=0 skips it.
pip install "flash-linear-attention>=0.2.2"
if [ "${INSTALL_CAUSAL_CONV1D:-1}" = 1 ]; then
  pip install causal-conv1d --no-build-isolation \
    || echo "NOTE: causal-conv1d did not install; continuing with torch conv1d (minor slowdown)."
fi

# check the fla kernel on the GPU: forward + backward against transformers' torch reference
python - <<'EOF'
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
assert torch.cuda.is_available(), 'CUDA not visible to torch'
print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB',
      '| torch', torch.__version__, 'CUDA', torch.version.cuda)
print('causal-conv1d:', 'yes' if m.causal_conv1d_fn is not None else 'no (torch conv1d fallback)')
assert m.chunk_gated_delta_rule is not None, 'flash-linear-attention not importable; training would be very slow'

torch.manual_seed(0)
B, T, H, D = 2, 200, 4, 64
q, k, v = (torch.randn(B, T, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
g = -torch.rand(B, T, H, device='cuda').mul(0.5)
beta = torch.rand(B, T, H, device='cuda', dtype=torch.bfloat16).sigmoid()
out, _ = m.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
out.float().pow(2).sum().backward()
ref, _ = m.torch_chunk_gated_delta_rule(q.detach(), k.detach(), v.detach(), g=g, beta=beta,
                                        use_qk_l2norm_in_kernel=True)
err = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
grads_ok = all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
print(f'fla gated delta rule: relative max error vs torch reference {err:.2e}, gradients finite: {grads_ok}')
assert err < 5e-2 and grads_ok, 'fla kernel output looks wrong on this GPU'
EOF

# 3. data (~100 MB) -----------------------------------------------------------------------------
for f in train.jsonl validation.jsonl; do
  [ -f "$DATA_DIR/$f" ] || curl -L --fail -o "$DATA_DIR/$f" \
    https://huggingface.co/datasets/jylin001206/SII-Decision-4B-Training-Data/resolve/main/$f
done
[ -f "$DATA_DIR/jeval_test.parquet" ] || curl -L --fail -o "$DATA_DIR/jeval_test.parquet" \
  https://huggingface.co/datasets/carlosxiang/JEVal/resolve/main/data/test.parquet
wc -l "$DATA_DIR"/*.jsonl

# 4. weights (~9 GB, into $HF_HOME) ---------------------------------------------------------------
python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B', revision='$REVISION'))"
du -sh "$WORK"
echo "setup done. next: bash scripts/run_4b.sh"
