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

# 2. fast linear-attention kernels for Qwen3.5's Gated DeltaNet layers --------------------------
#    flash-linear-attention is Triton (pure pip). causal-conv1d fetches a prebuilt wheel matching
#    your torch/CUDA when one exists; otherwise it compiles, which needs nvcc (CUDA toolkit).
pip install "flash-linear-attention>=0.2.2"
pip install causal-conv1d --no-build-isolation

python - <<'EOF'
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
assert torch.cuda.is_available(), 'CUDA not visible to torch'
print('GPU:', torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB')
print('Qwen3.5 fast path:', m.is_fast_path_available)
assert m.is_fast_path_available, 'fla / causal-conv1d kernels not importable; training would be very slow'
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
