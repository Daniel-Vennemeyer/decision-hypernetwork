#!/usr/bin/env bash
# One-time setup on a single-GPU Linux CUDA machine: environment, fast Qwen3.5 kernels, data, weights.
# Everything is written under $WORK (default /data/jiang/vennemdp/decision-hypernetwork, see env.sh).
#   bash scripts/setup_cuda.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "WORK=$WORK"

PY=${PY:-python3}
REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a   # Qwen3.5-4B revision used by SII-Decision

# 1. environment ------------------------------------------------------------------------------
if [ ! -d "$VENV" ]; then $PY -m venv "$VENV"; fi
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
