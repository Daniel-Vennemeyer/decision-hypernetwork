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

# 2. torch built for a CUDA version the NVIDIA driver supports -----------------------------------
#    PyPI's default torch targets the newest CUDA (13.0), which older drivers cannot run. Read the
#    driver's CUDA version from nvidia-smi and pick a matching build from download.pytorch.org:
#      driver >= 13.0 -> cu130 | 12.8-12.9 -> cu128 | 12.0-12.7 -> cu126 (via CUDA minor-version
#      compatibility; if the GPU check fails, setup falls back to torch 2.6.0 + cu124 for 12.4+).
#    Override with TORCH_CUDA=cu124 (or cu126, cu128, ...).
driver_cuda=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}' || true)
[ -n "$driver_cuda" ] || { echo "nvidia-smi did not report a CUDA version; is an NVIDIA driver installed?"; exit 1; }
if [ -z "${TORCH_CUDA:-}" ]; then
  case $driver_cuda in
    1[3-9].*) TORCH_CUDA=cu130 ;;
    12.[89]*) TORCH_CUDA=cu128 ;;
    12.*)     TORCH_CUDA=cu126 ;;
    *) echo "driver CUDA $driver_cuda is too old for current torch; update the driver"; exit 1 ;;
  esac
  AUTO_TORCH=1
fi
echo "driver supports CUDA $driver_cuda -> torch build $TORCH_CUDA"

install_torch() {  # $1 = cu126 etc.; replaces torch (and its triton) unless already that build
  local have
  have=$(python -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))' 2>/dev/null || true)
  if [ "$have" = "$1" ]; then echo "torch already built for $1"; return; fi
  pip uninstall -y torch triton pytorch-triton causal-conv1d >/dev/null 2>&1 || true
  pip install torch --index-url "https://download.pytorch.org/whl/$1"
}
install_torch "$TORCH_CUDA"
pip install -r requirements.txt huggingface_hub   # torch is already satisfied, so it stays as is

# 3. fast kernels for Qwen3.5's Gated DeltaNet layers -------------------------------------------
#    Required: flash-linear-attention (Triton, pure pip). It replaces the expensive part, the
#    chunked gated delta rule.
#    Optional: causal-conv1d. It only replaces a short depthwise conv; without it transformers
#    uses torch's conv1d at a small cost. Without a prebuilt wheel for your torch/CUDA it compiles
#    (slow, needs an nvcc matching torch's CUDA major version), so it is off by default;
#    INSTALL_CAUSAL_CONV1D=1 tries it.
pip install "flash-linear-attention>=0.2.2"
if [ "${INSTALL_CAUSAL_CONV1D:-0}" = 1 ]; then
  pip install causal-conv1d --no-build-isolation \
    || echo "NOTE: causal-conv1d did not install; continuing with torch conv1d (minor slowdown)."
fi

# GPU check (torch sees the GPU; fla kernel forward + backward matches the torch reference)
if ! python scripts/check_gpu.py; then
  if [ "${AUTO_TORCH:-0}" = 1 ] && [ "$TORCH_CUDA" = cu126 ] && \
     awk -v v="$driver_cuda" 'BEGIN { exit !(v >= 12.4 && v < 12.6) }'; then
    echo "GPU check failed with cu126 on a CUDA $driver_cuda driver; falling back to torch 2.6.0 + cu124"
    install_torch cu124
    python scripts/check_gpu.py || { echo "GPU check failed again; paste the output above"; exit 1; }
  else
    echo "GPU check failed; paste the output above"; exit 1
  fi
fi

# 4. data (~100 MB) -----------------------------------------------------------------------------
for f in train.jsonl validation.jsonl; do
  [ -f "$DATA_DIR/$f" ] || curl -L --fail -o "$DATA_DIR/$f" \
    https://huggingface.co/datasets/jylin001206/SII-Decision-4B-Training-Data/resolve/main/$f
done
[ -f "$DATA_DIR/jeval_test.parquet" ] || curl -L --fail -o "$DATA_DIR/jeval_test.parquet" \
  https://huggingface.co/datasets/carlosxiang/JEVal/resolve/main/data/test.parquet
wc -l "$DATA_DIR"/*.jsonl

# 5. weights (~9 GB, into $HF_HOME) ---------------------------------------------------------------
python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B', revision='$REVISION'))"
du -sh "$WORK"
echo "setup done. next: bash scripts/run_4b.sh"
