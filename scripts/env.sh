# Sourced by setup_cuda.sh and run_4b.sh. The repo itself can live anywhere (e.g. ~/); everything
# the pipeline writes goes under $WORK instead: environment, caches, data, model weights,
# checkpoints, results and logs. Nothing is written into the repo or the home directory.
# Override with:  WORK=/some/other/dir bash scripts/run_4b.sh
export WORK=${WORK:-/data/jiang/vennemdp/decision-hypernetwork}

export VENV=$WORK/.venv
export DATA_DIR=$WORK/data
export RUNS_DIR=$WORK/runs
export LOGS_DIR=$WORK/logs
export HF_HOME=$WORK/.cache/huggingface        # Qwen3.5-4B weights (~9 GB)
export PIP_CACHE_DIR=$WORK/.cache/pip
export TRITON_CACHE_DIR=$WORK/.cache/triton    # fla compiles Triton kernels on first use
export TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/inductor
export XDG_CACHE_HOME=$WORK/.cache                # catch-all for tools defaulting to ~/.cache
export TORCH_HOME=$WORK/.cache/torch
export CUDA_CACHE_PATH=$WORK/.cache/nv            # CUDA JIT cache (default ~/.nv)
export PYTHONPYCACHEPREFIX=$WORK/.cache/pycache   # keeps __pycache__ out of the repo
export TMPDIR=$WORK/tmp

mkdir -p "$WORK" "$DATA_DIR" "$RUNS_DIR" "$LOGS_DIR" "$HF_HOME" "$PIP_CACHE_DIR" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" \
         "$PYTHONPYCACHEPREFIX" "$TMPDIR"
