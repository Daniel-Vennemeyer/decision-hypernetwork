#!/usr/bin/env bash
# Single-GPU HyperHead pipeline on Qwen3.5-4B. All outputs go under $WORK (see env.sh):
#   $WORK/runs/<name>/{best,last}   checkpoints + eval.jsonl      $WORK/runs/<name>-{val,jeval}.json
#   $WORK/logs/<stage>.log
# Stages can be picked with STAGES, e.g.
#   bash scripts/run_4b.sh                       # timing -> baseline -> joint -> eval
#   STAGES="split eval_split" bash scripts/run_4b.sh
#   BATCH=8 ACCUM=2 bash scripts/run_4b.sh       # override the GPU-size preset (see below)
# Run it inside tmux/screen: the full joint run takes hours (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source "$VENV/bin/activate"
echo "WORK=$WORK"

BASE=${BASE:-Qwen/Qwen3.5-4B}
REVISION=${REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
# Presets by GPU memory (any variable set in the environment wins). Micro-batches are bounded
# by padded token counts: the spec (hypernetwork) pass carries gradients and dominates memory
# (~8 MB per padded token on Qwen3.5-4B without gradient checkpointing, measured on an A100).
#            rows  accum  grad ckpt  spec tokens  decision tokens  eval rows  eval decision tokens
#   >=70 GB   16     1      off         6144          16384           32          65536
#   >=40 GB    8     2      off         2560           8192           16          32768
#   smaller    4     4      on          2048           4096            8           8192
# Effective batch = rows x accum = 16 rows at most (micro-batches of long rows are smaller).
# If the first step's peak memory is close to the card's limit, lower MAX_SPEC_TOKENS or set
# NO_GRAD_CKPT=0.
GPU_GB=$(python -c 'import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))')
if   [ "$GPU_GB" -ge 70 ]; then P=(16 1 1 6144 16384 32 65536)
elif [ "$GPU_GB" -ge 40 ]; then P=(8 2 1 2560 8192 16 32768)
else                            P=(4 4 0 2048 4096 8 8192); fi
BATCH=${BATCH:-${P[0]}}
ACCUM=${ACCUM:-${P[1]}}
NO_GRAD_CKPT=${NO_GRAD_CKPT:-${P[2]}}
MAX_SPEC_TOKENS=${MAX_SPEC_TOKENS:-${P[3]}}
MAX_DECISION_TOKENS=${MAX_DECISION_TOKENS:-${P[4]}}
EVAL_BATCH=${EVAL_BATCH:-${P[5]}}
EVAL_DECISION_TOKENS=${EVAL_DECISION_TOKENS:-${P[6]}}
JEVAL_MAX_TOKENS=${JEVAL_MAX_TOKENS:-20480}   # JEVal's own input cap
echo "GPU ${GPU_GB} GiB: rows $BATCH x accum $ACCUM, no-grad-ckpt $NO_GRAD_CKPT, spec tokens $MAX_SPEC_TOKENS, decision tokens $MAX_DECISION_TOKENS, eval rows $EVAL_BATCH, eval decision tokens $EVAL_DECISION_TOKENS"
STAGES=${STAGES:-"timing baseline joint eval_joint"}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train() {  # $1 = mode, $2 = run name, rest = extra args
  local mode=$1 name=$2; shift 2
  python train.py --base "$BASE" --revision "$REVISION" --mode "$mode" \
    --data-dir "$DATA_DIR" --out "$RUNS_DIR/$name" \
    --batch "$BATCH" --accum "$ACCUM" --eval-rows 1000 --eval-every 250 \
    --max-spec-tokens "$MAX_SPEC_TOKENS" --max-decision-tokens "$MAX_DECISION_TOKENS" \
    $([ "$NO_GRAD_CKPT" = 1 ] && echo --no-grad-ckpt) "$@"
}
evaluate() {  # $1 = run name or "base", $2 = result name
  local args=(--checkpoint "$RUNS_DIR/$1/best"); [ "$1" = base ] && args=(--base "$BASE")
  python evaluate.py "${args[@]}" --dataset val --data-dir "$DATA_DIR" --batch "$EVAL_BATCH" \
    --max-decision-tokens "$EVAL_DECISION_TOKENS" --out "$RUNS_DIR/$2-val.json"
  python evaluate.py "${args[@]}" --dataset jeval --data-dir "$DATA_DIR" --batch "$EVAL_BATCH" \
    --max-tokens "$JEVAL_MAX_TOKENS" --max-decision-tokens "$EVAL_DECISION_TOKENS" --out "$RUNS_DIR/$2-jeval.json"
}

for stage in $STAGES; do
  echo "=== $stage  $(date)"
  log="$LOGS_DIR/$stage.log"
  case $stage in
    # ~40 optimizer steps: prints s/step, ETA for the full epoch and peak memory, then stops
    timing)     train joint timing --max-steps 40 --log-every 10 --eval-rows 50 2>&1 | tee "$log" ;;
    baseline)   evaluate base base 2>&1 | tee "$log" ;;
    joint)      train joint joint-4b 2>&1 | tee "$log" ;;
    eval_joint) evaluate joint-4b joint 2>&1 | tee "$log" ;;
    split)      train split split-4b 2>&1 | tee "$log" ;;
    eval_split) evaluate split-4b split 2>&1 | tee "$log" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
