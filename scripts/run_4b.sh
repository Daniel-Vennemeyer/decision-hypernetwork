#!/usr/bin/env bash
# Single-GPU HyperHead pipeline on Qwen3.5-4B. All outputs go under $WORK (see env.sh):
#   $WORK/runs/<name>/{best,last}   checkpoints + eval.jsonl      $WORK/runs/<name>-{val,jeval}.json
#   $WORK/logs/<stage>.log
# Stages can be picked with STAGES, e.g.
#   bash scripts/run_4b.sh                       # timing -> baseline -> joint -> eval
#   STAGES="split eval_split" bash scripts/run_4b.sh
#   BATCH=8 ACCUM=2 bash scripts/run_4b.sh                 # bigger micro-batch on 80 GB cards
#   BATCH=2 ACCUM=8 MAX_PADDED_CHARS=24000 bash scripts/run_4b.sh   # 24 GB cards
# Run it inside tmux/screen: the full joint run takes hours (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source "$VENV/bin/activate"
echo "WORK=$WORK"

BASE=${BASE:-Qwen/Qwen3.5-4B}
REVISION=${REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
BATCH=${BATCH:-4}            # micro-batch; effective batch = BATCH * ACCUM = 16 (as SII-Decision)
ACCUM=${ACCUM:-4}
EVAL_BATCH=${EVAL_BATCH:-8}
JEVAL_MAX_TOKENS=${JEVAL_MAX_TOKENS:-20480}   # JEVal's own input cap
MAX_PADDED_CHARS=${MAX_PADDED_CHARS:-64000}   # eval: batch x longest input; use ~24000 on 24 GB cards
STAGES=${STAGES:-"timing baseline joint eval_joint"}
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train() {  # $1 = mode, $2 = run name, rest = extra args
  local mode=$1 name=$2; shift 2
  python train.py --base "$BASE" --revision "$REVISION" --mode "$mode" \
    --data-dir "$DATA_DIR" --out "$RUNS_DIR/$name" \
    --batch "$BATCH" --accum "$ACCUM" --eval-rows 1000 --eval-every 250 "$@"
}
evaluate() {  # $1 = run name or "base", $2 = result name
  local args=(--checkpoint "$RUNS_DIR/$1/best"); [ "$1" = base ] && args=(--base "$BASE")
  python evaluate.py "${args[@]}" --dataset val --data-dir "$DATA_DIR" --batch "$EVAL_BATCH" \
    --max-padded-chars "$MAX_PADDED_CHARS" --out "$RUNS_DIR/$2-val.json"
  python evaluate.py "${args[@]}" --dataset jeval --data-dir "$DATA_DIR" --batch "$EVAL_BATCH" \
    --max-tokens "$JEVAL_MAX_TOKENS" --max-padded-chars "$MAX_PADDED_CHARS" --out "$RUNS_DIR/$2-jeval.json"
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
