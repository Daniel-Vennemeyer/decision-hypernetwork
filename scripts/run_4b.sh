#!/usr/bin/env bash
# Single-GPU HyperHead pipeline on Qwen3.5-4B. Stages can be picked with STAGES, e.g.
#   bash scripts/run_4b.sh                       # timing -> baseline -> joint -> eval
#   STAGES="split eval_split" bash scripts/run_4b.sh
#   BATCH=8 ACCUM=2 bash scripts/run_4b.sh                 # bigger micro-batch on 80 GB cards
#   BATCH=2 ACCUM=8 MAX_PADDED_CHARS=24000 bash scripts/run_4b.sh   # 24 GB cards
# Run it inside tmux/screen: the full joint run takes hours (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

BASE=${BASE:-Qwen/Qwen3.5-4B}
REVISION=${REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
BATCH=${BATCH:-4}            # micro-batch; effective batch = BATCH * ACCUM = 16 (as SII-Decision)
ACCUM=${ACCUM:-4}
EVAL_BATCH=${EVAL_BATCH:-8}
JEVAL_MAX_TOKENS=${JEVAL_MAX_TOKENS:-20480}   # JEVal's own input cap
MAX_PADDED_CHARS=${MAX_PADDED_CHARS:-64000}   # eval: batch x longest input; use ~24000 on 24 GB cards
STAGES=${STAGES:-"timing baseline joint eval_joint"}
mkdir -p runs logs
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train() {  # $1 = mode, $2 = out dir, rest = extra args
  local mode=$1 out=$2; shift 2
  python train.py --base "$BASE" --revision "$REVISION" --mode "$mode" --out "$out" \
    --batch "$BATCH" --accum "$ACCUM" --eval-rows 1000 --eval-every 250 "$@"
}
evaluate() {  # $1 = checkpoint dir or "base", $2 = name
  local args=(--checkpoint "$1"); [ "$1" = base ] && args=(--base "$BASE")
  python evaluate.py "${args[@]}" --dataset val --batch "$EVAL_BATCH" \
    --max-padded-chars "$MAX_PADDED_CHARS" --out "runs/$2-val.json"
  python evaluate.py "${args[@]}" --dataset jeval --batch "$EVAL_BATCH" \
    --max-tokens "$JEVAL_MAX_TOKENS" --max-padded-chars "$MAX_PADDED_CHARS" --out "runs/$2-jeval.json"
}

for stage in $STAGES; do
  echo "=== $stage  $(date)"
  case $stage in
    # ~40 optimizer steps: prints s/step, ETA for the full epoch and peak memory, then stops
    timing)     train joint runs/timing --max-steps 40 --log-every 10 --eval-rows 50 2>&1 | tee logs/timing.log ;;
    baseline)   evaluate base base 2>&1 | tee logs/baseline.log ;;
    joint)      train joint runs/joint-4b 2>&1 | tee logs/joint.log ;;
    eval_joint) evaluate runs/joint-4b/best joint 2>&1 | tee logs/eval_joint.log ;;
    split)      train split runs/split-4b 2>&1 | tee logs/split.log ;;
    eval_split) evaluate runs/split-4b/best split 2>&1 | tee logs/eval_split.log ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
