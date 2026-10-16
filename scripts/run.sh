#!/usr/bin/env bash
# Single-GPU HyperHead pipeline for any Qwen3.5 base model. All outputs go under $WORK (env.sh):
#   $WORK/runs/<mode>-<tag>[-<EXP>]/{best,last}         checkpoints (+ eval.jsonl)
#   $WORK/runs/<name>-<tag>[-quick]-{val,jeval}.json    results (+ .preds.jsonl per-row predictions)
#   $WORK/logs/<stage>-<tag>[-<EXP>].log
# where <tag> is the lowercased model name, e.g. qwen3.5-4b or qwen3.5-0.8b-base.
#
#   bash scripts/run.sh                                       # Qwen3.5-4B, full pipeline
#   BASE=Qwen/Qwen3.5-0.8B-Base QUICK=1 bash scripts/run.sh   # fast feedback loop
#   QUICK=1 EXP=lr1e3 TRAIN_ARGS="--lr-head 1e-3" bash scripts/run.sh   # a named variant
#   STAGES="split eval_split" bash scripts/run.sh
#
# QUICK=1: no evaluation during training (the final checkpoint is used), and evaluation on fixed
# stratified subsets (VAL_PER_SOURCE=25 rows per source -> ~1.1k validation rows,
# JEVAL_PER_SOURCE=30 -> ~1.7k JEVal rows), identical across runs so compare.py can pair them.
# The baseline (frozen readout) is evaluated once per model and reused; FORCE=1 recomputes it.
# After each evaluation the run is compared with the baseline (compare.py).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
source "$VENV/bin/activate"
echo "WORK=$WORK"

BASE=${BASE:-Qwen/Qwen3.5-4B}
if [ "$BASE" = Qwen/Qwen3.5-4B ]; then
  REVISION=${REVISION-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}   # revision used by SII-Decision
else
  REVISION=${REVISION:-}
fi
TAG=${TAG:-$(basename "$BASE" | tr '[:upper:]' '[:lower:]')}
EXP=${EXP:-}
QUICK=${QUICK:-0}
VAL_PER_SOURCE=${VAL_PER_SOURCE:-25}
JEVAL_PER_SOURCE=${JEVAL_PER_SOURCE:-30}
TRAIN_ARGS=${TRAIN_ARGS:-}
FORCE=${FORCE:-0}
if [ "$QUICK" = 1 ]; then
  STAGES=${STAGES:-"baseline joint eval_joint"}; SUFFIX=-quick
else
  STAGES=${STAGES:-"timing baseline joint eval_joint"}; SUFFIX=
fi

# Presets by GPU memory, for Qwen3.5-4B (any variable set in the environment wins). Token budgets
# bound padded tokens per micro-batch; the spec (hypernetwork) pass carries gradients and dominates
# memory (~8 MB per padded token on 4B without gradient checkpointing, measured on an A100).
# Smaller models get proportionally larger budgets (SCALE = 4B hidden x layers / model's, 1-8).
#            rows  accum  grad ckpt  spec tokens  decision tokens  eval rows  eval decision tokens
#   >=70 GB   16     1      off         6144          16384           32          65536
#   >=40 GB    8     2      off         2560           8192           16          32768
#   smaller    4     4      on          2048           4096            8           8192
read -r GPU_GB SCALE < <(python - "$BASE" "$REVISION" <<'EOF'
import sys, torch
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], revision=sys.argv[2] or None)
text = getattr(config, 'text_config', config)
scale = max(1.0, min(8.0, 2560 * 32 / (text.hidden_size * text.num_hidden_layers)))
print(int(torch.cuda.get_device_properties(0).total_memory / 2**30), f'{scale:.2f}')
EOF
)
if   [ "$GPU_GB" -ge 70 ]; then P=(16 1 1 6144 16384 32 65536)
elif [ "$GPU_GB" -ge 40 ]; then P=(8 2 1 2560 8192 16 32768)
else                            P=(4 4 0 2048 4096 8 8192); fi
scaled() { awk -v v="$1" -v s="$SCALE" 'BEGIN { printf "%d", v * s }'; }
BATCH=${BATCH:-${P[0]}}
ACCUM=${ACCUM:-${P[1]}}
NO_GRAD_CKPT=${NO_GRAD_CKPT:-${P[2]}}
MAX_SPEC_TOKENS=${MAX_SPEC_TOKENS:-$(scaled "${P[3]}")}
MAX_DECISION_TOKENS=${MAX_DECISION_TOKENS:-$(scaled "${P[4]}")}
EVAL_BATCH=${EVAL_BATCH:-${P[5]}}
EVAL_DECISION_TOKENS=${EVAL_DECISION_TOKENS:-$(scaled "${P[6]}")}
JEVAL_MAX_TOKENS=${JEVAL_MAX_TOKENS:-20480}   # JEVal's own input cap
echo "model $BASE (tag $TAG${EXP:+, exp $EXP}$([ "$QUICK" = 1 ] && echo ', quick')) | GPU ${GPU_GB} GiB, budget scale $SCALE: " \
     "rows $BATCH x accum $ACCUM, no-grad-ckpt $NO_GRAD_CKPT, spec tokens $MAX_SPEC_TOKENS," \
     "decision tokens $MAX_DECISION_TOKENS, eval rows $EVAL_BATCH, eval decision tokens $EVAL_DECISION_TOKENS"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_name() { echo "$1-$TAG${EXP:+-$EXP}"; }    # $1 = joint | split | timing -> checkpoint dir name
result() {  # $1 = base | joint | split, $2 = val | jeval -> result file
  if [ "$1" = base ]; then echo "$RUNS_DIR/base-$TAG$SUFFIX-$2.json"
  else echo "$RUNS_DIR/$(run_name "$1")$SUFFIX-$2.json"; fi
}

train() {  # $1 = mode, $2 = run name, rest = extra args
  local mode=$1 name=$2; shift 2
  local evals=(--eval-every 250 --eval-per-source "$VAL_PER_SOURCE")
  [ "$QUICK" = 1 ] && evals=(--eval-every 0)
  # shellcheck disable=SC2086
  python train.py --base "$BASE" ${REVISION:+--revision "$REVISION"} --mode "$mode" \
    --data-dir "$DATA_DIR" --out "$RUNS_DIR/$name" \
    --batch "$BATCH" --accum "$ACCUM" "${evals[@]}" \
    --max-spec-tokens "$MAX_SPEC_TOKENS" --max-decision-tokens "$MAX_DECISION_TOKENS" \
    $([ "$NO_GRAD_CKPT" = 1 ] && echo --no-grad-ckpt) "$@" $TRAIN_ARGS
}
evaluate() {  # $1 = base | joint | split
  local args=(--checkpoint "$RUNS_DIR/$(run_name "$1")/best")
  [ "$1" = base ] && args=(--base "$BASE" ${REVISION:+--revision "$REVISION"})
  local val_subset=() jeval_subset=()
  [ "$QUICK" = 1 ] && val_subset=(--per-source "$VAL_PER_SOURCE") && jeval_subset=(--per-source "$JEVAL_PER_SOURCE")
  python evaluate.py "${args[@]}" --dataset val --data-dir "$DATA_DIR" --batch "$EVAL_BATCH" \
    ${val_subset[@]+"${val_subset[@]}"} --max-decision-tokens "$EVAL_DECISION_TOKENS" --out "$(result "$1" val)"
  python evaluate.py "${args[@]}" --dataset jeval --data-dir "$DATA_DIR" --batch "$EVAL_BATCH" \
    ${jeval_subset[@]+"${jeval_subset[@]}"} --max-tokens "$JEVAL_MAX_TOKENS" --max-decision-tokens "$EVAL_DECISION_TOKENS" \
    --out "$(result "$1" jeval)"
}
compare_to_base() {  # $1 = joint | split
  for split in val jeval; do
    if [ -f "$(result base $split)" ] && [ -f "$(result "$1" $split)" ]; then
      echo "--- $split: $(run_name "$1") vs frozen readout ($TAG$SUFFIX)"
      python compare.py "$(result base $split)" "$(result "$1" $split)"
    fi
  done
}

for stage in $STAGES; do
  echo "=== $stage  $(date)"
  log="$LOGS_DIR/$stage-$TAG${EXP:+-$EXP}$SUFFIX.log"
  case $stage in
    # ~40 optimizer steps: prints s/step, ETA for the full epoch and peak memory, then stops
    timing)     train joint "$(run_name timing)" --max-steps 40 --log-every 10 --eval-every 0 2>&1 | tee "$log" ;;
    baseline)
      if [ "$FORCE" != 1 ] && [ -f "$(result base val)" ] && [ -f "$(result base jeval)" ]; then
        echo "baseline for $TAG$SUFFIX already evaluated; reusing (FORCE=1 recomputes)"
      else
        evaluate base 2>&1 | tee "$log"
      fi ;;
    joint)      train joint "$(run_name joint)" 2>&1 | tee "$log" ;;
    split)      train split "$(run_name split)" 2>&1 | tee "$log" ;;
    eval_joint) { evaluate joint; compare_to_base joint; } 2>&1 | tee "$log" ;;
    eval_split) { evaluate split; compare_to_base split; } 2>&1 | tee "$log" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
