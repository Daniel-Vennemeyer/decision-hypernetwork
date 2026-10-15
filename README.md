# HyperHead: Qwen3.5-4B writes its own decision head

A Jev-style direct-decision model (state + question + options → a probability distribution in
one pass, no generated tokens) where the decision head itself is **generated per decision spec**
by the same backbone.

```
spec (question + options) ──► Qwen3.5-4B + LoRA ──► head generator ──► { w_k, b_k per option,
                                (hypernetwork pass)                     m pooling queries q }
state [+ spec]            ──► Qwen3.5-4B, frozen ──► hidden states at 3 depths ──┐
                                (decision pass, LoRA off)                        ▼
                                              attention-pool with q → logit_k = <w_k, u> + b_k
```

One copy of the weights serves both passes: the LoRA is switched on only for the hypernetwork.

## Two modes

| | `joint` | `split` (amortized) |
|---|---|---|
| decision pass sees | state + question + lettered options (SII-Decision prompt) | state only |
| logits | frozen answer-slot readout **+** generated head (zero-init → starts exactly at the base readout) | generated head only |
| cost for N specs × M states | N·M decision passes | M state passes + N cached heads; each extra decision is a dot product |
| expected strength | ≥ base readout; competes with SII-Decision LoRA | bi-encoder: good for fixed label sets (intents, moderation, ratings), weak for per-item MCQA |

In the SII training data, only ~27% of rows share a spec reused ≥10 times. Most rows are
one-off multiple-choice questions, which is why `joint` is the default.

## Data and objective

- **Train:** `jylin001206/SII-Decision-4B-Training-Data` (29,326 rows). Each row carries
  `train_target`, the averaged answer distribution of Qwen3.5-4B *with thinking*, and `p0`, its
  no-thinking readout. The loss is SII's reasoning-to-readout distillation: type-weighted CE to
  the teacher, plus an ordinal term for score tasks, plus κ·KL(p0‖p) (`hyperhead/objective.py`).
- **Validation:** the SII validation split (7,506 rows, gold labels, capability-macro score).
- **Test:** JEVal (`carlosxiang/JEVal`, 11,257 rows), converted by `data.convert_jeval`.
- **Metrics:** SII score, accuracy, ECE, Brier, NLL, and ordinal scale utilization (effective
  number of predicted vs gold labels per score source; arXiv 2609.38827).

The teacher targets come from Qwen3.5-4B. With any other base, they are off-policy.

## Run on a single CUDA GPU

```bash
git clone https://github.com/Daniel-Vennemeyer/decision-hypernetwork.git ~/decision-hypernetwork
cd ~/decision-hypernetwork
bash scripts/setup_cuda.sh   # venv, torch, fla + causal-conv1d kernels (checked), data, weights
tmux new -s hyperhead        # the run takes hours
bash scripts/run_4b.sh       # timing -> baseline eval -> joint training -> eval
```

The repo (code only) lives in `~/`. Everything the scripts write goes under `$WORK`, which
defaults to `/data/jiang/vennemdp/decision-hypernetwork` (set in `scripts/env.sh`; override with
`WORK=/other/dir`):

| `$WORK/...` | contents |
|---|---|
| `.venv/` | Python environment |
| `.cache/huggingface/` | Qwen3.5-4B weights (~9 GB) |
| `.cache/` (pip, triton, inductor, torch, nv, pycache) | tool caches, including `__pycache__` |
| `tmp/` | temporary files |
| `data/` | SII-Decision training data, JEVal |
| `runs/` | checkpoints (`<run>/best`, `<run>/last`), `eval.jsonl`, result JSONs |
| `logs/` | one log per stage |

The scripts redirect every cache they know of there, so nothing should land in the repo or
elsewhere in your home directory. To run `train.py` or `evaluate.py` by hand with the same
locations, first run `source scripts/env.sh && source $VENV/bin/activate`, and pass
`--data-dir $DATA_DIR --out $RUNS_DIR/...`.

`scripts/run_4b.sh` runs stages in order (choose them with `STAGES=...`):

| stage | what it does |
|---|---|
| `timing` | 40 optimizer steps; logs s/step, the ETA for a full epoch, and peak GPU memory |
| `baseline` | frozen base readout on SII validation and JEVal (`$WORK/runs/base-*.json`) |
| `joint` / `split` | one epoch, effective batch 16 (`$WORK/runs/joint-4b/best`) |
| `eval_joint` / `eval_split` | the trained head on validation and JEVal |

Settings by GPU memory: on 80 GB use `BATCH=8 ACCUM=2`. On 24 GB use
`BATCH=2 ACCUM=8 MAX_PADDED_CHARS=24000`. The default (`BATCH=4`) targets 40–48 GB.

Python 3.10 or newer is required (transformers 5.x). `setup_cuda.sh` uses `$PY` if you set
it, otherwise the first `python3.12`/`3.11`/`3.10` on PATH, otherwise it creates Python 3.12
with conda in `$WORK/.python`. An existing `.venv` built on an older Python is rebuilt.

Of the two Qwen3.5 kernel packages, only flash-linear-attention is required: it runs the
expensive gated delta rule, and setup checks it on the GPU against the torch reference
(forward and backward). causal-conv1d is optional. It only replaces a short convolution, and
without it transformers uses torch's conv1d. It ships prebuilt wheels for only some torch/CUDA
combinations; elsewhere it compiles, which fails when the system nvcc differs from torch's CUDA
(e.g. nvcc 12.0 vs PyPI's CUDA 13.0 torch). Setup then prints a note and continues.
`INSTALL_CAUSAL_CONV1D=0` skips it.

### Expected time

These are estimates from FLOP counts on the measured token totals, not measurements. The
`timing` stage prints the real number in its first minutes.

- **Token totals.** Training data: 5.2M tokens through the decision pass (joint prompts, mean
  178 tokens) and 3.5M through the hypernetwork pass (spec prompts, mean 120). JEVal: 23.5M
  tokens (mean 2,090, p95 13k, 1,702 rows over 4,096 tokens), so evaluation is
  long-context-heavy.
- **Compute.** About 1.2e17 FLOPs per training epoch. Short sequences keep GPU utilisation
  low, so I assumed 10–20% of bf16 peak during training.

| GPU | joint, 1 epoch | full eval (val + JEVal), each | default pipeline |
|---|---|---|---|
| H100 80 GB | ~1 h (40–90 min) | ~30–45 min | ~2.5–3 h |
| A100 80 GB | ~1.5–2.5 h | ~45–75 min | ~4–5 h |
| L40S / RTX 4090 | ~3–5 h | ~1.5–2 h | ~7–9 h |

`split` training costs about 80% of `joint`. Without flash-linear-attention, everything is several times
slower; `train.py` prints a warning in that case.

## Run locally (CPU / Apple Silicon)

```bash
pip install -r requirements.txt

# data (≈100 MB)
mkdir -p data && cd data
curl -LO https://huggingface.co/datasets/jylin001206/SII-Decision-4B-Training-Data/resolve/main/train.jsonl
curl -LO https://huggingface.co/datasets/jylin001206/SII-Decision-4B-Training-Data/resolve/main/validation.jsonl
curl -L -o jeval_test.parquet https://huggingface.co/datasets/carlosxiang/JEVal/resolve/main/data/test.parquet
cd ..

# baseline: frozen answer-slot readout (== untrained joint HyperHead)
python evaluate.py --base Qwen/Qwen3.5-4B --dataset jeval --out runs/base-jeval.json

# train (one epoch, as in SII-Decision)
python train.py --base Qwen/Qwen3.5-4B --mode joint --out runs/joint-4b --batch 4 --accum 4
python train.py --base Qwen/Qwen3.5-4B --mode split --out runs/split-4b --batch 4 --accum 4

python evaluate.py --checkpoint runs/joint-4b/best --dataset jeval --out runs/joint-jeval.json
```

On Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Comparisons that make the result meaningful

1. Base readout (above): the floor.
2. `jylin001206/SII-Decision-4B-LoRA`: same teacher, but a 112M-parameter LoRA on the decision
   model itself. Does a spec-conditioned head on a *frozen* decision pass match it?
3. A **static** head: the same reader and head, but trained as one fixed head and not generated
   per spec. This shows whether generating the head is doing anything.
4. `split` vs `joint` broken down by spec reuse (fixed label sets vs per-item MCQA), and the
   amortized throughput of `split` at N specs × M states.

## Status

Verified on Qwen3.5-0.8B on an M2 Max (MPS):
- Joint and split modes train end to end.
- Both overfit 16 rows (CE 2.2 → 0.08 and 1.5 → 0.07).
- JEVal conversion covers all 11,257 rows.
- Save/reload reproduces the training-time eval exactly.

Not yet run: anything at 4B, anything on CUDA (including `scripts/setup_cuda.sh`), and any
real-length training run.

Caveat: training inputs are at most 1,400 tokens, but 15% of JEVal rows exceed 4,096 tokens.
Long-context JEVal results test length generalization of the head as well.
