# Mamba-2 130M CI-safe forgetting pilot (CPU-only)

This repository is a **minimal, reproducible pilot** to test whether short-context micro-CPT hurts long-context passkey retrieval in a pure Mamba-2 130M model, and whether mixed-length replay mitigates that harm.

## Hypothesis

1. Short-only micro-CPT on `seq_len=256` can reduce retrieval accuracy at longer contexts.
2. Adding mixed-length replay during the same micro-CPT budget can reduce that degradation.

## Compute assumptions (hard constraints)

- Runner: GitHub-hosted `ubuntu-latest`
- Budget: 4 vCPU, 16 GB RAM, 14 GB SSD
- Acceleration: **CPU-only**, no GPU
- Hard timeout ceiling: **< 6 hours** (workflow uses `timeout-minutes: 330`)
- Training budget per condition:
  - max 80 optimizer steps
  - batch size 1
  - train sequence length 256

## Why this fits CI limits

- Uses only one model family and one task (passkey retrieval).
- Uses tiny sample counts (default 3 examples per length).
- Uses parameter-efficient fallback (`lm_head_only`) for CPU feasibility.
- Saves checkpoints/metrics incrementally for resume on cache hits.
- Uses runtime guardrails with staged downscaling when projected runtime is too high.

## Runtime budget by stage (target: ~2–4h)

Configured estimates in `config.yaml`:

- baseline eval: 35 min
- short micro-CPT: 55 min
- eval post-short: 35 min
- replay micro-CPT: 55 min
- eval post-replay: 35 min
- report: 10 min

Total estimate: ~225 min (~3.75h), with 6h hard ceiling.

## Runtime guard behavior

After each major stage, runtime projection is recomputed. If risk is high, scaling is applied in this order:

1. reduce eval examples per length from 3 -> 2
2. skip context length 8192
3. reduce training steps from 80 -> 60

If projected runtime still exceeds the hard budget after all reductions, the run fails loudly.

## Project structure

- `config.yaml`: all experiment constants
- `src/mamba2_pilot/`
  - `data.py`: passkey sample generation
  - `modeling.py`: checkpoint/model loading + trainable-parameter mode
  - `train.py`: micro-CPT loop
  - `replay.py`: mixed-length replay policy
  - `eval.py`: passkey retrieval evaluation
  - `plotting.py`: final accuracy-vs-context plot
  - `runtime_guard.py`: runtime risk controls
- `scripts/`
  - `run_baseline_eval.py`
  - `run_micro_cpt.py`
  - `run_eval.py`
  - `make_report.py`
- `.github/workflows/ci.yml`: end-to-end pipeline

## Usage

```bash
make setup
make baseline
make train_short
make eval_short
make train_replay
make eval_replay
make report
```

Or run everything:

```bash
make ci
```

## Outputs

- JSON metrics in `artifacts/metrics/`
- final markdown summary in `artifacts/report/summary.md`
- final machine-readable verdict in `artifacts/report/final_summary.json`
- PNG plot in `artifacts/report/accuracy_vs_context.png`

The final summary explicitly states:
- whether short-only CPT reduced long-context retrieval
- whether replay helped relative to short-only CPT
