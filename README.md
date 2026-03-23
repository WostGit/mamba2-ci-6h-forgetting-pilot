# Mamba-2 130M micro-CPT forgetting pilot (CPU-only CI)

This repository is a **minimal, reproducible research pilot** designed for GitHub Actions (Ubuntu hosted runner) to test:

1. Whether short-context micro-CPT hurts longer-context passkey retrieval in a pure **Mamba-2 130M** model.
2. Whether adding simple mixed-length replay during micro-CPT reduces that harm.

## Hypothesis

- **H1:** Short-only micro-CPT (256-token training only) decreases retrieval accuracy at longer contexts.
- **H2:** Short-only micro-CPT + mixed-length replay (mostly short, some long replay) reduces the drop.

## Compute assumptions (hard constraints)

- Runner type: GitHub-hosted `ubuntu-latest`
- Budget assumptions: 4 vCPU, 16 GB RAM, ~14 GB SSD
- No GPU
- PyTorch CPU only
- Max training budget per condition: 80 optimizer steps, batch size 1, seq len 256
- Evaluations: passkey retrieval at 1024/2048/4096/8192, 3 examples per length

## Why this should fit GitHub Actions limits

- Uses tiny synthetic passkey data generated on the fly (no large datasets).
- Uses a **parameter-efficient fallback** by default (`lm_head_only`) to avoid full-model CPU fine-tuning.
- Caches HF model files and adapter artifacts between runs.
- Saves intermediate JSON/model artifacts per stage so re-runs can skip finished work.
- Uses a staged runtime guard to auto-scale down if needed:
  1. reduce eval examples per length 3 → 2
  2. skip 8192-token evaluation
  3. reduce train steps 80 → 60
- Fails loudly if projected runtime still exceeds budget.

## Expected runtime budget (CPU CI estimate)

These are conservative planning numbers for CI safety:

- Dependency install: 10–20 min
- Baseline eval: 20–45 min
- Short-only micro-CPT: 35–70 min
- Replay micro-CPT: 35–70 min
- Final eval + report: 30–60 min
- **Total target:** ~2–4 hours (hard ceiling 6 hours)

## Repository layout

- `config/experiment_config.json` — all experiment constants.
- `src/mamba2_pilot/data.py` — passkey data generation.
- `src/mamba2_pilot/modeling.py` — model loading + trainable parameter selection.
- `src/mamba2_pilot/train.py` — micro-CPT loop.
- `src/mamba2_pilot/replay.py` — replay length sampling.
- `src/mamba2_pilot/eval.py` — passkey evaluation.
- `src/mamba2_pilot/plotting.py` — PNG accuracy plot.
- `scripts/run_baseline_eval.py`
- `scripts/run_micro_cpt.py`
- `scripts/run_eval.py`
- `scripts/make_report.py`
- `.github/workflows/ci.yml`

## Quickstart

```bash
make setup
make baseline
make train_short
make train_replay
make eval
make report
```

Or run the complete pipeline:

```bash
make ci
```

## Outputs

Expected artifacts under `outputs/`:

- `baseline_eval.json`
- `train_short.json`
- `train_replay.json`
- `all_eval_metrics.json`
- `summary.md`
- `accuracy_plot.png`

`summary.md` contains a single final statement on:
- whether short-only CPT reduced long-context retrieval
- whether replay helped relative to short-only CPT

## Notes

- The default config uses model id `state-spaces/mamba2-130m` with fallback `state-spaces/mamba-130m-hf` for robustness.
- This is intentionally not a full paper reproduction; it is a CI-safe feasibility pilot.
