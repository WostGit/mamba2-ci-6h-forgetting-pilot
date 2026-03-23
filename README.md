# Mamba-2 130M CI Pilot: short-context micro-CPT vs long-context retrieval

This repository is a **minimal, reproducible CI-safe pilot** designed to test one question:

> Does short-context micro-CPT hurt longer-context passkey retrieval in a pure Mamba-2 130M model, and does mixed-length replay reduce that harm?

## Hypothesis
1. Short-only micro-CPT (256-token training sequences) will reduce retrieval at long context lengths.
2. Adding mixed-length replay during the same micro-CPT budget will partially recover long-context retrieval.

## Compute assumptions (hard-coded design target)
- Runner: GitHub-hosted `ubuntu-latest`
- CPU only (no GPU)
- Budget: ~4 vCPU, 16 GB RAM, 14 GB SSD
- CI wall clock target: ~2–4 hours
- Hard ceiling: `< 6` hours (workflow timeout is 330 minutes)

## Why this should fit GitHub Actions limits
- Single model only: `state-spaces/mamba2-130m`
- Tiny training budget: max 80 optimizer steps, batch size 1, sequence length 256
- Passkey-only evaluation with tiny sample count
- Runtime guard auto-scales down in this order when time risk is detected:
  1. examples/length: 3 -> 2
  2. remove 8192 context
  3. steps: 80 -> 60
- Small artifacts only (JSON + markdown + one PNG + compact checkpoints)
- Aggressive caching for pip and Hugging Face caches

## Expected runtime budget by stage
Approximate estimates used by the runtime guard:
- baseline eval: 60 min
- short micro-CPT: 40 min
- replay micro-CPT: 40 min
- final eval: 60 min
- report: 5 min

These estimates are conservative and intentionally noisy for ephemeral runners.

## Project layout
- `config/experiment.json`: all constants
- `src/`: data generation, model loading, training, replay, eval, plotting, runtime guard
- `scripts/`: entrypoints
  - `run_baseline_eval.py`
  - `run_micro_cpt.py`
  - `run_eval.py`
  - `make_report.py`
- `.github/workflows/ci.yml`: full pipeline

## Quick start
```bash
make setup
make ci
```

Or stage-by-stage:
```bash
make baseline
make train_short
make train_replay
make eval
make report
```

## Outputs
Written to `outputs/`:
- `eval_baseline.json`
- `train_short.json`
- `train_replay.json`
- `eval_post_short_cpt.json`
- `eval_post_replay_cpt.json`
- `metrics.json`
- `summary.md`
- `accuracy_vs_context.png`

## Notes on parameter-efficient fallback
Full fine-tuning is often too slow for CPU-only GH Actions. By default, this pilot uses a lightweight fallback (`full_finetune=false`) and updates a minimal trainable subset defined in config. This is explicitly logged in `train_*.json`.
