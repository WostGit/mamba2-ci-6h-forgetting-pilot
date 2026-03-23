# Mamba-2 130M CI Pilot: micro-CPT forgetting vs mixed-length replay

This repository is a **minimal, reproducible, CI-safe pilot** to test:

1. whether short-context micro-CPT hurts longer-context retrieval in a pure Mamba-2 130M model
2. whether a simple mixed-length replay mitigation reduces that harm

## Hypothesis

- **H1:** short-only micro-CPT (train sequence length 256) decreases passkey retrieval accuracy at long contexts.
- **H2:** adding mixed-length replay during micro-CPT partially restores long-context retrieval.

## Compute assumptions (hard constraints)

Designed for a standard GitHub-hosted Ubuntu runner:

- CPU only (no GPU)
- 4 vCPU, 16 GB RAM, ~14 GB SSD
- workflow hard timeout: **340 minutes** (< 6 hours)

Model target is fixed to **`state-spaces/mamba2-130m`**.

> Current status note: if the checkpoint/config pairing cannot be verified as a real Mamba-2 130M match in CI, the pipeline is expected to fail fast during baseline loading instead of pretending the pilot is trustworthy.

## Why this fits GitHub Actions limits

Runtime-safety decisions:

- tiny train budget: max 80 optimizer steps, batch size 1, seq len 256
- tiny eval budget: passkey only, 4 context lengths, 3 examples/length
- aggressive caching of Hugging Face cache and outputs
- small artifacts only (`metrics.json`, summary markdown, png plot, logs)
- parameter-efficient fallback for CPU: **LM-head-only** updates
- resume-friendly scripts: each stage skips when output already exists

Runtime guard auto-scales down when at risk:

1. reduce eval examples/length from 3 -> 2
2. skip context length 8192
3. reduce training steps from 80 -> 60

If the guard predicts the run still cannot finish under budget, it **fails loudly**.

## Expected runtime budget by stage

These are planning estimates in `config.yaml` used by runtime checks:

- baseline eval: ~25 min
- short-only micro-CPT train: ~60 min
- post-short eval: ~20 min
- replay micro-CPT train: ~60 min
- post-replay eval: ~20 min
- reporting: a few minutes

Target total: **~2-4 hours**, hard ceiling < 6 hours.

## Experiment conditions

1. baseline eval on official checkpoint
2. short-only micro-CPT
3. short-only micro-CPT + mixed-length replay

Training budget per training condition:

- <= 80 steps
- batch size 1
- train length 256
- CPU only

Evaluation:

- passkey retrieval
- lengths: 1024, 2048, 4096, 8192 (guard may skip 8192)
- examples: 3 per length (guard may reduce to 2)
- rounds: baseline, post-short-CPT, post-replay-CPT

## Repository layout

- `src/` core modules
  - passkey data generation
  - model loading/checkpoint logic
  - micro-CPT training loop
  - replay sampling
  - evaluation
  - plotting
- `scripts/` entrypoints
  - `run_baseline_eval.py`
  - `run_micro_cpt.py`
  - `run_eval.py`
  - `make_report.py`
- `.github/workflows/ci.yml` full CPU pipeline
- `config.yaml` all constants

## Usage

```bash
make setup
make baseline
make train_short
make eval_short
make train_replay
make eval
make report
# or run everything:
make ci
```

Outputs are written to `outputs/`:

- `metrics.json`
- `report.md`
- `passkey_accuracy.png`
- `stage_log.jsonl`
- `runtime_state.json`

## Notes on reproducibility

- deterministic seeds are set in every entrypoint
- stage logs include estimated remaining time after each major stage
- intermediate artifacts are persisted to support warm-cache resumes
