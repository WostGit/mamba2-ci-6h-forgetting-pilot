#!/usr/bin/env python3
"""Run baseline passkey retrieval evaluation for official Mamba-2 checkpoint."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import adjusted_context_lengths, load_config, load_or_init_runtime_state, save_runtime_state
from src.eval import run_passkey_eval
from src.model import load_model_and_tokenizer
from src.runtime_guard import RuntimeGuard
from src.utils import set_seed


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    state = load_or_init_runtime_state(cfg)
    guard = RuntimeGuard(cfg, state)

    out_file = Path(cfg["output_dir"]) / "eval_baseline.json"
    if out_file.exists():
        print(f"[resume] Using cached baseline eval: {out_file}")
        return

    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg)
    run_passkey_eval(
        model,
        tok,
        cfg,
        label="baseline",
        context_lengths=adjusted_context_lengths(cfg, state),
        examples_per_length=state.examples_per_length,
    )
    note = guard.apply_after_stage("baseline_eval", time.time() - t0)
    print(note)
    save_runtime_state(cfg, state)


if __name__ == "__main__":
    main()
