#!/usr/bin/env python3
"""Evaluate post-training checkpoints and aggregate metrics."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import adjusted_context_lengths, load_config, load_or_init_runtime_state, save_runtime_state
from src.eval import run_passkey_eval
from src.model import load_checkpoint, load_model_and_tokenizer
from src.runtime_guard import RuntimeGuard
from src.utils import load_json, save_json, set_seed


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])

    state = load_or_init_runtime_state(cfg)
    guard = RuntimeGuard(cfg, state)

    metrics_out = Path(cfg["output_dir"]) / "metrics.json"
    if metrics_out.exists():
        print(f"[resume] Using cached metrics bundle: {metrics_out}")
        return

    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg)

    short_ckpt = Path(cfg["output_dir"]) / "checkpoints" / "short_final.pt"
    replay_ckpt = Path(cfg["output_dir"]) / "checkpoints" / "replay_final.pt"
    if not short_ckpt.exists() or not replay_ckpt.exists():
        raise FileNotFoundError("Missing training checkpoints; run train_short and train_replay first.")

    load_checkpoint(model, short_ckpt)
    short_eval = run_passkey_eval(
        model,
        tok,
        cfg,
        label="post_short_cpt",
        context_lengths=adjusted_context_lengths(cfg, state),
        examples_per_length=state.examples_per_length,
    )

    model, tok = load_model_and_tokenizer(cfg)
    load_checkpoint(model, replay_ckpt)
    replay_eval = run_passkey_eval(
        model,
        tok,
        cfg,
        label="post_replay_cpt",
        context_lengths=adjusted_context_lengths(cfg, state),
        examples_per_length=state.examples_per_length,
    )

    baseline = load_json(Path(cfg["output_dir"]) / "eval_baseline.json")
    save_json(
        metrics_out,
        {
            "baseline": baseline,
            "post_short_cpt": short_eval,
            "post_replay_cpt": replay_eval,
            "runtime_state": state.__dict__,
        },
    )

    note = guard.apply_after_stage("final_eval", time.time() - t0)
    print(note)
    save_runtime_state(cfg, state)


if __name__ == "__main__":
    main()
