#!/usr/bin/env python3
"""Run baseline passkey retrieval eval on official Mamba-2 130M checkpoint."""

from __future__ import annotations

from pathlib import Path

from mamba2_pilot.config import load_config
from mamba2_pilot.eval import evaluate_passkey
from mamba2_pilot.io_utils import log_runtime_stage
from mamba2_pilot.modeling import load_model_and_tokenizer
from mamba2_pilot.runtime_guard import RuntimeGuard


def main() -> None:
    cfg = load_config()
    model_cfg = cfg.section("model")
    eval_cfg = cfg.section("evaluation")
    runtime_cfg = cfg.section("runtime")
    train_cfg = cfg.section("training")
    paths_cfg = cfg.section("paths")

    guard = RuntimeGuard(
        hard_budget_hours=float(runtime_cfg["hard_budget_hours"]),
        stage_estimates_minutes=runtime_cfg["stage_estimates_minutes"],
        initial_examples=int(eval_cfg["examples_per_length"]),
        min_examples=int(eval_cfg["fallback_examples_per_length"]),
        initial_steps=int(train_cfg["max_steps"]),
        min_steps=int(train_cfg["min_steps_on_guard"]),
        state_path=Path(paths_cfg["metrics_dir"]) / "runtime_state.json",
    )

    lengths = list(eval_cfg["lengths"])
    if not guard.state.include_8192 and 8192 in lengths:
        lengths.remove(8192)

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        cache_dir=model_cfg["cache_dir"],
        dtype=model_cfg["torch_dtype"],
    )
    evaluate_passkey(
        model=model,
        tokenizer=tokenizer,
        lengths=lengths,
        examples_per_length=guard.state.examples_per_length,
        passkey_digits=int(eval_cfg["passkey_digits"]),
        max_new_tokens=int(eval_cfg["max_new_tokens"]),
        seed=cfg.seed,
        out_file=Path(paths_cfg["metrics_dir"]) / "baseline_eval.json",
        stage_name="baseline",
    )

    rec = guard.mark_stage_complete("baseline_eval")
    log_runtime_stage(Path(paths_cfg["metrics_dir"]) / "runtime_log.json", rec)
    print(f"[runtime] {rec}")


if __name__ == "__main__":
    main()
