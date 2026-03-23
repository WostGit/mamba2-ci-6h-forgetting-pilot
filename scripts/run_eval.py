#!/usr/bin/env python3
"""Evaluate a named checkpoint on passkey retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from mamba2_pilot.config import load_config
from mamba2_pilot.eval import evaluate_passkey
from mamba2_pilot.io_utils import log_runtime_stage
from mamba2_pilot.modeling import load_local_checkpoint, load_model_and_tokenizer
from mamba2_pilot.runtime_guard import RuntimeGuard


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", choices=["short", "replay"], required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    model_cfg = cfg.section("model")
    runtime_cfg = cfg.section("runtime")
    eval_cfg = cfg.section("evaluation")
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

    _, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg["name"],
        cache_dir=model_cfg["cache_dir"],
        dtype=model_cfg["torch_dtype"],
    )

    ckpt_dir = Path(train_cfg["checkpoint_subdir"]) / ("short_cpt" if args.checkpoint == "short" else "replay_cpt")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
    model = load_local_checkpoint(ckpt_dir, cache_dir=model_cfg["cache_dir"])

    stage_name = "post_short" if args.checkpoint == "short" else "post_replay"
    out_file = Path(paths_cfg["metrics_dir"]) / f"{stage_name}_eval.json"
    evaluate_passkey(
        model=model,
        tokenizer=tokenizer,
        lengths=lengths,
        examples_per_length=guard.state.examples_per_length,
        passkey_digits=int(eval_cfg["passkey_digits"]),
        max_new_tokens=int(eval_cfg["max_new_tokens"]),
        seed=cfg.seed,
        out_file=out_file,
        stage_name=stage_name,
    )

    stage = "eval_post_short" if args.checkpoint == "short" else "eval_post_replay"
    rec = guard.mark_stage_complete(stage)
    guard.fail_if_unrecoverable(stage)
    log_runtime_stage(Path(paths_cfg["metrics_dir"]) / "runtime_log.json", rec)
    print(f"[runtime] {rec}")


if __name__ == "__main__":
    main()
