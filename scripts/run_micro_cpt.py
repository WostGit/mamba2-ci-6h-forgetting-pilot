#!/usr/bin/env python3
"""Run short-only or replay micro-CPT with resumable checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from mamba2_pilot.config import load_config
from mamba2_pilot.io_utils import log_runtime_stage
from mamba2_pilot.modeling import (
    apply_parameter_efficient_mode,
    load_local_checkpoint,
    load_model_and_tokenizer,
    save_checkpoint,
)
from mamba2_pilot.runtime_guard import RuntimeGuard
from mamba2_pilot.train import run_micro_cpt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["short", "replay"], required=True)
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

    train_cfg["max_steps"] = guard.state.train_max_steps

    ckpt_dir = Path(train_cfg["checkpoint_subdir"])
    if args.mode == "short":
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_cfg["name"],
            cache_dir=model_cfg["cache_dir"],
            dtype=model_cfg["torch_dtype"],
        )
        trainable = apply_parameter_efficient_mode(model, mode=train_cfg["trainable_mode"])
        print(f"[train] trainable parameters: {trainable}")

        out_dir = ckpt_dir / "short_cpt"
        run_micro_cpt(model, tokenizer, train_cfg, out_dir, seed=cfg.seed, use_replay=False)
        save_checkpoint(model, out_dir)
        stage = "train_short"
    else:
        short_ckpt = ckpt_dir / "short_cpt"
        if not short_ckpt.exists():
            raise FileNotFoundError("Short checkpoint missing. Run --mode short first.")
        _, tokenizer = load_model_and_tokenizer(
            model_name=model_cfg["name"],
            cache_dir=model_cfg["cache_dir"],
            dtype=model_cfg["torch_dtype"],
        )
        model = load_local_checkpoint(short_ckpt, cache_dir=model_cfg["cache_dir"])
        trainable = apply_parameter_efficient_mode(model, mode=train_cfg["trainable_mode"])
        print(f"[train] trainable parameters: {trainable}")

        out_dir = ckpt_dir / "replay_cpt"
        run_micro_cpt(model, tokenizer, train_cfg, out_dir, seed=cfg.seed, use_replay=True)
        save_checkpoint(model, out_dir)
        stage = "train_replay"

    rec = guard.mark_stage_complete(stage)
    guard.fail_if_unrecoverable(stage)
    log_runtime_stage(Path(paths_cfg["metrics_dir"]) / "runtime_log.json", rec)
    print(f"[runtime] {rec}")


if __name__ == "__main__":
    main()
