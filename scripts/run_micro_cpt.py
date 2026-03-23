#!/usr/bin/env python3
"""Run micro-CPT for short-only or replay condition."""

from __future__ import annotations

import argparse
import time

from mamba2_pilot.config import load_config
from mamba2_pilot.modeling import apply_trainable_mode, load_model_and_tokenizer, save_adapter_state
from mamba2_pilot.train import run_micro_cpt
from mamba2_pilot.utils import RuntimeGuard, ensure_dir, read_json, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_config.json")
    parser.add_argument("--mode", choices=["short", "replay"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    out_dir = ensure_dir(cfg["output_dir"])
    model_dir = ensure_dir(out_dir / "models")
    out_metrics = out_dir / f"train_{args.mode}.json"
    out_adapter = model_dir / f"{args.mode}_adapter.pt"

    if out_metrics.exists() and out_adapter.exists() and not args.force:
        print(f"[skip] train {args.mode} exists")
        return

    guard = RuntimeGuard(cfg)
    runtime_state_path = out_dir / "runtime_state.json"
    if runtime_state_path.exists():
        runtime_state = read_json(runtime_state_path)
        guard.state.examples_per_length = int(runtime_state["examples_per_length"])
        guard.state.context_lengths = list(runtime_state["context_lengths"])
        guard.state.max_steps = int(runtime_state["max_steps"])

    model, tok, model_id = load_model_and_tokenizer(cfg)
    trainable = apply_trainable_mode(model, cfg["train"]["trainable_mode"])

    t0 = time.time()
    train_metrics = run_micro_cpt(
        model,
        tok,
        cfg["train"],
        runtime_state={"max_steps": guard.state.max_steps},
        mode=args.mode,
        seed=cfg["seed"],
    )
    stage_name = f"train_{args.mode}"
    guard.record_stage(stage_name, time.time() - t0)

    actions = guard.maybe_scale_down()
    guard.assert_within_budget()

    save_adapter_state(model, out_adapter)
    write_json(
        out_metrics,
        {
            **train_metrics,
            "model_id": model_id,
            "trainable_parameters": trainable,
            "guard_actions": actions,
            "runtime_log": guard.log_eta_line(stage_name),
        },
    )
    write_json(
        runtime_state_path,
        {
            "examples_per_length": guard.state.examples_per_length,
            "context_lengths": guard.state.context_lengths,
            "max_steps": guard.state.max_steps,
            "runtime_log": guard.log_eta_line(stage_name),
        },
    )
    print(guard.log_eta_line(stage_name))
    if actions:
        print("[runtime_guard] " + " | ".join(actions))


if __name__ == "__main__":
    main()
