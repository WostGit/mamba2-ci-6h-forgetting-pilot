#!/usr/bin/env python3
"""Evaluate baseline + two trained adapters on passkey retrieval."""

from __future__ import annotations

import argparse
import time

from mamba2_pilot.config import load_config
from mamba2_pilot.eval import evaluate_passkey
from mamba2_pilot.modeling import (
    apply_trainable_mode,
    load_adapter_state,
    load_model_and_tokenizer,
)
from mamba2_pilot.utils import RuntimeGuard, ensure_dir, read_json, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    model_dir = ensure_dir(out_dir / "models")
    out_file = out_dir / "all_eval_metrics.json"

    if out_file.exists() and not args.force:
        print(f"[skip] eval exists: {out_file}")
        return

    guard = RuntimeGuard(cfg)
    runtime_state_path = out_dir / "runtime_state.json"
    if runtime_state_path.exists():
        runtime_state = read_json(runtime_state_path)
        guard.state.examples_per_length = int(runtime_state["examples_per_length"])
        guard.state.context_lengths = list(runtime_state["context_lengths"])
        guard.state.max_steps = int(runtime_state["max_steps"])

    model, tok, model_id = load_model_and_tokenizer(cfg)

    t0 = time.time()
    baseline = evaluate_passkey(
        model,
        tok,
        guard.state.context_lengths,
        guard.state.examples_per_length,
        cfg["eval"]["max_new_tokens"],
        cfg["seed"],
    )

    apply_trainable_mode(model, cfg["train"]["trainable_mode"])
    load_adapter_state(model, model_dir / "short_adapter.pt")
    short_eval = evaluate_passkey(
        model,
        tok,
        guard.state.context_lengths,
        guard.state.examples_per_length,
        cfg["eval"]["max_new_tokens"],
        cfg["seed"] + 100,
    )

    model, tok, _ = load_model_and_tokenizer(cfg)
    apply_trainable_mode(model, cfg["train"]["trainable_mode"])
    load_adapter_state(model, model_dir / "replay_adapter.pt")
    replay_eval = evaluate_passkey(
        model,
        tok,
        guard.state.context_lengths,
        guard.state.examples_per_length,
        cfg["eval"]["max_new_tokens"],
        cfg["seed"] + 200,
    )
    guard.record_stage("final_eval", time.time() - t0)

    actions = guard.maybe_scale_down()
    guard.assert_within_budget()

    all_results = {
        "model_id": model_id,
        "baseline": baseline,
        "post_short_cpt": short_eval,
        "post_replay_cpt": replay_eval,
        "guard_actions": actions,
        "runtime_log": guard.log_eta_line("final_eval"),
    }
    write_json(out_file, all_results)
    print(guard.log_eta_line("final_eval"))


if __name__ == "__main__":
    main()
