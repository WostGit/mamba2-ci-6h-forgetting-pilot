#!/usr/bin/env python3
"""Run baseline passkey retrieval on the official checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mamba2_pilot.config import load_config
from mamba2_pilot.eval import evaluate_passkey
from mamba2_pilot.modeling import load_model_and_tokenizer
from mamba2_pilot.utils import RuntimeGuard, ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment_config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    out_dir = ensure_dir(cfg["output_dir"])
    out_file = out_dir / "baseline_eval.json"
    guard_file = out_dir / "runtime_state.json"

    if out_file.exists() and not args.force:
        print(f"[skip] baseline exists: {out_file}")
        return

    guard = RuntimeGuard(cfg)

    model, tok, model_id = load_model_and_tokenizer(cfg)
    t0 = time.time()
    results = evaluate_passkey(
        model,
        tok,
        guard.state.context_lengths,
        guard.state.examples_per_length,
        cfg["eval"]["max_new_tokens"],
        cfg["seed"],
    )
    guard.record_stage("baseline_eval", time.time() - t0)

    results["model_id"] = model_id
    results["stage"] = "baseline"
    results["runtime_log"] = guard.log_eta_line("baseline_eval")
    write_json(out_file, results)

    write_json(
        guard_file,
        {
            "examples_per_length": guard.state.examples_per_length,
            "context_lengths": guard.state.context_lengths,
            "max_steps": guard.state.max_steps,
            "runtime_log": guard.log_eta_line("baseline_eval"),
        },
    )
    print(results["runtime_log"])


if __name__ == "__main__":
    main()
