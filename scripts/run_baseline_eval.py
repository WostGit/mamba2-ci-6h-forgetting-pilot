#!/usr/bin/env python3
"""Run baseline passkey evaluation on official Mamba-2 checkpoint."""

from __future__ import annotations

import time
import sys
from pathlib import Path

from common import load_metrics, save_metrics, stage_done

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation import evaluate_passkey
from src.modeling import load_model_and_tokenizer
from src.utils import load_runtime_state, set_seed


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    start = time.time()

    metrics = load_metrics(cfg["paths"]["metrics_json"])
    if "baseline" in metrics:
        print("baseline metrics already present; skipping")
        stage_done(cfg, "baseline_eval", start)
        return

    model, tok = load_model_and_tokenizer(cfg)
    state = load_runtime_state(cfg)
    metrics["baseline"] = evaluate_passkey(
        model,
        tok,
        cfg,
        examples_per_length=state.examples_per_length,
        include_8192=state.include_8192,
    )
    save_metrics(cfg["paths"]["metrics_json"], metrics)
    stage_done(cfg, "baseline_eval", start)


if __name__ == "__main__":
    main()
