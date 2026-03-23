#!/usr/bin/env python3
"""Evaluate post-training checkpoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import load_metrics, save_metrics, stage_done

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation import evaluate_passkey
from src.modeling import load_model_and_tokenizer, load_trainable_checkpoint
from src.utils import load_runtime_state, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["short", "replay"], required=True)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    start = time.time()
    state = load_runtime_state(cfg)
    metrics = load_metrics(cfg["paths"]["metrics_json"])

    model, tok = load_model_and_tokenizer(cfg)

    if args.mode == "short":
        key = "post_short_cpt"
        stage = "eval_short"
        ckpt = cfg["paths"]["short_ckpt"]
    else:
        key = "post_replay_cpt"
        stage = "eval_replay"
        ckpt = cfg["paths"]["replay_ckpt"]

    if key in metrics:
        print(f"{key} already evaluated; skipping")
        stage_done(cfg, stage, start)
        return

    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    load_trainable_checkpoint(model, ckpt)
    metrics[key] = evaluate_passkey(
        model,
        tok,
        cfg,
        examples_per_length=state.examples_per_length,
        include_8192=state.include_8192,
    )
    save_metrics(cfg["paths"]["metrics_json"], metrics)
    stage_done(cfg, stage, start)


if __name__ == "__main__":
    main()
