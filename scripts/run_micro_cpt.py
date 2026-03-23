#!/usr/bin/env python3
"""Run micro-CPT training for short-only or replay condition."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import load_metrics, save_metrics, stage_done

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.modeling import load_model_and_tokenizer, load_trainable_checkpoint, save_trainable_checkpoint
from src.training import run_micro_cpt
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
        ckpt_out = cfg["paths"]["short_ckpt"]
        metrics_key = "short_cpt_train"
        stage = "train_short"
        if Path(ckpt_out).exists() and metrics_key in metrics:
            print("short checkpoint exists; skipping training")
            stage_done(cfg, stage, start)
            return
        train_info = run_micro_cpt(model, tok, cfg, steps=state.train_steps, use_replay=False)
    else:
        if not Path(cfg["paths"]["short_ckpt"]).exists():
            raise FileNotFoundError("Replay training requires short checkpoint first")
        load_trainable_checkpoint(model, cfg["paths"]["short_ckpt"])
        ckpt_out = cfg["paths"]["replay_ckpt"]
        metrics_key = "replay_cpt_train"
        stage = "train_replay"
        if Path(ckpt_out).exists() and metrics_key in metrics:
            print("replay checkpoint exists; skipping training")
            stage_done(cfg, stage, start)
            return
        train_info = run_micro_cpt(model, tok, cfg, steps=state.train_steps, use_replay=True)

    save_trainable_checkpoint(model, ckpt_out)
    metrics[metrics_key] = train_info
    save_metrics(cfg["paths"]["metrics_json"], metrics)
    stage_done(cfg, stage, start)


if __name__ == "__main__":
    main()
