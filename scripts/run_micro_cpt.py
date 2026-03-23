#!/usr/bin/env python3
"""Run short-only or replay micro-CPT."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config import load_config, load_or_init_runtime_state, save_runtime_state
from src.model import load_checkpoint, load_model_and_tokenizer
from src.runtime_guard import RuntimeGuard
from src.train import run_micro_cpt
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["short", "replay"], required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])

    state = load_or_init_runtime_state(cfg)
    guard = RuntimeGuard(cfg, state)

    final_ckpt = Path(cfg["output_dir"]) / "checkpoints" / f"{args.condition}_final.pt"
    if final_ckpt.exists():
        print(f"[resume] Using cached training checkpoint: {final_ckpt}")
        return

    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg)

    if args.condition == "replay":
        short_ckpt = Path(cfg["output_dir"]) / "checkpoints" / "short_final.pt"
        if not short_ckpt.exists():
            raise FileNotFoundError("Missing short_final.pt; run short condition first.")
        load_checkpoint(model, short_ckpt)

    run_micro_cpt(model, tok, cfg, args.condition, steps=state.train_steps)

    stage = "train_short" if args.condition == "short" else "train_replay"
    note = guard.apply_after_stage(stage, time.time() - t0)
    print(note)
    save_runtime_state(cfg, state)


if __name__ == "__main__":
    main()
