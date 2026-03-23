"""Micro-CPT training loop constrained for CPU CI execution."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import torch

from src.data import build_train_example
from src.model import configure_trainable_parameters, save_checkpoint
from src.replay import choose_replay_length


def run_micro_cpt(
    model: torch.nn.Module,
    tokenizer,
    cfg: Dict[str, Any],
    condition: str,
    steps: int,
) -> Dict[str, Any]:
    """Run short-only or replay-augmented micro-CPT."""
    seed = cfg["seed"] + (100 if condition == "replay" else 0)
    rng = random.Random(seed)

    trainable = configure_trainable_parameters(model, cfg)
    if trainable == 0:
        raise RuntimeError("No trainable parameters selected; cannot run micro-CPT.")

    model.train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    started = time.time()
    losses = []
    for step in range(steps):
        replay_len = choose_replay_length(cfg, rng) if condition == "replay" else None
        ex = build_train_example(cfg["train"]["seq_len"], seed + step, replay_len)
        full_text = f"{ex.prompt} {ex.answer}"
        toks = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=cfg["train"]["seq_len"])
        input_ids = toks["input_ids"]
        attn = toks["attention_mask"]

        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))

        if (step + 1) % cfg["train"]["save_every"] == 0:
            save_checkpoint(model, cfg["output_dir"], f"{condition}_step_{step+1}.pt")

    last_ckpt = save_checkpoint(model, cfg["output_dir"], f"{condition}_final.pt")
    report = {
        "condition": condition,
        "steps": steps,
        "trainable_params": trainable,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(last_ckpt),
        "parameter_efficient_fallback": not cfg["train"]["full_finetune"],
    }
    out_path = Path(cfg["output_dir"]) / f"train_{condition}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
