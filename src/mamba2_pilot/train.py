"""Micro-CPT training loop with CPU-friendly defaults."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from .data import build_passkey_example
from .replay import sample_train_length


def run_micro_cpt(
    model: Any,
    tokenizer: Any,
    train_cfg: Dict[str, Any],
    runtime_state: Dict[str, Any],
    mode: str,
    seed: int,
) -> Dict[str, Any]:
    """Run tiny CPT loop and return training metrics."""
    model.train()
    max_steps = int(runtime_state["max_steps"])
    lr = float(train_cfg["learning_rate"])
    grad_clip = float(train_cfg["grad_clip"])
    short_len = int(train_cfg["seq_len"])

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected; choose a trainable_mode fallback.")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(train_cfg["weight_decay"]))
    losses = []
    started = time.time()

    replay_lengths = [512, 1024, 2048]
    for step in range(max_steps):
        seq_len = sample_train_length(mode, short_len, replay_lengths, step, seed)
        ex = build_passkey_example(context_length=seq_len, seed=seed + step)

        toks = tokenizer(
            ex.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=False,
        )
        input_ids = toks["input_ids"]
        attention_mask = toks.get("attention_mask")

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))

    return {
        "mode": mode,
        "steps": max_steps,
        "avg_loss": sum(losses) / len(losses),
        "last_loss": losses[-1],
        "wall_seconds": time.time() - started,
    }
