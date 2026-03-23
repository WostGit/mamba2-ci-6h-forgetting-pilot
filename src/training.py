"""Micro-CPT loop with short-only and replay variants."""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.data import make_cpt_text
from src.modeling import apply_lm_head_only_training
from src.replay import sample_replay_length


def run_micro_cpt(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    steps: int,
    use_replay: bool,
) -> Dict[str, Any]:
    """Run tiny CPU micro-CPT for at most `steps` updates."""
    trainable = apply_lm_head_only_training(model)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found for PEFT fallback.")

    model.train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    losses = []
    for step in range(steps):
        seq_len = cfg["training"]["short_length"]
        if use_replay:
            seq_len = sample_replay_length(cfg)
        txt = make_cpt_text(seq_len=seq_len, short_only=not use_replay)
        batch = tokenizer(
            txt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg["training"]["train_seq_len"],
        )
        batch["labels"] = batch["input_ids"].clone()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg["training"]["grad_clip"],
        )
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu().item()))

        if (step + 1) % 10 == 0:
            print(f"step={step + 1}/{steps} loss={losses[-1]:.4f} seq_len={seq_len}")

    return {
        "steps": steps,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "trainable_params": trainable,
        "mode": "replay" if use_replay else "short_only",
    }
