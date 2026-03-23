"""Micro-CPT loop with CPU-safe defaults and resume artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW

from .data import make_passkey_example
from .replay import sample_train_length


def _train_step(model, tokenizer, text: str, sequence_length: int) -> float:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=sequence_length,
        padding="max_length",
    )
    inputs = {k: v.to("cpu") for k, v in encoded.items()}
    labels = inputs["input_ids"].clone()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    return float(loss.detach().cpu().item())


def run_micro_cpt(
    model,
    tokenizer,
    train_cfg: dict,
    out_dir: Path,
    seed: int,
    use_replay: bool,
    replay_ratio: float = 0.2,
) -> dict:
    """Run micro-CPT for a tiny fixed-step budget."""
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=float(train_cfg["learning_rate"]))
    short_len = int(train_cfg["short_length"])
    replay_len = int(train_cfg["replay_length"])
    max_steps = int(train_cfg["max_steps"])
    seq_len = int(train_cfg["sequence_length"])

    losses: list[float] = []
    model.train()
    for step in range(1, max_steps + 1):
        length = sample_train_length(short_len, replay_len, replay_ratio) if use_replay else short_len
        ex = make_passkey_example(context_length=length, digits=8)

        optimizer.zero_grad(set_to_none=True)
        loss = _train_step(model, tokenizer, ex.prompt, sequence_length=seq_len)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["max_grad_norm"]))
        optimizer.step()
        losses.append(loss)

        if step % 10 == 0:
            (out_dir / "progress.json").write_text(
                json.dumps({"step": step, "avg_loss_last_10": sum(losses[-10:]) / 10.0}, indent=2),
                encoding="utf-8",
            )

    metrics = {
        "steps": max_steps,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "final_loss": losses[-1] if losses else None,
        "use_replay": use_replay,
        "replay_ratio": replay_ratio if use_replay else 0.0,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
