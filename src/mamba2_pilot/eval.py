"""Passkey retrieval evaluation helpers."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from .data import build_batch


def _extract_digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def evaluate_passkey(
    model: Any,
    tokenizer: Any,
    context_lengths: list[int],
    examples_per_length: int,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute retrieval accuracy by context length."""
    model.eval()
    torch.set_grad_enabled(False)

    by_length = {}
    total_correct = 0
    total = 0
    started = time.time()

    for clen in context_lengths:
        examples = build_batch(clen, examples_per_length, seed + clen)
        correct = 0
        for ex in examples:
            toks = tokenizer(ex.prompt, return_tensors="pt", truncation=True, max_length=clen)
            gen = model.generate(
                input_ids=toks["input_ids"],
                attention_mask=toks.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            new_tokens = gen[0][toks["input_ids"].shape[1] :]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred_digits = _extract_digits(pred)
            if ex.key in pred_digits:
                correct += 1

        acc = correct / examples_per_length
        by_length[str(clen)] = {
            "correct": correct,
            "total": examples_per_length,
            "accuracy": acc,
        }
        total_correct += correct
        total += examples_per_length

    return {
        "by_length": by_length,
        "overall_accuracy": total_correct / max(total, 1),
        "wall_seconds": time.time() - started,
        "examples_per_length": examples_per_length,
        "context_lengths": context_lengths,
    }
