"""Passkey retrieval evaluation for fixed context lengths."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import make_eval_set


def _extract_digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def evaluate_passkey(
    model,
    tokenizer,
    lengths: list[int],
    examples_per_length: int,
    passkey_digits: int,
    max_new_tokens: int,
    seed: int,
    out_file: Path,
    stage_name: str,
) -> dict:
    """Run passkey retrieval and emit JSON metrics."""
    torch.manual_seed(seed)
    model.eval()
    samples = make_eval_set(lengths, examples_per_length, passkey_digits)

    by_length = defaultdict(lambda: {"correct": 0, "total": 0})
    with torch.no_grad():
        for ex in samples:
            encoded = tokenizer(ex.prompt, return_tensors="pt", truncation=True, max_length=ex.context_length)
            encoded = {k: v.to("cpu") for k, v in encoded.items()}
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = out[0][encoded["input_ids"].shape[1] :]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred_digits = _extract_digits(pred)[:passkey_digits]
            is_correct = int(pred_digits == ex.target)
            by_length[ex.context_length]["correct"] += is_correct
            by_length[ex.context_length]["total"] += 1

    results = {"stage": stage_name, "examples_per_length": examples_per_length, "by_length": {}}
    for length in lengths:
        rec = by_length[length]
        total = max(rec["total"], 1)
        acc = rec["correct"] / total
        results["by_length"][str(length)] = {
            "accuracy": acc,
            "correct": rec["correct"],
            "total": rec["total"],
        }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
