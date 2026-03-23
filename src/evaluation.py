"""Passkey retrieval evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List

import torch

from src.data import make_passkey_example


def _extract_digits(text: str) -> str:
    m = re.findall(r"\d+", text)
    return m[-1] if m else ""


def evaluate_passkey(model, tokenizer, cfg: Dict[str, Any], examples_per_length: int, include_8192: bool) -> Dict[str, Any]:
    """Evaluate exact-match passkey retrieval across context lengths."""
    lengths: List[int] = list(cfg["evaluation"]["lengths"])
    if not include_8192:
        lengths = [l for l in lengths if l != 8192]

    per_length = {}
    model.eval()
    with torch.no_grad():
        for ctx_len in lengths:
            ok = 0
            rows = []
            for _ in range(examples_per_length):
                ex = make_passkey_example(
                    context_len=ctx_len,
                    digits=cfg["evaluation"]["passkey_digits"],
                )
                inputs = tokenizer(
                    ex.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(cfg["model"]["max_length"], ctx_len),
                )
                gen = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
                pred = _extract_digits(decoded)
                hit = int(pred == ex.answer)
                ok += hit
                rows.append({"answer": ex.answer, "pred": pred, "hit": hit})

            per_length[str(ctx_len)] = {
                "n": examples_per_length,
                "correct": ok,
                "accuracy": ok / examples_per_length,
                "samples": rows,
            }

    macro = sum(v["accuracy"] for v in per_length.values()) / max(1, len(per_length))
    return {"macro_accuracy": macro, "per_length": per_length}
