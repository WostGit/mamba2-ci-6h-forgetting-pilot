"""Passkey retrieval evaluation."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.data import PasskeyExample, build_eval_set


def _predict_answer(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    toks = tokenizer(prompt, return_tensors="pt", truncation=False)
    with torch.no_grad():
        out = model.generate(
            input_ids=toks["input_ids"],
            attention_mask=toks.get("attention_mask"),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][toks["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_passkey_eval(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    label: str,
    context_lengths: List[int],
    examples_per_length: int,
) -> Dict[str, Any]:
    """Evaluate retrieval accuracy by context length."""
    model.eval()
    started = time.time()
    dataset: List[PasskeyExample] = build_eval_set(context_lengths, examples_per_length, cfg["seed"])

    hits = defaultdict(int)
    totals = defaultdict(int)
    rows = []
    for ex in dataset:
        pred = _predict_answer(model, tokenizer, ex.prompt, cfg["eval"]["max_new_tokens"])
        ok = ex.answer in pred
        hits[ex.context_length] += int(ok)
        totals[ex.context_length] += 1
        rows.append(
            {
                "context_length": ex.context_length,
                "target": ex.answer,
                "prediction": pred,
                "correct": ok,
            }
        )

    by_len = {
        str(k): {
            "accuracy": hits[k] / max(1, totals[k]),
            "correct": hits[k],
            "total": totals[k],
        }
        for k in sorted(totals)
    }
    result = {
        "label": label,
        "context_lengths": context_lengths,
        "examples_per_length": examples_per_length,
        "by_length": by_len,
        "elapsed_seconds": time.time() - started,
        "rows": rows,
    }
    out_path = Path(cfg["output_dir"]) / f"eval_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
