"""Synthetic passkey-retrieval data generators."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PasskeyExample:
    """One passkey retrieval example."""

    prompt: str
    answer: str


def _filler(n_chars: int) -> str:
    words = ["alpha", "beta", "gamma", "delta", "theta", "lambda"]
    out = []
    while len(" ".join(out)) < n_chars:
        out.append(random.choice(words))
    return " ".join(out)


def make_passkey_example(context_len: int, digits: int = 6) -> PasskeyExample:
    """Create a passkey example with a noisy context and exact key answer."""
    key = "".join(random.choice("0123456789") for _ in range(digits))
    pre = _filler(max(16, context_len // 2))
    post = _filler(max(16, context_len // 2))
    prompt = (
        f"Memorize this secret key: {key}.\n"
        f"{pre}\n{post}\n"
        "Question: What is the secret key? Answer with digits only:"
    )
    return PasskeyExample(prompt=prompt, answer=key)


def make_cpt_text(seq_len: int, short_only: bool = True) -> str:
    """Create short synthetic text for micro-CPT updates."""
    marker = "SHORT" if short_only else "MIXED"
    base = f"{marker} context adaptation sample. "
    text = (base * (seq_len // len(base) + 1))[:seq_len]
    return text
