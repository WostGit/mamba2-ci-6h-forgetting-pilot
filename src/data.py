"""Synthetic passkey retrieval data generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class PasskeyExample:
    """One passkey retrieval sample."""

    prompt: str
    answer: str
    context_length: int


def _noise_chunk(rng: random.Random, token_budget: int) -> str:
    vocab = [
        "alpha", "beta", "gamma", "delta", "omega", "lambda", "sigma", "kappa", "zeta", "theta"
    ]
    return " ".join(rng.choice(vocab) for _ in range(token_budget))


def make_passkey_example(context_length: int, seed: int) -> PasskeyExample:
    """Build a deterministic prompt containing a hidden numeric passkey."""
    rng = random.Random(seed)
    passkey = f"{rng.randint(10000, 99999)}"
    prefix_budget = max(64, context_length // 2)
    suffix_budget = max(64, context_length - prefix_budget)

    prompt = (
        "You must remember a numeric passkey buried in filler text.\n"
        f"{_noise_chunk(rng, prefix_budget)}\n"
        f"PASSKEY: {passkey}\n"
        f"{_noise_chunk(rng, suffix_budget)}\n"
        "Question: what is the passkey? Answer with digits only.\nAnswer:"
    )
    return PasskeyExample(prompt=prompt, answer=passkey, context_length=context_length)


def build_eval_set(context_lengths: List[int], examples_per_length: int, seed: int) -> List[PasskeyExample]:
    """Create the full passkey retrieval evaluation set."""
    out: List[PasskeyExample] = []
    cursor = seed
    for cl in context_lengths:
        for _ in range(examples_per_length):
            out.append(make_passkey_example(cl, cursor))
            cursor += 1
    return out


def build_train_example(seq_len: int, seed: int, long_context: int | None = None) -> PasskeyExample:
    """Create short training samples, with optional long-context replay source."""
    source_len = long_context if long_context is not None else seq_len
    ex = make_passkey_example(source_len, seed)
    # Keep training sequence short for CI speed while preserving supervision target.
    chopped = ex.prompt[: seq_len * 6]
    return PasskeyExample(prompt=chopped, answer=ex.answer, context_length=seq_len)
