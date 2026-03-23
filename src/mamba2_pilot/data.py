"""Passkey retrieval synthetic data generation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PasskeyExample:
    """Container for one passkey retrieval sample."""

    prompt: str
    target: str
    context_length: int


def set_seed(seed: int) -> None:
    """Set python RNG seed for deterministic generation."""
    random.seed(seed)


def make_passkey_example(context_length: int, digits: int) -> PasskeyExample:
    """Create a single passkey example near a target context length."""
    key = "".join(str(random.randint(0, 9)) for _ in range(digits))
    prefix = (
        "You are reading irrelevant notes. Ignore all filler text. "
        "At one point, a passkey appears. Keep it in memory.\n"
    )
    key_line = f"PASSKEY: {key}\n"
    suffix_q = "\nQuestion: What is the passkey? Answer with digits only.\nAnswer:"

    filler_token = " lorem"
    target_body_len = max(context_length - len(prefix) - len(key_line) - len(suffix_q), 0)
    filler = (filler_token * (target_body_len // len(filler_token) + 1))[:target_body_len]
    prompt = prefix + filler + "\n" + key_line + suffix_q
    return PasskeyExample(prompt=prompt, target=key, context_length=context_length)


def make_eval_set(lengths: list[int], examples_per_length: int, digits: int) -> list[PasskeyExample]:
    """Create evaluation samples for all requested lengths."""
    samples: list[PasskeyExample] = []
    for length in lengths:
        for _ in range(examples_per_length):
            samples.append(make_passkey_example(length, digits))
    return samples
