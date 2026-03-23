"""Passkey retrieval synthetic data generation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PasskeyExample:
    """Single retrieval sample with full prompt and expected key."""

    prompt: str
    key: str
    context_length: int


def _make_noise_tokens(n_tokens: int, rng: random.Random) -> str:
    words = ["alpha", "beta", "gamma", "delta", "omega", "kappa", "lambda"]
    return " ".join(rng.choice(words) for _ in range(n_tokens))


def build_passkey_example(context_length: int, seed: int) -> PasskeyExample:
    """Create one passkey sample approximating a target token length."""
    rng = random.Random(seed)
    key = f"{rng.randint(10000, 99999)}"
    key_sentence = f"The passkey is {key}. Remember it exactly."
    query = "\n\nQuestion: What is the passkey? Answer with digits only.\nAnswer:"

    # Rough token budgeting: 1 word ~ 1 token for this pilot's synthetic text.
    reserve = len((key_sentence + query).split()) + 8
    noise_tokens = max(context_length - reserve, 16)
    prefix = _make_noise_tokens(noise_tokens // 2, rng)
    suffix = _make_noise_tokens(noise_tokens - (noise_tokens // 2), rng)
    prompt = f"{prefix} {key_sentence} {suffix} {query}"
    return PasskeyExample(prompt=prompt, key=key, context_length=context_length)


def build_batch(context_length: int, n_examples: int, seed: int) -> list[PasskeyExample]:
    """Generate deterministic batch of passkey examples for one context length."""
    return [build_passkey_example(context_length, seed + i) for i in range(n_examples)]
