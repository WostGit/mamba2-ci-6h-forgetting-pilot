"""Replay sampling policies for mixed-length CPT."""

from __future__ import annotations

import random


def sample_train_length(mode: str, short_len: int, replay_lengths: list[int], step: int, seed: int) -> int:
    """Return train sequence length for this step under selected mode."""
    if mode == "short":
        return short_len
    if mode == "replay":
        rng = random.Random(seed + step)
        # Runtime-saving policy: 75% short, 25% replay from longer lengths.
        if rng.random() < 0.75:
            return short_len
        return rng.choice(replay_lengths)
    raise ValueError(f"Unsupported mode={mode}")
