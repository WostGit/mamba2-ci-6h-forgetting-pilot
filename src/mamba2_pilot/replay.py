"""Replay sampling policy for mixed-length micro-CPT."""

from __future__ import annotations

import random


def sample_train_length(short_length: int, replay_length: int, replay_ratio: float) -> int:
    """Return short length most of the time, replay length occasionally."""
    if random.random() < replay_ratio:
        return replay_length
    return short_length
