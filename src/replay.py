"""Replay length sampling utilities."""

from __future__ import annotations

import random
from typing import Dict, Any


def sample_replay_length(cfg: Dict[str, Any]) -> int:
    """Sample one replay length from configured mixed pool."""
    return random.choice(cfg["training"]["replay_lengths"])
