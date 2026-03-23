"""Replay sampling for mixed-length micro-CPT."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional


def choose_replay_length(cfg: Dict[str, Any], rng: random.Random) -> Optional[int]:
    """Sample a long-context source length for replay examples."""
    if rng.random() > cfg["replay"]["replay_probability"]:
        return None
    return rng.choice(cfg["replay"]["long_context_candidates"])
