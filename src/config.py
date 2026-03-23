"""Configuration utilities for experiment constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
