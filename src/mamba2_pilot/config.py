"""Configuration utilities for the pilot experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config/experiment_config.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load JSON config and return a mutable dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
