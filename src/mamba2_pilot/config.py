"""Config loader helpers for the CI-safe Mamba-2 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Simple typed wrapper around a nested YAML config dictionary."""

    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw[name])


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load experiment configuration from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw)
