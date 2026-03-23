"""Configuration loading for the pilot experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RuntimeState:
    """Tracks mutable runtime knobs used for automatic scale-down."""

    examples_per_length: int
    include_8192: bool
    train_steps: int
    elapsed_seconds: float = 0.0


def load_config(path: str | Path = "config/experiment.json") -> Dict[str, Any]:
    """Load experiment constants from JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def state_path(cfg: Dict[str, Any]) -> Path:
    """Return path for runtime state persistence."""
    return Path(cfg["output_dir"]) / "runtime_state.json"


def load_or_init_runtime_state(cfg: Dict[str, Any]) -> RuntimeState:
    """Load runtime state from disk or initialize defaults."""
    p = state_path(cfg)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
        return RuntimeState(**payload)
    return RuntimeState(
        examples_per_length=cfg["eval"]["examples_per_length"],
        include_8192=True,
        train_steps=cfg["train"]["steps"],
        elapsed_seconds=0.0,
    )


def save_runtime_state(cfg: Dict[str, Any], state: RuntimeState) -> None:
    """Persist runtime state for resume across CI stages."""
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")


def adjusted_context_lengths(cfg: Dict[str, Any], state: RuntimeState) -> List[int]:
    """Return active context lengths based on guard decisions."""
    lengths = list(cfg["eval"]["context_lengths"])
    if not state.include_8192:
        lengths = [x for x in lengths if x != 8192]
    return lengths
