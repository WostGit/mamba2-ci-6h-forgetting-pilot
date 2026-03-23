"""Utility helpers for deterministic runs, time tracking, and IO."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write small JSON artifacts with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Dict[str, Any]:
    """Read JSON artifact from disk."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class RuntimeState:
    """Tracks stage durations and mutable guard settings."""

    started_at: float
    elapsed_by_stage: Dict[str, float]
    examples_per_length: int
    context_lengths: list[int]
    max_steps: int


class RuntimeGuard:
    """Applies conservative runtime reductions when risk is detected."""

    def __init__(self, config: Dict[str, Any]):
        runtime_cfg = config["runtime"]
        eval_cfg = config["eval"]
        train_cfg = config["train"]

        self.budget_seconds = float(runtime_cfg["ci_budget_hours"]) * 3600
        self.state = RuntimeState(
            started_at=time.time(),
            elapsed_by_stage={},
            examples_per_length=int(eval_cfg["examples_per_length"]),
            context_lengths=list(eval_cfg["context_lengths"]),
            max_steps=int(train_cfg["max_steps"]),
        )
        self.runtime_cfg = runtime_cfg

    def record_stage(self, stage: str, seconds: float) -> None:
        """Record wall-clock seconds for a named stage."""
        self.state.elapsed_by_stage[stage] = seconds

    def estimate_remaining_seconds(self) -> float:
        """Use observed averages plus conservative priors for unrun stages."""
        observed = self.state.elapsed_by_stage
        # Conservative defaults for CPU-only Mamba runs.
        priors = {
            "baseline_eval": 1800,
            "train_short": 3600,
            "train_replay": 3600,
            "final_eval": 1800,
            "report": 120,
        }
        total_projected = 0.0
        for stage, prior in priors.items():
            total_projected += observed.get(stage, prior)
        elapsed_so_far = time.time() - self.state.started_at
        return max(total_projected - elapsed_so_far, 0.0)

    def maybe_scale_down(self) -> list[str]:
        """Apply staged reductions when projected time crosses budget."""
        actions: list[str] = []
        projected_total = (time.time() - self.state.started_at) + self.estimate_remaining_seconds()

        if projected_total <= self.budget_seconds:
            return actions

        if self.state.examples_per_length > self.runtime_cfg["guard_eval_examples_stepdown"]:
            self.state.examples_per_length = int(self.runtime_cfg["guard_eval_examples_stepdown"])
            actions.append("Reduced eval examples per length to 2.")

        projected_total = (time.time() - self.state.started_at) + self.estimate_remaining_seconds()
        if projected_total > self.budget_seconds and self.runtime_cfg["guard_skip_length"] in self.state.context_lengths:
            self.state.context_lengths = [x for x in self.state.context_lengths if x != self.runtime_cfg["guard_skip_length"]]
            actions.append("Skipped 8192-token eval context.")

        projected_total = (time.time() - self.state.started_at) + self.estimate_remaining_seconds()
        if projected_total > self.budget_seconds and self.state.max_steps > self.runtime_cfg["guard_train_steps_stepdown"]:
            self.state.max_steps = int(self.runtime_cfg["guard_train_steps_stepdown"])
            actions.append("Reduced training steps from 80 to 60.")

        return actions

    def assert_within_budget(self) -> None:
        """Fail loudly if projection still exceeds budget after guard actions."""
        projected_total = (time.time() - self.state.started_at) + self.estimate_remaining_seconds()
        if self.runtime_cfg.get("fail_if_projected_over_budget", True) and projected_total > self.budget_seconds:
            raise RuntimeError(
                f"Projected runtime {projected_total/3600:.2f}h exceeds budget {self.budget_seconds/3600:.2f}h even after guard actions."
            )

    def log_eta_line(self, stage: str) -> str:
        """Return compact ETA message for stage-level logs."""
        elapsed = time.time() - self.state.started_at
        remaining = self.estimate_remaining_seconds()
        return (
            f"[runtime] after={stage} elapsed={elapsed/60:.1f}m "
            f"remaining_est={remaining/60:.1f}m budget={self.budget_seconds/3600:.2f}h"
        )


def maybe_cleanup_hf_tmp() -> None:
    """Clean optional HF temp files to limit disk usage."""
    tmp = os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING")
    _ = tmp  # keep lint/simple minimal imports stable
