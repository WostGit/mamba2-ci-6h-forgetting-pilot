"""Runtime guard to keep CI execution within strict wall-clock limits."""

from __future__ import annotations

import time
from typing import Any, Dict

from src.config import RuntimeState


class RuntimeGuard:
    """Estimates remaining time and applies ordered scale-down decisions."""

    def __init__(self, cfg: Dict[str, Any], state: RuntimeState) -> None:
        self.cfg = cfg
        self.state = state
        self.started_at = time.time()

    def apply_after_stage(self, stage_name: str, stage_seconds: float) -> str:
        """Update elapsed runtime, log estimate, and apply degradations if needed."""
        self.state.elapsed_seconds += stage_seconds
        hard_budget = self.cfg["runtime"]["hard_budget_seconds"]
        est_remaining = max(0.0, hard_budget - self.state.elapsed_seconds)

        upcoming = self._estimated_upcoming_seconds(stage_name)
        note = (
            f"[runtime] stage={stage_name} elapsed_total={self.state.elapsed_seconds:.1f}s "
            f"remaining={est_remaining:.1f}s est_upcoming={upcoming:.1f}s"
        )

        # Runtime-saving decisions are intentionally ordered per user requirement.
        if upcoming > est_remaining and self.state.examples_per_length > self.cfg["eval"]["fallback_examples_per_length"]:
            self.state.examples_per_length = self.cfg["eval"]["fallback_examples_per_length"]
            note += " | guard: examples_per_length -> 2"
        if self._estimated_upcoming_seconds(stage_name) > est_remaining and self.state.include_8192:
            self.state.include_8192 = False
            note += " | guard: dropped_8192"
        if self._estimated_upcoming_seconds(stage_name) > est_remaining and self.state.train_steps > self.cfg["train"]["fallback_steps"]:
            self.state.train_steps = self.cfg["train"]["fallback_steps"]
            note += " | guard: train_steps -> 60"

        if self._estimated_upcoming_seconds(stage_name) > est_remaining:
            raise RuntimeError(
                "Projected runtime exceeds hard budget even after guard scale-down. "
                "Failing loudly to protect CI ceiling."
            )
        return note

    def _estimated_upcoming_seconds(self, completed_stage: str) -> float:
        estimates = self.cfg["runtime"]["stage_estimates_seconds"]
        seen = False
        total = 0.0
        for name, seconds in estimates.items():
            if seen:
                total += float(seconds)
            if name == completed_stage:
                seen = True
        if not seen:
            total = float(sum(estimates.values()))
        # Cheap scaling heuristic for reduced eval/training workload.
        eval_factor = self.state.examples_per_length / self.cfg["eval"]["examples_per_length"]
        len_factor = 0.75 if not self.state.include_8192 else 1.0
        step_factor = self.state.train_steps / self.cfg["train"]["steps"]
        return total * eval_factor * len_factor * step_factor
