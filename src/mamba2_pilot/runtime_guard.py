"""Runtime guardrails for keeping CI execution under budget."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class RuntimeState:
    """Mutable state for runtime-based auto-scaling decisions."""

    examples_per_length: int
    include_8192: bool
    train_max_steps: int
    elapsed_minutes: float = 0.0


class RuntimeGuard:
    """Tracks elapsed time and applies staged downscaling when needed."""

    def __init__(
        self,
        hard_budget_hours: float,
        stage_estimates_minutes: dict[str, float],
        initial_examples: int,
        min_examples: int,
        initial_steps: int,
        min_steps: int,
        state_path: Path,
    ) -> None:
        self.hard_budget_minutes = hard_budget_hours * 60.0
        self.stage_estimates_minutes = stage_estimates_minutes
        self.min_examples = min_examples
        self.min_steps = min_steps
        self.state_path = state_path
        self._start = time.time()
        self.state = RuntimeState(
            examples_per_length=initial_examples,
            include_8192=True,
            train_max_steps=initial_steps,
        )
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.state = RuntimeState(**payload)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")

    def mark_stage_complete(self, stage_name: str) -> dict[str, Any]:
        """Update elapsed time and auto-scale if projected budget overrun is detected."""
        self.state.elapsed_minutes = (time.time() - self._start) / 60.0
        remaining_estimate = self._remaining_estimate(stage_name)
        projected_total = self.state.elapsed_minutes + remaining_estimate

        decisions: list[str] = []
        if projected_total > self.hard_budget_minutes and self.state.examples_per_length > self.min_examples:
            self.state.examples_per_length = self.min_examples
            decisions.append("Reduced examples per length to fallback minimum.")
            projected_total = self.state.elapsed_minutes + self._remaining_estimate(stage_name)

        if projected_total > self.hard_budget_minutes and self.state.include_8192:
            self.state.include_8192 = False
            decisions.append("Skipped 8192 context length.")
            projected_total = self.state.elapsed_minutes + self._remaining_estimate(stage_name)

        if projected_total > self.hard_budget_minutes and self.state.train_max_steps > self.min_steps:
            self.state.train_max_steps = self.min_steps
            decisions.append("Reduced training max steps to guard minimum.")

        self.save()
        return {
            "stage": stage_name,
            "elapsed_minutes": round(self.state.elapsed_minutes, 2),
            "estimated_remaining_minutes": round(self._remaining_estimate(stage_name), 2),
            "projected_total_minutes": round(self.state.elapsed_minutes + self._remaining_estimate(stage_name), 2),
            "decisions": decisions,
        }

    def fail_if_unrecoverable(self, current_stage: str) -> None:
        """Fail loud if even the most reduced config is still likely to exceed budget."""
        if (
            self.state.examples_per_length == self.min_examples
            and not self.state.include_8192
            and self.state.train_max_steps == self.min_steps
        ):
            projected_total = self.state.elapsed_minutes + self._remaining_estimate(current_stage)
            if projected_total > self.hard_budget_minutes:
                raise RuntimeError(
                    f"Projected runtime {projected_total:.1f}m exceeds hard budget "
                    f"{self.hard_budget_minutes:.1f}m even after all runtime guards."
                )

    def _remaining_estimate(self, current_stage: str) -> float:
        seen = False
        remaining = 0.0
        for stage, minutes in self.stage_estimates_minutes.items():
            if stage == current_stage:
                seen = True
                continue
            if seen:
                remaining += float(minutes)

        if self.state.examples_per_length == self.min_examples:
            remaining *= 0.9
        if not self.state.include_8192:
            remaining *= 0.85
        if self.state.train_max_steps == self.min_steps:
            remaining *= 0.9
        return remaining
