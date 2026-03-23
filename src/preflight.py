"""Preflight budget checks for CI runtime safety."""

from __future__ import annotations

from typing import Any, Dict


def assert_budget_feasible(cfg: Dict[str, Any]) -> None:
    """Fail loudly if configured stage estimates exceed hard budget."""
    total_est = float(sum(cfg["runtime"]["stage_estimates_seconds"].values()))
    hard = float(cfg["runtime"]["hard_budget_seconds"])
    if total_est > hard:
        raise RuntimeError(
            f"Configured estimate ({total_est:.0f}s) exceeds hard budget ({hard:.0f}s)."
        )
