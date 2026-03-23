"""Runtime, seed, and artifact helpers."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class RuntimeState:
    """Mutable runtime knobs that can be scaled down to stay inside CI budget."""

    examples_per_length: int
    include_8192: bool
    train_steps: int
    job_start_s: float


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible CPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now_s() -> float:
    """Current timestamp in seconds."""
    return time.time()


def ensure_parent(path: str | Path) -> None:
    """Create parent directory for a file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_runtime_state(cfg: Dict[str, Any]) -> RuntimeState:
    """Load runtime state from disk if present; otherwise initialize defaults."""
    state_path = Path(cfg["paths"]["runtime_state_json"])
    if state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if "job_start_s" not in data:
            data["job_start_s"] = now_s()
        return RuntimeState(**data)
    return RuntimeState(
        examples_per_length=cfg["evaluation"]["examples_per_length"],
        include_8192=True,
        train_steps=cfg["training"]["max_steps"],
        job_start_s=now_s(),
    )


def save_runtime_state(cfg: Dict[str, Any], state: RuntimeState) -> None:
    """Persist runtime state so resumed jobs keep scale-down decisions."""
    out = Path(cfg["paths"]["runtime_state_json"])
    ensure_parent(out)
    out.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")


def append_stage_log(cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Append a JSONL stage log entry."""
    path = Path(cfg["paths"]["stage_log_jsonl"])
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def apply_runtime_guard(
    cfg: Dict[str, Any],
    state: RuntimeState,
    stage_name: str,
    elapsed_s: float,
) -> RuntimeState:
    """Scale down work in a fixed sequence when runtime risk is high.

    Order: reduce eval examples -> skip 8192 context -> reduce train steps.
    """
    budget_s = cfg["budget"]["hard_timeout_hours"] * 3600
    estimates = cfg["budget"]["stage_estimates_seconds"]
    stage_order = [
        "baseline_eval",
        "train_short",
        "eval_short",
        "train_replay",
        "eval_replay",
    ]
    try:
        idx = stage_order.index(stage_name)
    except ValueError:
        idx = -1
    remaining_est = sum(estimates[s] for s in stage_order[idx + 1 :] if s in estimates)

    # Fail loudly if even the smallest setup likely cannot finish.
    min_remaining = 2 * 1000 + 2 * 2800
    if elapsed_s + min_remaining > budget_s:
        raise RuntimeError(
            "Runtime budget likely exceeded even at minimum settings; aborting early."
        )

    over_budget = elapsed_s + remaining_est > budget_s
    if over_budget and state.examples_per_length > cfg["evaluation"]["fallback_examples_per_length"]:
        state.examples_per_length = cfg["evaluation"]["fallback_examples_per_length"]
    elif over_budget and state.include_8192:
        state.include_8192 = False
    elif over_budget and state.train_steps > cfg["training"]["fallback_steps"]:
        state.train_steps = cfg["training"]["fallback_steps"]

    return state
