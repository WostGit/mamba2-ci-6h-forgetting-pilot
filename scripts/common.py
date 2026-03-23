"""Common IO helpers used by script entrypoints."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import append_stage_log, apply_runtime_guard, load_runtime_state, save_runtime_state


def load_metrics(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_metrics(path: str, metrics: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def stage_done(cfg: Dict[str, Any], stage: str, start_s: float) -> None:
    state = load_runtime_state(cfg)
    elapsed_total = time.time() - state.job_start_s
    stage_elapsed = time.time() - start_s
    state = apply_runtime_guard(cfg, state, stage, elapsed_total)
    save_runtime_state(cfg, state)
    remain = cfg["budget"]["hard_timeout_hours"] * 3600 - elapsed_total
    append_stage_log(
        cfg,
        {
            "stage": stage,
            "elapsed_s": elapsed_total,
            "stage_elapsed_s": stage_elapsed,
            "estimated_remaining_s": remain,
            "examples_per_length": state.examples_per_length,
            "include_8192": state.include_8192,
            "train_steps": state.train_steps,
        },
    )
    print(
        f"[runtime] stage={stage} elapsed_total={elapsed_total:.1f}s stage_elapsed={stage_elapsed:.1f}s "
        f"est_remaining={remain/3600:.2f}h "
        f"examples={state.examples_per_length} include_8192={state.include_8192} train_steps={state.train_steps}"
    )
