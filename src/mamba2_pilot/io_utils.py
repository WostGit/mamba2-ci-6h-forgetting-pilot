"""I/O helpers for stage logging and resumable artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_runtime_stage(log_path: Path, rec: dict) -> None:
    """Append one runtime-stage record for transparency in CI logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if log_path.exists():
        existing = json.loads(log_path.read_text(encoding="utf-8"))
    existing.append(rec)
    log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def clean_temp_dir(temp_dir: Path) -> None:
    """Drop temporary files to keep disk pressure low on ephemeral runners."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
