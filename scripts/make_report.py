#!/usr/bin/env python3
"""Build final markdown + plot summary from JSON metrics."""

from __future__ import annotations

import json
from pathlib import Path

from mamba2_pilot.config import load_config
from mamba2_pilot.plotting import plot_accuracy_curves


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _avg_acc(payload: dict) -> float:
    vals = [v["accuracy"] for v in payload["by_length"].values()]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    cfg = load_config()
    paths_cfg = cfg.section("paths")
    metrics_dir = Path(paths_cfg["metrics_dir"])
    report_dir = Path(paths_cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load(metrics_dir / "baseline_eval.json")
    post_short = _load(metrics_dir / "post_short_eval.json")
    post_replay = _load(metrics_dir / "post_replay_eval.json")

    plot_path = report_dir / "accuracy_vs_context.png"
    plot_accuracy_curves(
        [metrics_dir / "baseline_eval.json", metrics_dir / "post_short_eval.json", metrics_dir / "post_replay_eval.json"],
        plot_path,
    )

    b = _avg_acc(baseline)
    s = _avg_acc(post_short)
    r = _avg_acc(post_replay)
    short_hurt = s < b
    replay_helped = r > s

    final_summary = {
        "baseline_avg_accuracy": b,
        "post_short_avg_accuracy": s,
        "post_replay_avg_accuracy": r,
        "short_only_reduced_long_context_retrieval": short_hurt,
        "replay_helped_relative_to_short_only": replay_helped,
    }
    (report_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    md = f"""# CI-safe Mamba-2 Pilot Summary

- Baseline average retrieval accuracy: **{b:.3f}**
- Post short-only micro-CPT average retrieval accuracy: **{s:.3f}**
- Post replay micro-CPT average retrieval accuracy: **{r:.3f}**

## Final verdict

- Did short-only CPT reduce long-context retrieval? **{short_hurt}**
- Did mixed-length replay help vs short-only? **{replay_helped}**

See `accuracy_vs_context.png` for the curve and `final_summary.json` for machine-readable output.
"""
    (report_dir / "summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
