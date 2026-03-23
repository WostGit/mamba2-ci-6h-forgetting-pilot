#!/usr/bin/env python3
"""Generate markdown summary + PNG from final metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from mamba2_pilot.plotting import save_accuracy_plot
from mamba2_pilot.utils import read_json


def _delta(a: float, b: float) -> float:
    return round(b - a, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/all_eval_metrics.json")
    parser.add_argument("--train-short", default="outputs/train_short.json")
    parser.add_argument("--train-replay", default="outputs/train_replay.json")
    parser.add_argument("--out-md", default="outputs/summary.md")
    parser.add_argument("--out-png", default="outputs/accuracy_plot.png")
    args = parser.parse_args()

    metrics = read_json(args.metrics)
    train_short = read_json(args.train_short)
    train_replay = read_json(args.train_replay)

    save_accuracy_plot(metrics, args.out_png)

    base = float(metrics["baseline"]["overall_accuracy"])
    short = float(metrics["post_short_cpt"]["overall_accuracy"])
    replay = float(metrics["post_replay_cpt"]["overall_accuracy"])

    short_hurt = short < base
    replay_helped = replay > short

    lines = [
        "# Mamba-2 130M micro-CPT forgetting pilot",
        "",
        f"- Baseline overall accuracy: **{base:.3f}**",
        f"- Post short-only CPT: **{short:.3f}** (delta { _delta(base, short):+.3f})",
        f"- Post replay CPT: **{replay:.3f}** (delta vs short { _delta(short, replay):+.3f})",
        "",
        "## Final CI conclusion",
        f"- Did short-only CPT reduce long-context retrieval? **{'Yes' if short_hurt else 'No'}**",
        f"- Did mixed-length replay help vs short-only CPT? **{'Yes' if replay_helped else 'No'}**",
        "",
        "## Runtime notes",
        f"- Short CPT wall time: {train_short['wall_seconds']:.1f}s",
        f"- Replay CPT wall time: {train_replay['wall_seconds']:.1f}s",
        f"- Guard actions observed: {metrics.get('guard_actions', [])}",
    ]

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
