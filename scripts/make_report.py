#!/usr/bin/env python3
"""Generate markdown summary and PNG plot from metrics."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import load_config, load_or_init_runtime_state, save_runtime_state
from src.plotting import make_accuracy_plot
from src.runtime_guard import RuntimeGuard
from src.utils import load_json


def mean_accuracy(payload: dict) -> float:
    vals = [v["accuracy"] for v in payload["by_length"].values()]
    return sum(vals) / max(1, len(vals))


def main() -> None:
    cfg = load_config()
    state = load_or_init_runtime_state(cfg)
    guard = RuntimeGuard(cfg, state)

    t0 = time.time()
    metrics_path = Path(cfg["output_dir"]) / "metrics.json"
    metrics = load_json(metrics_path)

    make_accuracy_plot(
        {
            "baseline": metrics["baseline"],
            "post_short_cpt": metrics["post_short_cpt"],
            "post_replay_cpt": metrics["post_replay_cpt"],
        },
        out_png=Path(cfg["output_dir"]) / "accuracy_vs_context.png",
    )

    b = mean_accuracy(metrics["baseline"])
    s = mean_accuracy(metrics["post_short_cpt"])
    r = mean_accuracy(metrics["post_replay_cpt"])

    harmed = s < b
    helped = r > s
    summary = (
        "# CI Pilot Summary\n\n"
        f"- Baseline mean accuracy: {b:.3f}\n"
        f"- Post short-only micro-CPT mean accuracy: {s:.3f}\n"
        f"- Post replay micro-CPT mean accuracy: {r:.3f}\n\n"
        f"Conclusion: short-only CPT {'reduced' if harmed else 'did not reduce'} long-context retrieval; "
        f"replay {'helped' if helped else 'did not help'}.\n"
    )
    (Path(cfg["output_dir"]) / "summary.md").write_text(summary, encoding="utf-8")

    note = guard.apply_after_stage("report", time.time() - t0)
    print(note)
    save_runtime_state(cfg, state)


if __name__ == "__main__":
    main()
