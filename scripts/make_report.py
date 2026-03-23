#!/usr/bin/env python3
"""Build markdown summary and plot from collected metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.plotting import make_accuracy_plot


def main() -> None:
    cfg = load_config()
    metrics = json.loads(Path(cfg["paths"]["metrics_json"]).read_text(encoding="utf-8"))

    make_accuracy_plot(metrics, cfg["paths"]["plot_png"])

    b = metrics["baseline"]["macro_accuracy"]
    s = metrics["post_short_cpt"]["macro_accuracy"]
    r = metrics["post_replay_cpt"]["macro_accuracy"]

    short_hurt = s < b
    replay_helped = r > s
    final = (
        f"Short-only micro-CPT {'reduced' if short_hurt else 'did not reduce'} long-context retrieval; "
        f"mixed-length replay {'helped recover' if replay_helped else 'did not recover'} that loss."
    )

    md = [
        "# CI Pilot Summary",
        "",
        "## Final conclusion",
        final,
        "",
        "## Macro accuracy",
        f"- baseline: {b:.3f}",
        f"- post_short_cpt: {s:.3f}",
        f"- post_replay_cpt: {r:.3f}",
        "",
        "## Notes",
        "- This pilot uses CPU-only, 80-step max micro-CPT and tiny passkey eval for CI feasibility.",
        "- PEFT fallback is LM-head-only training to keep updates tractable on 4 vCPU runners.",
    ]

    Path(cfg["paths"]["report_md"]).write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
