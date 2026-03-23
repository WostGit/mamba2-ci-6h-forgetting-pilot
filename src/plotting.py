"""Plotting helpers for passkey results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def make_accuracy_plot(metrics: Dict[str, Any], out_png: str) -> None:
    """Create accuracy-vs-context-length plot for all conditions."""
    plt.figure(figsize=(7, 4))
    for condition in ["baseline", "post_short_cpt", "post_replay_cpt"]:
        if condition not in metrics:
            continue
        per = metrics[condition]["per_length"]
        xs = sorted(int(k) for k in per.keys())
        ys = [per[str(x)]["accuracy"] for x in xs]
        plt.plot(xs, ys, marker="o", label=condition)

    plt.xlabel("Context length")
    plt.ylabel("Passkey exact-match accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
