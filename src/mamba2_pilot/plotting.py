"""Plotting utilities for CI-sized experiment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def save_accuracy_plot(eval_json: Dict, out_png: str | Path) -> None:
    """Write a compact PNG with one line per eval round."""
    rounds = ["baseline", "post_short_cpt", "post_replay_cpt"]
    plt.figure(figsize=(7, 4))

    for r in rounds:
        by_len = eval_json[r]["by_length"]
        xs = sorted(int(k) for k in by_len.keys())
        ys = [by_len[str(x)]["accuracy"] for x in xs]
        plt.plot(xs, ys, marker="o", label=r)

    plt.xlabel("Context length (tokens)")
    plt.ylabel("Passkey retrieval accuracy")
    plt.ylim(-0.05, 1.05)
    plt.title("Mamba-2 130M: short-CPT forgetting pilot")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()
