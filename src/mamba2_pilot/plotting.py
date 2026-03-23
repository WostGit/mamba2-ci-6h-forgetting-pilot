"""Plotting helpers for final report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_accuracy_curves(metric_files: list[Path], out_png: Path) -> None:
    """Plot retrieval accuracy vs context length for each eval stage."""
    plt.figure(figsize=(7, 4.5))
    for mf in metric_files:
        payload = json.loads(mf.read_text(encoding="utf-8"))
        xs = sorted(int(k) for k in payload["by_length"].keys())
        ys = [payload["by_length"][str(x)]["accuracy"] for x in xs]
        plt.plot(xs, ys, marker="o", label=payload["stage"])

    plt.xscale("log", base=2)
    plt.xticks(xs, xs)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Context length")
    plt.ylabel("Passkey retrieval accuracy")
    plt.title("Mamba-2 130M: passkey retrieval under micro-CPT")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
