"""Plotting helper for passkey accuracy curves."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def make_accuracy_plot(metrics: Dict[str, Dict], out_png: str) -> None:
    """Create compact PNG for CI artifact upload."""
    plt.figure(figsize=(6, 4))
    for label, payload in metrics.items():
        lengths: List[int] = [int(x) for x in payload["by_length"].keys()]
        acc = [payload["by_length"][str(l)]["accuracy"] for l in lengths]
        plt.plot(lengths, acc, marker="o", label=label)

    plt.xscale("log", base=2)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Context length (tokens)")
    plt.ylabel("Passkey retrieval accuracy")
    plt.title("Mamba-2 130M passkey retrieval (CI pilot)")
    plt.legend()
    plt.grid(alpha=0.3)
    p = Path(out_png)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=140)
    plt.close()
