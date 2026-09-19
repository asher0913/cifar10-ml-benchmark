"""Plotting helpers storing figures under outputs directory.

The plots generate a line plot for curves of accuracy of 
different feature dimensions and a bar plot for
discrete hyper-parameter comparisons.
They write PNGs to outputs folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .utils import ensure_dir


def plot_metric_curve(
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Line plot with point annotations for sweep results."""
    if not xs:
        return
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    for x, y, label in zip(xs, ys, labels):
        # Draw a short label near each point to make sweeps readable without a legend.
        plt.annotate(label, (x, y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_param_bar(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Bar chart to compare discrete hyper-parameter configurations."""
    if not labels:
        return
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    positions = range(len(labels))
    plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
