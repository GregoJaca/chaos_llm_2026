import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def apply_style(cfg: Dict) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": cfg["plots"]["dpi"],
            "savefig.dpi": cfg["plots"]["dpi"],
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_paths: List[str],
    bins: int,
    grid: bool,
    xlim: Optional[List[float]],
    ylim: Optional[List[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(values, bins=bins, color="#2a6f97", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if grid:
        ax.grid(True, linestyle=":", alpha=0.5)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dependency_curves(
    series: Dict[str, Dict[str, List[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_paths: List[str],
    grid: bool,
    color_map: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap(color_map)

    for idx, (label, data) in enumerate(series.items()):
        x_vals = data["x"]
        y_vals = data["y"]
        y_err = data.get("yerr")
        linestyle = data.get("linestyle", "-")
        color = cmap(idx % cmap.N)
        if y_err is not None:
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                marker="o",
                linestyle=linestyle,
                label=label,
                color=color,
            )
        else:
            ax.plot(x_vals, y_vals, marker="o", linestyle=linestyle, label=label, color=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, linestyle=":", alpha=0.5)
    if series:
        ax.legend(frameon=False)

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
