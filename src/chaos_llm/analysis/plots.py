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
    series: Dict[str, Dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_paths: List[str],
    grid: bool,
    color_map: str,
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap(color_map)

    for idx, (label, data) in enumerate(series.items()):
        x_vals = data["x"]
        y_vals = data.get("y") or data.get("y_median")
        y_err = data.get("yerr")
        linestyle = data.get("linestyle", "-")
        color = cmap(idx % cmap.N)
        
        # Plot main curve
        ax.plot(x_vals, y_vals, marker="o", linestyle=linestyle, label=label, color=color, linewidth=2)
        
        # Error bars (standard)
        if y_err is not None:
            ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='none', color=color, capsize=3, alpha=0.5)
            
        # Fan bands (percentiles)
        if "fans" in data and data["fans"]:
            fan_dict = data["fans"]
            keys = sorted(fan_dict.keys()) # e.g. ["q05", "q25", "q75", "q95"]
            num_bands = len(keys) // 2
            for i in range(num_bands):
                low_key = keys[i]
                high_key = keys[-(i+1)]
                alpha = 0.1 + (i / num_bands) * 0.2 # progressive alpha
                ax.fill_between(x_vals, fan_dict[low_key], fan_dict[high_key], color=color, alpha=alpha)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    # Improve log scale limits if needed
    if yscale == "log":
        all_y = []
        for data in series.values():
            y = data.get("y") or data.get("y_median")
            if y is not None: all_y.extend(np.array(y).flatten())
            if "fans" in data:
                for f_vals in data["fans"].values():
                    all_y.extend(np.array(f_vals).flatten())
            if "yerr" in data and y is not None:
                all_y.extend((np.array(y) - np.array(data["yerr"])).flatten())
        
        all_y = np.array(all_y)
        pos_y = all_y[(all_y > 0) & np.isfinite(all_y)]
        if pos_y.size > 0:
            ymin = pos_y.min()
            ymax = pos_y.max()
            ax.set_ylim(ymin * 0.8, ymax * 1.2)

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


def plot_time_series(
    x: np.ndarray,
    mean: np.ndarray,
    median: np.ndarray,
    std: Optional[np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    output_paths: List[str],
    grid: bool,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, mean, color="#1f77b4", label="mean")
    ax.plot(x, median, color="#ff7f0e", linestyle="--", label="median")
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.2)

    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False)

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)




def plot_survival_curves(
    series: Dict[str, Dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_paths: List[str],
    grid: bool,
    color_map: str,
    yscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap(color_map)

    for idx, (label, data) in enumerate(series.items()):
        color = cmap(idx % cmap.N)
        ax.plot(data["x"], data["y"], label=label, color=color)

    ax.set_yscale(yscale)
    if yscale == "log":
        all_y = []
        for data in series.values():
            all_y.extend(np.array(data["y"]).flatten())
        all_y = np.array(all_y)
        pos_y = all_y[(all_y > 0) & np.isfinite(all_y)]
        if pos_y.size > 0:
            ymin = min(pos_y.min(), 0.01)
            ax.set_ylim(ymin * 0.8, 1.1)
        else:
            ax.set_ylim(1e-3, 1.1)

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
