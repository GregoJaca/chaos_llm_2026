import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    split_y_for_stable: bool = False,
) -> None:
    has_stable = False
    if split_y_for_stable:
        for label, data in series.items():
            y_vals = data.get("y") or data.get("y_median")
            if y_vals is not None:
                if np.any(np.isinf(y_vals)):
                    has_stable = True
                    break

    if has_stable:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(7.5, 5.5), gridspec_kw={"height_ratios": [0.15, 0.85]}
        )
        plt.subplots_adjust(hspace=0.08)
        cmap = plt.get_cmap(color_map)

        # Determine y limits for finite values
        all_y_vals = []
        for label, data in series.items():
            y = data.get("y") or data.get("y_median")
            if y is not None:
                all_y_vals.extend(y)
            if "fans" in data and data["fans"]:
                for q_vals in data["fans"].values():
                    all_y_vals.extend(q_vals)
        finite_y = np.array([v for v in all_y_vals if np.isfinite(v) and v is not None])
        if finite_y.size > 0:
            max_finite_y = finite_y.max()
            min_finite_y = finite_y.min()
        else:
            max_finite_y = 1.0
            min_finite_y = 0.0

        padding = (max_finite_y - min_finite_y) * 0.1 if max_finite_y > min_finite_y else 1.0
        if padding == 0:
            padding = 1.0
        ax_bot.set_ylim(max(0.0, min_finite_y - padding), max_finite_y + padding)

        ax_top.set_ylim(0.9, 1.1)
        ax_top.set_yticks([1.0])
        ax_top.set_yticklabels(["stable"])
        ax_top.patch.set_facecolor("#f0f0f0")

        # Hide spines between subplots
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)
        ax_bot.xaxis.tick_bottom()

        # Diagonal break lines
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        for idx, (label, data) in enumerate(series.items()):
            x_vals = np.array(data["x"])
            y_vals = np.array(data.get("y") or data.get("y_median"))
            y_err = data.get("yerr")
            linestyle = data.get("linestyle", "-")
            color = cmap(idx % cmap.N)
            is_fan = "fans" in data and data["fans"]

            finite_mask = np.isfinite(y_vals)
            inf_mask = ~finite_mask

            y_bot = np.where(finite_mask, y_vals, np.nan)
            ax_bot.plot(x_vals, y_bot, marker="o", linestyle=linestyle, label=label, color=color, linewidth=2)

            if np.any(inf_mask):
                x_inf = x_vals[inf_mask]
                y_inf = np.ones_like(x_inf)
                ax_top.plot(x_inf, y_inf, marker="^", color="red", linestyle="None", markersize=8)

            if y_err is not None:
                y_err_finite = np.where(finite_mask, y_err, 0.0)
                ax_bot.errorbar(x_vals, y_bot, yerr=y_err_finite, fmt='none', color=color, capsize=3, alpha=0.5)

            if is_fan:
                fan_dict = data["fans"]
                keys = sorted(fan_dict.keys())
                num_bands = len(keys) // 2
                _, ymax_bot = ax_bot.get_ylim()
                for i in range(num_bands):
                    low_key = keys[i]
                    high_key = keys[-(i+1)]
                    alpha = 0.1 + (i / num_bands) * 0.2

                    low_vals = np.array(fan_dict[low_key])
                    high_vals = np.array(fan_dict[high_key])

                    low_vals_capped = np.where(np.isfinite(low_vals), low_vals, ymax_bot)
                    high_vals_capped = np.where(np.isfinite(high_vals), high_vals, ymax_bot)

                    ax_bot.fill_between(x_vals, low_vals_capped, high_vals_capped, color=color, alpha=alpha)

        ax_bot.set_xscale(xscale)
        ax_bot.set_yscale(yscale)

        ax_top.set_title(title)
        ax_bot.set_xlabel(xlabel)
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=12)

        if grid:
            ax_top.grid(True, linestyle=":", alpha=0.5)
            ax_bot.grid(True, linestyle=":", alpha=0.5)

        if series:
            ax_bot.legend(frameon=False)

    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cmap = plt.get_cmap(color_map)

        for idx, (label, data) in enumerate(series.items()):
            x_vals = data["x"]
            y_vals = data.get("y") or data.get("y_median")
            y_err = data.get("yerr")
            linestyle = data.get("linestyle", "-")
            color = cmap(idx % cmap.N)

            ax.plot(x_vals, y_vals, marker="o", linestyle=linestyle, label=label, color=color, linewidth=2)

            if y_err is not None:
                ax.errorbar(x_vals, y_vals, yerr=y_err, fmt='none', color=color, capsize=3, alpha=0.5)

            if "fans" in data and data["fans"]:
                fan_dict = data["fans"]
                keys = sorted(fan_dict.keys())
                num_bands = len(keys) // 2
                for i in range(num_bands):
                    low_key = keys[i]
                    high_key = keys[-(i+1)]
                    alpha = 0.1 + (i / num_bands) * 0.2
                    ax.fill_between(x_vals, fan_dict[low_key], fan_dict[high_key], color=color, alpha=alpha)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

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
    xscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, mean, color="#1f77b4", label="mean")
    ax.plot(x, median, color="#ff7f0e", linestyle="--", label="median")
    if std is not None:
        ax.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.2)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
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
    xscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap(color_map)

    for idx, (label, data) in enumerate(series.items()):
        color = cmap(idx % cmap.N)
        ax.plot(data["x"], data["y"], label=label, color=color)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
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
