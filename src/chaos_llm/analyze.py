import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chaos_llm.analysis.config import load_analysis_config
from chaos_llm.analysis.data import discover_runs, load_run_metadata, load_tokens
from chaos_llm.analysis.divergence import divergence_any_pair, divergence_pairwise, divergence_vs_baseline
from chaos_llm.analysis.plots import apply_style, plot_dependency_curves, plot_histogram


def _filter_prompts(meta: Dict[str, Any], include: Optional[List[str]], exclude: Optional[List[str]]) -> bool:
    name = meta.get("prompt", {}).get("name", "unknown")
    if include and name not in include:
        return False
    if exclude and name in exclude:
        return False
    return True


def _extract_runtime(meta: Dict[str, Any]) -> Tuple[str, str, int]:
    runtime = meta.get("runtime", {})
    window = str(runtime.get("sliding_window", "unknown"))
    mag = str(runtime.get("perturbation_magnitude", "unknown"))
    prompt_len = int(runtime.get("prompt_len", -1))
    return window, mag, prompt_len


def _write_summary(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_outputs(base_path: str, formats: List[str]) -> List[str]:
    return [f"{base_path}.{ext}" for ext in formats]


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mode(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    uniques, counts = np.unique(values, return_counts=True)
    if uniques.size == 0:
        return None
    max_count = counts.max()
    candidates = uniques[counts == max_count]
    return float(candidates.min())


def _select_primary_values(
    primary_metric: str,
    any_pair_value: Optional[int],
    baseline_divergence: Optional[np.ndarray],
    pairwise_divergence: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    if primary_metric == "baseline_per_sequence" and baseline_divergence is not None:
        return baseline_divergence, "baseline"
    if primary_metric == "pairwise" and pairwise_divergence is not None:
        return pairwise_divergence, "pairwise"
    if primary_metric == "any_pair" and any_pair_value is not None:
        return np.array([any_pair_value], dtype=np.int32), "any_pair"
    return None, "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze chaos-llm outputs")
    parser.add_argument("--config", required=True, help="Path to analysis.yaml")
    args = parser.parse_args()

    cfg = load_analysis_config(args.config)
    apply_style(cfg)

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    run_list = cfg["paths"]["run_list"]

    runs = discover_runs(input_dir, run_list)
    if not runs:
        raise ValueError("No run folders found")

    include = cfg["prompt_filter"]["include"]
    exclude = cfg["prompt_filter"]["exclude"]

    summary_rows: List[Dict[str, Any]] = []
    per_prompt_values: Dict[str, List[int]] = {}
    per_run_stats: List[Dict[str, Any]] = []

    for run_dir in runs:
        meta = load_run_metadata(run_dir)
        if not _filter_prompts(meta, include, exclude):
            continue

        tokens = load_tokens(run_dir, mmap_mode=cfg["performance"]["mmap_mode"])
        baseline_ids = tokens["baseline_ids"]
        perturbed_ids = tokens["perturbed_ids"]
        lengths = tokens["perturbed_lengths"]
        prompt_len = int(tokens.get("prompt_len", meta.get("runtime", {}).get("prompt_len", 0)))

        exclude_baseline = bool(cfg["divergence"].get("exclude_baseline", False))

        any_pair_value = None
        if cfg["divergence"]["any_pair"]:
            any_pair_value = divergence_any_pair(
                perturbed_ids=perturbed_ids,
                lengths=lengths,
                prompt_len=prompt_len,
                include_baseline=cfg["divergence"]["include_baseline_in_any_pair"] and not exclude_baseline,
                baseline_ids=None if exclude_baseline else baseline_ids,
                index_reference=cfg["divergence"]["index_reference"],
            )

        baseline_divergence = None
        if cfg["divergence"]["baseline_per_sequence"] and not exclude_baseline:
            baseline_divergence = divergence_vs_baseline(
                perturbed_ids=perturbed_ids,
                lengths=lengths,
                baseline_ids=baseline_ids,
                prompt_len=prompt_len,
                index_reference=cfg["divergence"]["index_reference"],
            )

        pairwise_divergence = None
        if cfg["divergence"]["pairwise"]:
            pairwise_divergence = divergence_pairwise(
                perturbed_ids=perturbed_ids,
                lengths=lengths,
                prompt_len=prompt_len,
                index_reference=cfg["divergence"]["index_reference"],
                max_pairs=cfg["divergence"]["pairwise_max_pairs"],
            )

        window, mag, prompt_len_meta = _extract_runtime(meta)
        prompt_name = meta.get("prompt", {}).get("name", "unknown")
        run_name = os.path.basename(run_dir)

        row: Dict[str, Any] = {
            "run": run_name,
            "prompt": prompt_name,
            "sliding_window": window,
            "perturbation_magnitude": mag,
            "prompt_len": prompt_len,
            "num_sequences": int(perturbed_ids.shape[0]),
            "any_pair_divergence": any_pair_value if any_pair_value is not None else "",
        }

        if baseline_divergence is not None:
            values = baseline_divergence
            no_div = cfg["divergence"]["no_divergence_value"]
            keep = values != no_div
            filtered = values[keep]
            row["baseline_no_divergence"] = int((~keep).sum())
            if filtered.size:
                row["baseline_mean"] = float(filtered.mean())
                row["baseline_std"] = float(filtered.std(ddof=1)) if filtered.size > 1 else 0.0
                row["baseline_median"] = float(np.median(filtered))
                row["baseline_mode"] = _mode(filtered)
                for q in cfg["summary"]["quantiles"]:
                    row[f"baseline_q{int(q*100):02d}"] = float(np.quantile(filtered, q))
            else:
                row["baseline_mean"] = ""
                row["baseline_std"] = ""
                row["baseline_median"] = ""
                row["baseline_mode"] = ""

            if cfg["divergence"]["primary_metric"] == "baseline_per_sequence":
                per_prompt_values.setdefault(prompt_name, []).extend(filtered.tolist())

            if cfg["divergence"]["primary_metric"] == "baseline_per_sequence":
                per_run_stats.append(
                    {
                        "prompt": prompt_name,
                        "sliding_window": _to_float(window),
                        "perturbation_magnitude": _to_float(mag),
                        "metric": "baseline",
                        "mean": float(filtered.mean()) if filtered.size else None,
                        "median": float(np.median(filtered)) if filtered.size else None,
                        "mode": _mode(filtered),
                        "std": float(filtered.std(ddof=1)) if filtered.size > 1 else 0.0,
                        "var": float(filtered.var(ddof=1)) if filtered.size > 1 else 0.0,
                    }
                )

        if pairwise_divergence is not None:
            values = pairwise_divergence
            row["pairwise_num_pairs"] = int(values.shape[0])
            no_div = cfg["divergence"]["no_divergence_value"]
            keep = values != no_div
            filtered = values[keep]
            row["pairwise_no_divergence"] = int((~keep).sum())
            if filtered.size:
                row["pairwise_mean"] = float(filtered.mean())
                row["pairwise_std"] = float(filtered.std(ddof=1)) if filtered.size > 1 else 0.0
                row["pairwise_median"] = float(np.median(filtered))
                row["pairwise_mode"] = _mode(filtered)
                for q in cfg["summary"]["quantiles"]:
                    row[f"pairwise_q{int(q*100):02d}"] = float(np.quantile(filtered, q))
            else:
                row["pairwise_mean"] = ""
                row["pairwise_std"] = ""
                row["pairwise_median"] = ""
                row["pairwise_mode"] = ""

            if cfg["divergence"]["primary_metric"] == "pairwise":
                per_prompt_values.setdefault(prompt_name, []).extend(filtered.tolist())

            if cfg["divergence"]["primary_metric"] == "pairwise":
                per_run_stats.append(
                    {
                        "prompt": prompt_name,
                        "sliding_window": _to_float(window),
                        "perturbation_magnitude": _to_float(mag),
                        "metric": "pairwise",
                        "mean": float(filtered.mean()) if filtered.size else None,
                        "median": float(np.median(filtered)) if filtered.size else None,
                        "mode": _mode(filtered),
                        "std": float(filtered.std(ddof=1)) if filtered.size > 1 else 0.0,
                        "var": float(filtered.var(ddof=1)) if filtered.size > 1 else 0.0,
                    }
                )

        summary_rows.append(row)

        if cfg["plots"]["enabled"] and cfg["plots"]["per_run"]:
            plot_values, label = _select_primary_values(
                cfg["divergence"]["primary_metric"],
                any_pair_value,
                baseline_divergence,
                pairwise_divergence,
            )
            if cfg["divergence"]["exclude_no_divergence_from_plots"]:
                if plot_values is not None:
                    plot_values = plot_values[plot_values != cfg["divergence"]["no_divergence_value"]]
            if plot_values is not None and plot_values.size:
                title = f"{cfg['plots']['title_prefix']} ({label}) - {run_name}"
                output_base = os.path.join(output_dir, "figures", f"hist_{label}_{run_name}")
                output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                plot_histogram(
                    values=plot_values,
                    title=title,
                    xlabel="divergence index",
                    output_paths=output_paths,
                    bins=int(cfg["plots"]["bins"]),
                    grid=bool(cfg["plots"]["grid"]),
                    xlim=cfg["plots"]["xlim"],
                    ylim=cfg["plots"]["ylim"],
                )

    if not summary_rows:
        raise ValueError("No runs matched the prompt filters")

    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    _write_summary(os.path.join(output_dir, "summary.csv"), summary_rows, fieldnames)

    if cfg["plots"]["enabled"] and cfg["plots"]["per_prompt"]:
        for prompt_name, values in per_prompt_values.items():
            if not values:
                continue
            label = cfg["divergence"]["primary_metric"]
            title = f"{cfg['plots']['title_prefix']} ({label}) - {prompt_name}"
            output_base = os.path.join(output_dir, "figures", f"hist_prompt_{label}_{prompt_name}")
            output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
            plot_histogram(
                values=np.array(values, dtype=np.int32),
                title=title,
                xlabel="divergence index",
                output_paths=output_paths,
                bins=int(cfg["plots"]["bins"]),
                grid=bool(cfg["plots"]["grid"]),
                xlim=cfg["plots"]["xlim"],
                ylim=cfg["plots"]["ylim"],
            )

    dep_cfg = cfg["plots"].get("dependencies", {})
    if cfg["plots"]["enabled"] and dep_cfg.get("enabled", True):
        metrics = dep_cfg.get("metrics", ["mean"])
        error_bars = dep_cfg.get("error_bars", "std")
        per_prompt = dep_cfg.get("per_prompt", True)
        x_axes = dep_cfg.get("x_axis", ["sliding_window", "perturbation_magnitude"])

        def build_series(
            x_key: str,
            line_key: str,
            prompt_name: Optional[str],
            metric_name: str,
            use_error: bool,
            linestyle: str,
        ) -> Dict[str, Dict[str, List[float]]]:
            series: Dict[str, Dict[str, List[float]]] = {}
            for row in per_run_stats:
                if row.get("metric") != cfg["divergence"]["primary_metric"]:
                    continue
                if prompt_name is not None and row.get("prompt") != prompt_name:
                    continue
                x_val = row.get(x_key)
                line_val = row.get(line_key)
                y_val = row.get(metric_name)
                y_err = row.get(error_bars) if use_error else None
                if x_val is None or line_val is None:
                    continue
                label = f"{line_key}={line_val} ({metric_name})"
                series.setdefault(label, {"x": [], "y": [], "yerr": [], "linestyle": linestyle})
                series[label]["x"].append(x_val)
                series[label]["y"].append(y_val)
                if use_error:
                    series[label]["yerr"].append(y_err)

            for label, data in series.items():
                points = list(zip(data["x"], data["y"], data["yerr"]))
                points = [p for p in points if p[1] is not None]
                points.sort(key=lambda p: p[0])
                data["x"] = [p[0] for p in points]
                data["y"] = [p[1] for p in points]
                if use_error:
                    data["yerr"] = [p[2] for p in points]
                else:
                    data.pop("yerr", None)
            return series

        prompts = sorted({row["prompt"] for row in per_run_stats})
        if not per_prompt:
            prompts = [None]

        for prompt_name in prompts:
            suffix = f"_{prompt_name}" if prompt_name else ""

            if "sliding_window" in x_axes:
                series: Dict[str, Dict[str, List[float]]] = {}
                for metric in metrics:
                    use_error = metric == "mean" and error_bars in ("std", "var")
                    linestyle = "-" if metric == "mean" else ("--" if metric == "median" else ":")
                    series.update(
                        build_series(
                            "sliding_window",
                            "perturbation_magnitude",
                            prompt_name,
                            metric,
                            use_error,
                            linestyle,
                        )
                    )
                title = f"{cfg['plots']['title_prefix']} (mean/median/mode){suffix}"
                output_base = os.path.join(output_dir, "figures", f"dep_window_mean_median_mode{suffix}")
                output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                plot_dependency_curves(
                    series=series,
                    title=title,
                    xlabel="sliding window",
                    ylabel="divergence index",
                    output_paths=output_paths,
                    grid=bool(cfg["plots"]["grid"]),
                    color_map=str(cfg["plots"]["color_map"]),
                )

            if "perturbation_magnitude" in x_axes:
                series = {}
                for metric in metrics:
                    use_error = metric == "mean" and error_bars in ("std", "var")
                    linestyle = "-" if metric == "mean" else ("--" if metric == "median" else ":")
                    series.update(
                        build_series(
                            "perturbation_magnitude",
                            "sliding_window",
                            prompt_name,
                            metric,
                            use_error,
                            linestyle,
                        )
                    )
                title = f"{cfg['plots']['title_prefix']} (mean/median/mode){suffix}"
                output_base = os.path.join(output_dir, "figures", f"dep_magnitude_mean_median_mode{suffix}")
                output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                plot_dependency_curves(
                    series=series,
                    title=title,
                    xlabel="perturbation magnitude",
                    ylabel="divergence index",
                    output_paths=output_paths,
                    grid=bool(cfg["plots"]["grid"]),
                    color_map=str(cfg["plots"]["color_map"]),
                )


if __name__ == "__main__":
    main()
