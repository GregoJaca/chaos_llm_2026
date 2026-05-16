import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

from chaos_llm.analysis.config import load_analysis_config
from chaos_llm.analysis.data import discover_runs, load_run_metadata, load_tokens
from chaos_llm.analysis.divergence import divergence_any_pair, divergence_pairwise, divergence_vs_baseline
from chaos_llm.analysis.agreement import agreement_all_pairs, agreement_with_baseline
from chaos_llm.analysis.logits import aggregate_time_series, load_logit_metrics
from chaos_llm.analysis.plots import (
    apply_style, plot_dependency_curves, plot_histogram, 
    plot_time_series, plot_survival_curves
)


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


def _harmonic_mean(vals: np.ndarray) -> float:
    if not vals.size: return 0.0
    # treat inf as 0 in the sum of reciprocals
    reciprocals = 1.0 / vals.astype(float)
    reciprocals[np.isinf(vals)] = 0.0
    s = reciprocals.sum()
    return float(vals.size / s) if s > 0 else float('inf')


def _calculate_survival(values: np.ndarray, max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate survival curve: fraction of non-diverged sequences vs step."""
    # values: (num_sequences,) containing divergence indices
    # sequences that never diverge have values like np.inf or max_gen
    steps = np.arange(max_steps + 1)
    survival = np.zeros(max_steps + 1)
    for t in steps:
        # alive if divergence index > t
        alive = (values > t).sum()
        survival[t] = alive / values.size
    return steps, survival


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
                h_vals = baseline_divergence.astype(float)
                h_vals[baseline_divergence == no_div] = np.inf
                row["baseline_harmonic_mean"] = _harmonic_mean(h_vals)
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
                if filtered.size:
                    per_run_stats.append(
                        {
                            "prompt": prompt_name,
                            "sliding_window": _to_float(window),
                            "perturbation_magnitude": _to_float(mag),
                            "metric": "baseline",
                            "mean": float(filtered.mean()),
                            "median": float(np.median(filtered)),
                            "mode": _mode(filtered),
                            "std": float(filtered.std(ddof=1)) if filtered.size > 1 else 0.0,
                            "var": float(filtered.var(ddof=1)) if filtered.size > 1 else 0.0,
                            "q05": float(np.quantile(filtered, 0.05)),
                            "q25": float(np.quantile(filtered, 0.25)),
                            "q75": float(np.quantile(filtered, 0.75)),
                            "q95": float(np.quantile(filtered, 0.95)),
                        }
                    )
                    if cfg["plots"]["dependencies"].get("inverse"):
                        full_vals = baseline_divergence.astype(float)
                        no_div = cfg["divergence"]["no_divergence_value"]
                        full_vals[baseline_divergence == no_div] = np.inf
                        inv_vals = 1.0 / full_vals
                        per_run_stats.append({
                            "prompt": prompt_name,
                            "sliding_window": _to_float(window),
                            "perturbation_magnitude": _to_float(mag),
                            "metric": "baseline_inverse",
                            "mean": float(inv_vals.mean()),
                            "median": float(np.median(inv_vals)),
                            "mode": _mode(inv_vals),
                            "std": float(inv_vals.std(ddof=1)) if inv_vals.size > 1 else 0.0,
                            "var": float(inv_vals.var(ddof=1)) if inv_vals.size > 1 else 0.0,
                            "q05": float(np.quantile(inv_vals, 0.05)),
                            "q25": float(np.quantile(inv_vals, 0.25)),
                            "q75": float(np.quantile(inv_vals, 0.75)),
                            "q95": float(np.quantile(inv_vals, 0.95)),
                        })
                elif row.get("baseline_no_divergence", 0) > 0:
                    stable_val = cfg["divergence"].get("stable_divergence_value")
                    if stable_val == "auto":
                        max_gen = int(meta["generation"]["max_new_tokens"])
                        if cfg["divergence"]["index_reference"] == "absolute":
                            stable_val = float(prompt_len + max_gen)
                        else:
                            stable_val = float(max_gen)

                    if stable_val is not None:
                        per_run_stats.append({
                            "prompt": prompt_name,
                            "sliding_window": _to_float(window),
                            "perturbation_magnitude": _to_float(mag),
                            "metric": "baseline",
                            "mean": float(stable_val),
                            "median": float(stable_val),
                            "mode": float(stable_val),
                            "std": 0.0,
                            "var": 0.0,
                        })
                        if cfg["plots"]["dependencies"].get("inverse"):
                            # If stable_val is None or auto, inverse is 0. 
                            # If stable_val is a number, inverse is 1/stable_val.
                            inv_mean = 1.0 / float(stable_val) if (stable_val is not None and stable_val != 0) else 0.0
                            if stable_val == float('inf'): inv_mean = 0.0
                            per_run_stats.append({
                                "prompt": prompt_name,
                                "sliding_window": _to_float(window),
                                "perturbation_magnitude": _to_float(mag),
                                "metric": "baseline_inverse",
                                "mean": inv_mean,
                                "median": inv_mean,
                                "mode": inv_mean,
                                "std": 0.0,
                                "var": 0.0,
                            })

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
                h_vals = values.astype(float)
                h_vals[values == no_div] = np.inf
                row["pairwise_harmonic_mean"] = _harmonic_mean(h_vals)
                for q in cfg["summary"]["quantiles"]:
                    row[f"pairwise_q{int(q*100):02d}"] = float(np.quantile(filtered, q))
            else:
                row["pairwise_mean"] = ""
                row["pairwise_std"] = ""
                row["pairwise_median"] = ""
                row["pairwise_mode"] = ""

            if cfg["divergence"]["primary_metric"] == "pairwise":
                per_prompt_values.setdefault(prompt_name, []).extend(filtered.tolist())
                if filtered.size:
                    per_run_stats.append(
                        {
                            "prompt": prompt_name,
                            "sliding_window": _to_float(window),
                            "perturbation_magnitude": _to_float(mag),
                            "metric": "pairwise",
                            "mean": float(filtered.mean()),
                            "median": float(np.median(filtered)),
                            "mode": _mode(filtered),
                            "std": float(filtered.std(ddof=1)) if filtered.size > 1 else 0.0,
                            "var": float(filtered.var(ddof=1)) if filtered.size > 1 else 0.0,
                            "q05": float(np.quantile(filtered, 0.05)),
                            "q25": float(np.quantile(filtered, 0.25)),
                            "q75": float(np.quantile(filtered, 0.75)),
                            "q95": float(np.quantile(filtered, 0.95)),
                        }
                    )
                    if cfg["plots"]["dependencies"].get("inverse"):
                        # Use inf for stable runs in the inverse calculation
                        full_vals = values.astype(float)
                        full_vals[values == no_div] = np.inf
                        inv_vals = 1.0 / full_vals
                        per_run_stats.append({
                            "prompt": prompt_name,
                            "sliding_window": _to_float(window),
                            "perturbation_magnitude": _to_float(mag),
                            "metric": "pairwise_inverse",
                            "mean": float(inv_vals.mean()),
                            "median": float(np.median(inv_vals)),
                            "mode": _mode(inv_vals),
                            "std": float(inv_vals.std(ddof=1)) if inv_vals.size > 1 else 0.0,
                            "var": float(inv_vals.var(ddof=1)) if inv_vals.size > 1 else 0.0,
                            "q05": float(np.quantile(inv_vals, 0.05)),
                            "q25": float(np.quantile(inv_vals, 0.25)),
                            "q75": float(np.quantile(inv_vals, 0.75)),
                            "q95": float(np.quantile(inv_vals, 0.95)),
                        })
                elif row.get("pairwise_no_divergence", 0) > 0:
                    stable_val = cfg["divergence"].get("stable_divergence_value")
                    if stable_val == "auto":
                        max_gen = int(meta["generation"]["max_new_tokens"])
                        if cfg["divergence"]["index_reference"] == "absolute":
                            stable_val = float(prompt_len + max_gen)
                        else:
                            stable_val = float(max_gen)

                    if stable_val is not None:
                        per_run_stats.append({
                            "prompt": prompt_name,
                            "sliding_window": _to_float(window),
                            "perturbation_magnitude": _to_float(mag),
                            "metric": "pairwise",
                            "mean": float(stable_val),
                            "median": float(stable_val),
                            "mode": float(stable_val),
                            "std": 0.0,
                            "var": 0.0,
                        })
                        if cfg["plots"]["dependencies"].get("inverse"):
                            inv_mean = 1.0 / float(stable_val) if (stable_val is not None and stable_val != 0) else 0.0
                            if stable_val == float('inf'): inv_mean = 0.0
                            per_run_stats.append({
                                "prompt": prompt_name,
                                "sliding_window": _to_float(window),
                                "perturbation_magnitude": _to_float(mag),
                                "metric": "pairwise_inverse",
                                "mean": inv_mean,
                                "median": inv_mean,
                                "mode": inv_mean,
                                "std": 0.0,
                                "var": 0.0,
                            })

        if any_pair_value is not None and cfg["divergence"]["primary_metric"] == "any_pair":
            per_run_stats.append({
                "prompt": prompt_name,
                "sliding_window": _to_float(window),
                "perturbation_magnitude": _to_float(mag),
                "metric": "any_pair",
                "mean": float(any_pair_value),
                "median": float(any_pair_value),
                "mode": float(any_pair_value),
                "std": 0.0,
                "var": 0.0,
                "q05": float(any_pair_value),
                "q25": float(any_pair_value),
                "q75": float(any_pair_value),
                "q95": float(any_pair_value),
            })

        summary_rows.append(row)

        primary_vals, _ = _select_primary_values(
            cfg["divergence"]["primary_metric"],
            any_pair_value,
            baseline_divergence,
            pairwise_divergence,
        )
        if cfg["survival"]["enabled"] and cfg["survival"]["plot_individual"] and primary_vals is not None:
            # Treat no-divergence as inf for survival calc
            s_vals = primary_vals.astype(float)
            s_vals[primary_vals == cfg["divergence"]["no_divergence_value"]] = np.inf
            steps, survival = _calculate_survival(s_vals, int(meta["generation"]["max_new_tokens"]))
            title = f"Trajectory Survival - {prompt_name} (mag={mag}, window={window})"
            output_base = os.path.join(output_dir, "figures", f"survival_run_{window}_{mag}_{prompt_name}")
            output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
            plot_survival_curves(
                series={"survival": {"x": steps, "y": survival}},
                title=title,
                xlabel="token step",
                ylabel="fraction of stable sequences",
                output_paths=output_paths,
                grid=bool(cfg["plots"]["grid"]),
                color_map=str(cfg["plots"]["color_map"]),
                yscale=cfg["survival"]["yscale"]
            )

        hist_cfg = cfg["plots"].get("histograms", {})
        if cfg["plots"]["enabled"] and hist_cfg.get("enabled", True) and hist_cfg.get("per_run", True):
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

        if cfg["agreement"]["enabled"]:
            if cfg["agreement"]["baseline"]:
                steps, rates = agreement_with_baseline(
                    perturbed_ids, lengths, baseline_ids, prompt_len, cfg["agreement"]["max_steps"]
                )
                output_base = os.path.join(output_dir, "figures", f"agreement_baseline_{run_name}")
                output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                plot_time_series(
                    steps, rates, rates, None,
                    f"Agreement with Baseline - {run_name}", "step", "agreement rate",
                    output_paths, bool(cfg["plots"]["grid"]),
                    yscale=cfg["agreement"]["yscale"]
                )

            if cfg["agreement"]["all_pairs"]:
                steps, rates = agreement_all_pairs(
                    perturbed_ids, lengths, prompt_len, cfg["agreement"]["max_steps"]
                )
                output_base = os.path.join(output_dir, "figures", f"agreement_all_pairs_{run_name}")
                output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                plot_time_series(
                    steps, rates, rates, None,
                    f"Agreement All-Pairs - {run_name}", "step", "agreement rate",
                    output_paths, bool(cfg["plots"]["grid"]),
                    yscale=cfg["agreement"]["yscale"]
                )

        if cfg["logits"]["enabled"]:
            try:
                logit_metrics = load_logit_metrics(
                    run_dir, cfg["logits"]["filename"], cfg["performance"]["mmap_mode"]
                )
                for name, values in logit_metrics.items():
                    steps, mean, median, std = aggregate_time_series(
                        values, lengths, cfg["logits"]["max_steps"]
                    )
                    output_base = os.path.join(output_dir, "figures", f"logits_{name}_{run_name}")
                    output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                    plot_time_series(
                        steps, mean, median, std,
                        f"Logit {name} - {run_name}", "step", name,
                        output_paths, bool(cfg["plots"]["grid"]),
                        yscale=cfg["logits"]["yscale"]
                    )
            except FileNotFoundError:
                pass

    if not summary_rows:
        raise ValueError("No runs matched the prompt filters")

    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    _write_summary(os.path.join(output_dir, "summary.csv"), summary_rows, fieldnames)

    hist_cfg = cfg["plots"].get("histograms", {})
    if cfg["plots"]["enabled"] and hist_cfg.get("enabled", True) and hist_cfg.get("per_prompt", True):
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
        x_scales = dep_cfg.get("x_scale", ["linear"])
        y_scales = dep_cfg.get("y_scale", ["linear"])

        def build_series(
            x_key: str,
            line_key: str,
            prompt_name: Optional[str],
            metric_name: str,
            use_error: bool,
            linestyle: str,
            filter_key: Optional[str] = None,
            filter_val: Optional[Any] = None,
            metric_type: Optional[str] = None,
            is_fan: bool = False,
        ) -> Dict[str, Dict[str, Any]]:
            target_metric = metric_type or cfg["divergence"]["primary_metric"]
            series: Dict[str, Dict[str, Any]] = {}
            for row in per_run_stats:
                if row.get("metric") != target_metric:
                    continue
                if prompt_name is not None and row.get("prompt") != prompt_name:
                    continue
                if filter_key is not None and row.get(filter_key) != filter_val:
                    continue
                x_val = row.get(x_key)
                line_val = row.get(line_key)
                y_val = row.get(metric_name)
                y_err = row.get(error_bars) if use_error else None
                
                if x_val is None or line_val is None:
                    continue
                
                label = f"{line_key}={line_val} ({metric_name})"
                if is_fan:
                    series.setdefault(label, {"x": [], "y_median": [], "y_q05": [], "y_q25": [], "y_q75": [], "y_q95": []})
                    series[label]["x"].append(x_val)
                    series[label]["y_median"].append(y_val)
                    series[label]["y_q05"].append(row.get("q05", y_val))
                    series[label]["y_q25"].append(row.get("q25", y_val))
                    series[label]["y_q75"].append(row.get("q75", y_val))
                    series[label]["y_q95"].append(row.get("q95", y_val))
                else:
                    series.setdefault(label, {"x": [], "y": [], "yerr": [], "linestyle": linestyle})
                    series[label]["x"].append(x_val)
                    series[label]["y"].append(y_val)
                    if use_error:
                        series[label]["yerr"].append(y_err)

            for label, data in series.items():
                if is_fan:
                    keys = ["y_median", "y_q05", "y_q25", "y_q75", "y_q95"]
                    points = list(zip(data["x"], *[data[k] for k in keys]))
                    points.sort(key=lambda p: p[0])
                    data["x"] = [p[0] for p in points]
                    for i, k in enumerate(keys):
                        data[k] = [p[i+1] for p in points]
                else:
                    if use_error:
                        points = list(zip(data["x"], data["y"], data["yerr"]))
                    else:
                        points = list(zip(data["x"], data["y"]))

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
        magnitudes = sorted({row["perturbation_magnitude"] for row in per_run_stats if row["perturbation_magnitude"] is not None})
        windows = sorted({row["sliding_window"] for row in per_run_stats if row["sliding_window"] is not None})
        if not per_prompt:
            prompts = [None]

        for prompt_name in prompts:
            suffix = f"_{prompt_name}" if prompt_name else ""

            if "sliding_window" in x_axes:
                for mode_name, ylabel, filename_suffix in [
                    (None, "divergence index", ""),
                    (f"{cfg['divergence']['primary_metric']}_inverse", "inverse divergence index", "_inverse")
                ]:
                    if mode_name and not dep_cfg.get("inverse"):
                        continue

                    for x_scale in x_scales:
                        for y_scale in y_scales:
                            series = {}
                            for metric in metrics:
                                use_error = (metric == "mean" and error_bars in ("std", "var")) or (metric == "median" and error_bars == "fan")
                                linestyle = "-" if metric == "mean" else ("--" if metric == "median" else ":")
                                series.update(
                                    build_series(
                                        "sliding_window",
                                        "perturbation_magnitude",
                                        prompt_name,
                                        metric,
                                        use_error,
                                        linestyle,
                                        metric_type=mode_name,
                                        is_fan=(error_bars == "fan")
                                    )
                                )
                            
                            scale_suffix = f"_{x_scale}_{y_scale}"
                            title = f"{cfg['plots']['title_prefix']} (window dependence){suffix}{filename_suffix} [{x_scale}/{y_scale}]"
                            output_base = os.path.join(output_dir, "figures", f"dep_window{suffix}{filename_suffix}{scale_suffix}")
                            output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                            plot_dependency_curves(
                                series=series,
                                title=title,
                                xlabel="sliding window",
                                ylabel=ylabel,
                                output_paths=output_paths,
                                grid=bool(cfg["plots"]["grid"]),
                                color_map=str(cfg["plots"]["color_map"]),
                                xscale=x_scale,
                                yscale=y_scale,
                            )

            if "perturbation_magnitude" in x_axes:
                for mode_name, ylabel, filename_suffix in [
                    (None, "divergence index", ""),
                    (f"{cfg['divergence']['primary_metric']}_inverse", "inverse divergence index", "_inverse")
                ]:
                    if mode_name and not dep_cfg.get("inverse"):
                        continue
                    
                    for x_scale in x_scales:
                        for y_scale in y_scales:
                            series = {}
                            for metric in metrics:
                                use_error = (metric == "mean" and error_bars in ("std", "var")) or (metric == "median" and error_bars == "fan")
                                linestyle = "-" if metric == "mean" else ("--" if metric == "median" else ":")
                                series.update(
                                    build_series(
                                        "perturbation_magnitude",
                                        "sliding_window",
                                        prompt_name,
                                        metric,
                                        use_error,
                                        linestyle,
                                        metric_type=mode_name,
                                        is_fan=(error_bars == "fan")
                                    )
                                )
                            
                            scale_suffix = f"_{x_scale}_{y_scale}"
                            title = f"{cfg['plots']['title_prefix']} (magnitude dependence){suffix}{filename_suffix} [{x_scale}/{y_scale}]"
                            output_base = os.path.join(output_dir, "figures", f"dep_magnitude{suffix}{filename_suffix}{scale_suffix}")
                            output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
                            plot_dependency_curves(
                                series=series,
                                title=title,
                                xlabel="perturbation magnitude",
                                ylabel=ylabel,
                                output_paths=output_paths,
                                grid=bool(cfg["plots"]["grid"]),
                                color_map=str(cfg["plots"]["color_map"]),
                                xscale=x_scale,
                                yscale=y_scale,
                            )


    # Combined survival plots
    if cfg["survival"]["enabled"] and cfg["survival"]["plot_together"]:
        # group by (prompt, magnitude) -> series {window_label -> (steps, survival)}
        groups: Dict[str, Dict[str, Any]] = {}
        
        for run_dir in runs:
            meta = load_run_metadata(run_dir)
            if not _filter_prompts(meta, include, exclude): continue
            
            window, mag, _ = _extract_runtime(meta)
            prompt_name = meta.get("prompt", {}).get("name", "unknown")
            
            try:
                tokens = load_tokens(run_dir, cfg["performance"]["mmap_mode"])
                perturbed_ids = tokens["perturbed_ids"]
                baseline_ids = tokens["baseline_ids"]
                lengths = tokens["perturbed_lengths"]
                prompt_len = int(tokens.get("prompt_len", meta.get("runtime", {}).get("prompt_len", 0)))
                max_gen = int(meta["generation"]["max_new_tokens"])
                
                div_indices = divergence_vs_baseline(
                    perturbed_ids, lengths, baseline_ids, prompt_len, cfg["divergence"]["index_reference"]
                )
                # Treat no-divergence as inf
                div_indices = div_indices.astype(float)
                div_indices[div_indices == cfg["divergence"]["no_divergence_value"]] = np.inf
                
                steps, survival = _calculate_survival(div_indices, max_gen)
                
                group_key = f"{prompt_name}_mag_{mag}"
                groups.setdefault(group_key, {})
                groups[group_key][f"window={window}"] = {"x": steps, "y": survival}
            except FileNotFoundError:
                continue
                
        for group_key, series in groups.items():
            output_base = os.path.join(output_dir, "figures", f"survival_together_{group_key}")
            output_paths = _format_outputs(output_base, cfg["plots"]["formats"])
            plot_survival_curves(
                series=series,
                title=f"Combined Trajectory Survival (mag={mag})",
                xlabel="token step",
                ylabel="fraction of stable sequences",
                output_paths=output_paths,
                grid=bool(cfg["plots"]["grid"]),
                color_map=str(cfg["plots"]["color_map"]),
                yscale=cfg["survival"]["yscale"]
            )


if __name__ == "__main__":
    main()
