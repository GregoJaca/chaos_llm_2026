import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chaos_llm.analysis.config import load_analysis_config
from chaos_llm.analysis.data import discover_runs, load_run_metadata, load_tokens
from chaos_llm.analysis.divergence import divergence_any_pair, divergence_pairwise, divergence_vs_baseline
from chaos_llm.analysis.plots import apply_style, plot_histogram


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
    root, _ = os.path.splitext(base_path)
    return [f"{root}.{ext}" for ext in formats]


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

    for run_dir in runs:
        meta = load_run_metadata(run_dir)
        if not _filter_prompts(meta, include, exclude):
            continue

        tokens = load_tokens(run_dir, mmap_mode=cfg["performance"]["mmap_mode"])
        baseline_ids = tokens["baseline_ids"]
        perturbed_ids = tokens["perturbed_ids"]
        lengths = tokens["perturbed_lengths"]
        prompt_len = int(tokens.get("prompt_len", meta.get("runtime", {}).get("prompt_len", 0)))

        any_pair_value = None
        if cfg["divergence"]["any_pair"]:
            any_pair_value = divergence_any_pair(
                perturbed_ids=perturbed_ids,
                lengths=lengths,
                prompt_len=prompt_len,
                include_baseline=cfg["divergence"]["include_baseline_in_any_pair"],
                baseline_ids=baseline_ids,
                index_reference=cfg["divergence"]["index_reference"],
            )

        baseline_divergence = None
        if cfg["divergence"]["baseline_per_sequence"]:
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
                for q in cfg["summary"]["quantiles"]:
                    row[f"baseline_q{int(q*100):02d}"] = float(np.quantile(filtered, q))
            else:
                row["baseline_mean"] = ""
                row["baseline_std"] = ""
                row["baseline_median"] = ""

            if cfg["divergence"]["primary_metric"] == "baseline_per_sequence":
                per_prompt_values.setdefault(prompt_name, []).extend(filtered.tolist())

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
                for q in cfg["summary"]["quantiles"]:
                    row[f"pairwise_q{int(q*100):02d}"] = float(np.quantile(filtered, q))
            else:
                row["pairwise_mean"] = ""
                row["pairwise_std"] = ""
                row["pairwise_median"] = ""

            if cfg["divergence"]["primary_metric"] == "pairwise":
                per_prompt_values.setdefault(prompt_name, []).extend(filtered.tolist())

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


if __name__ == "__main__":
    main()
