from copy import deepcopy
from typing import Any, Dict

import yaml


def load_analysis_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_defaults(cfg)
    validate_config(cfg)
    return cfg


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    cfg.setdefault("paths", {})
    cfg.setdefault("prompt_filter", {})
    cfg.setdefault("divergence", {})
    cfg.setdefault("summary", {})
    cfg.setdefault("plots", {})
    cfg.setdefault("performance", {})

    cfg["paths"].setdefault("input_dir", "results")
    cfg["paths"].setdefault("output_dir", "analysis_outputs")
    cfg["paths"].setdefault("run_list", None)

    cfg["prompt_filter"].setdefault("include", None)
    cfg["prompt_filter"].setdefault("exclude", None)

    cfg["divergence"].setdefault("any_pair", True)
    cfg["divergence"].setdefault("baseline_per_sequence", True)
    cfg["divergence"].setdefault("pairwise", False)
    cfg["divergence"].setdefault("pairwise_max_pairs", None)
    cfg["divergence"].setdefault("exclude_baseline", False)
    cfg["divergence"].setdefault("include_baseline_in_any_pair", False)
    cfg["divergence"].setdefault("index_reference", "generated")
    cfg["divergence"].setdefault("no_divergence_value", -1)
    cfg["divergence"].setdefault("exclude_no_divergence_from_plots", True)
    cfg["divergence"].setdefault("primary_metric", "baseline_per_sequence")
    cfg["divergence"].setdefault("stable_divergence_value", "auto")
    cfg["divergence"].setdefault("exclude_prompt", True)

    cfg["summary"].setdefault("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])

    cfg["plots"].setdefault("enabled", True)
    cfg["plots"].setdefault("formats", ["png"])
    cfg["plots"].setdefault("dpi", 300)
    cfg["plots"].setdefault("bins", 60)
    cfg["plots"].setdefault("histograms", {})
    cfg["plots"]["histograms"].setdefault("enabled", True)
    cfg["plots"]["histograms"].setdefault("per_run", True)
    cfg["plots"]["histograms"].setdefault("per_prompt", True)
    cfg["plots"].setdefault("title_prefix", "Divergence")
    cfg["plots"].setdefault("grid", False)
    cfg["plots"].setdefault("color_map", "tab10")
    cfg["plots"].setdefault("xlim", None)
    cfg["plots"].setdefault("ylim", None)
    cfg["plots"].setdefault("label_no_divergence", "no divergence")
    cfg["plots"].setdefault("dependencies", {})
    cfg["plots"]["dependencies"].setdefault("enabled", True)
    cfg["plots"]["dependencies"].setdefault("metrics", ["mean", "median", "mode"])
    cfg["plots"]["dependencies"].setdefault("error_bars", "std")
    cfg["plots"]["dependencies"].setdefault("fan_quantiles", [0.05, 0.25, 0.75, 0.95])
    cfg["plots"]["dependencies"].setdefault("per_prompt", True)
    cfg["plots"]["dependencies"].setdefault("x_axis", ["sliding_window", "perturbation_magnitude"])
    cfg["plots"]["dependencies"].setdefault("inverse", False)
    cfg["plots"]["dependencies"].setdefault("x_scale", ["linear"])
    cfg["plots"]["dependencies"].setdefault("y_scale", ["linear"])
    cfg["plots"]["dependencies"].setdefault("plot_separate_magnitude", False)

    cfg.setdefault("survival", {})
    cfg["survival"].setdefault("enabled", False)
    cfg["survival"].setdefault("plot_individual", True)
    cfg["survival"].setdefault("plot_together", True)
    cfg["survival"].setdefault("yscale", "linear")
    cfg["survival"].setdefault("xscale", "linear")

    cfg["performance"].setdefault("mmap_mode", "r")

    cfg.setdefault("agreement", {})
    cfg["agreement"].setdefault("enabled", False)
    cfg["agreement"].setdefault("baseline", True)
    cfg["agreement"].setdefault("all_pairs", True)
    cfg["agreement"].setdefault("max_steps", None)
    cfg["agreement"].setdefault("yscale", "linear")
    cfg["agreement"].setdefault("xscale", "linear")

    cfg.setdefault("logits", {})
    cfg["logits"].setdefault("enabled", False)
    cfg["logits"].setdefault("filename", "logits_metrics.npz")
    cfg["logits"].setdefault("max_steps", None)
    cfg["logits"].setdefault("yscale", "linear")
    cfg["logits"].setdefault("xscale", "linear")

    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    primary = cfg["divergence"]["primary_metric"]
    if primary not in ("any_pair", "baseline_per_sequence", "pairwise"):
        raise ValueError("divergence.primary_metric must be any_pair, baseline_per_sequence, or pairwise")

    index_ref = cfg["divergence"]["index_reference"]
    if index_ref not in ("generated", "absolute"):
        raise ValueError("divergence.index_reference must be generated or absolute")
