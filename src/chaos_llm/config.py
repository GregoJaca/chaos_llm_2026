import json
from copy import deepcopy
from typing import Any, Dict

import yaml
import torch


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_defaults(cfg)
    validate_config(cfg)
    return cfg


def apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    cfg.setdefault("project", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("prompts", {})
    cfg.setdefault("perturbation", {})
    cfg.setdefault("attention", {})
    cfg.setdefault("generation", {})
    cfg.setdefault("output", {})

    cfg["project"].setdefault("seed", 0)
    cfg["project"].setdefault("device", "auto")
    cfg["project"].setdefault("dtype", "float16")

    cfg["paths"].setdefault("output_dir", "outputs")

    cfg["prompts"].setdefault("format", "line")
    cfg["prompts"].setdefault("encoding", "utf-8")

    cfg["perturbation"].setdefault("apply_to_first_n_tokens", None)
    cfg["perturbation"].setdefault("subspace_mode", "random")
    cfg["perturbation"].setdefault("subspace_seed", cfg["project"]["seed"])

    cfg["attention"].setdefault("use_sliding_window", True)

    cfg["generation"].setdefault("adaptive_stop", False)
    cfg["generation"].setdefault("do_sample", False)
    cfg["generation"].setdefault("temperature", 0.0)
    cfg["generation"].setdefault("top_k", 1)
    cfg["generation"].setdefault("top_p", 1.0)
    cfg["generation"].setdefault("num_beams", 1)

    cfg["output"].setdefault("pad_token_id", -1)
    cfg["output"].setdefault("save_text", False)
    cfg["output"].setdefault("include_prompt_tokens", True)

    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    required = [
        ("paths", "model_path"),
        ("paths", "prompts_path"),
        ("perturbation", "num_conditions"),
        ("perturbation", "magnitude_list"),
        ("attention", "sliding_window_list"),
        ("generation", "max_new_tokens"),
    ]
    for section, key in required:
        if key not in cfg.get(section, {}):
            raise ValueError(f"Missing config value: {section}.{key}")

    if cfg["perturbation"]["num_conditions"] <= 0:
        raise ValueError("perturbation.num_conditions must be > 0")
    if not cfg["perturbation"]["magnitude_list"]:
        raise ValueError("perturbation.magnitude_list must be non-empty")
    if not cfg["attention"]["sliding_window_list"]:
        raise ValueError("attention.sliding_window_list must be non-empty")
    if cfg["generation"]["max_new_tokens"] <= 0:
        raise ValueError("generation.max_new_tokens must be > 0")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = mapping[dtype_str]
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def jsonable_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))
