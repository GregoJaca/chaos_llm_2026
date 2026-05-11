import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np


def _format_float(value: float) -> str:
    text = f"{value:.8f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out) or "prompt"


def make_run_dir(base_dir: str, sliding_window: int, magnitude: float, prompt_name: str) -> str:
    prompt_name = sanitize_name(prompt_name)
    mag_str = _format_float(magnitude)
    run_name = f"run_{sliding_window}_{mag_str}_{prompt_name}"
    path = os.path.join(base_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_config_snapshot(path: str, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def save_tokens_npz(
    path: str,
    baseline_ids: np.ndarray,
    perturbed_ids: List[np.ndarray],
    divergence_index: List[int],
    pad_token_id: int,
    prompt_len: int,
) -> None:
    lengths = np.array([len(seq) for seq in perturbed_ids], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0

    padded = np.full((len(perturbed_ids), max_len), pad_token_id, dtype=np.int32)
    for i, seq in enumerate(perturbed_ids):
        padded[i, : len(seq)] = seq

    np.savez_compressed(
        path,
        baseline_ids=baseline_ids.astype(np.int32),
        perturbed_ids=padded,
        perturbed_lengths=lengths,
        divergence_index=np.array(divergence_index, dtype=np.int32),
        prompt_len=np.array(prompt_len, dtype=np.int32),
    )
