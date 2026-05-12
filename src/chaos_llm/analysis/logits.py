import os
from typing import Dict, Optional, Tuple

import numpy as np


def load_logit_metrics(run_dir: str, filename: str, mmap_mode: Optional[str]) -> Dict[str, np.ndarray]:
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {filename} in {run_dir}")
    with np.load(path, mmap_mode=mmap_mode, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def aggregate_time_series(
    values: np.ndarray,
    lengths: Optional[np.ndarray],
    max_steps: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lengths is None:
        lengths = np.full((values.shape[0],), values.shape[1], dtype=np.int32)

    max_len = int(lengths.max()) if len(lengths) else 0
    if max_steps is not None:
        max_len = min(max_len, int(max_steps))

    mean = np.full((max_len,), np.nan, dtype=np.float32)
    median = np.full((max_len,), np.nan, dtype=np.float32)
    std = np.full((max_len,), np.nan, dtype=np.float32)

    for t in range(max_len):
        active = lengths > t
        if not active.any():
            continue
        vals = values[active, t]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        mean[t] = float(np.mean(vals))
        median[t] = float(np.median(vals))
        std[t] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

    steps = np.arange(max_len, dtype=np.int32)
    return steps, mean, median, std
