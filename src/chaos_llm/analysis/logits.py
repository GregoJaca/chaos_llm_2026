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
    # 1. Ensure values is a 2D matrix [N, T]
    if values.dtype == object or values.ndim == 1:
        if values.size == 0:
            return np.array([], dtype=np.int32), np.array([]), np.array([]), np.array([])
        
        # If it's an object array of arrays, or just a 1D array we treat as one sequence
        if values.dtype == object:
            N = len(values)
            T_max = max((len(v) if v is not None and hasattr(v, "__len__") else 0) for v in values)
            new_values = np.full((N, T_max), np.nan, dtype=np.float32)
            for i, v in enumerate(values):
                if v is not None and hasattr(v, "__len__"):
                    new_values[i, :len(v)] = v
            values = new_values
        else:
            values = values.reshape(1, -1)

    N_runs, T_logits = values.shape

    # 2. Align lengths with T_logits
    if lengths is not None:
        # lengths might have a different batch size than values (e.g. if values is just one baseline)
        # If lengths has 250 elements but values has 1, we should broadcast values
        if N_runs == 1 and len(lengths) > 1:
            values = np.repeat(values, len(lengths), axis=0)
            N_runs = len(lengths)
        
        # Ensure we only use as many lengths as we have rows in values
        lengths_subset = lengths[:N_runs]
        gen_lengths = np.clip(lengths_subset, 0, T_logits)
    else:
        gen_lengths = np.full((N_runs,), T_logits, dtype=np.int32)

    max_len = T_logits
    if max_steps is not None:
        max_len = min(max_len, int(max_steps))

    mean = np.full((max_len,), np.nan, dtype=np.float32)
    median = np.full((max_len,), np.nan, dtype=np.float32)
    std = np.full((max_len,), np.nan, dtype=np.float32)

    for t in range(max_len):
        # Sequence is active at time t if its (relative) length > t
        active = gen_lengths > t
        if not active.any():
            continue
        
        # Extract values for active sequences at time t
        vals = values[active, t]
        # Remove any NaNs (padding or actual missing data)
        vals = vals[~np.isnan(vals)]
        
        if vals.size == 0:
            continue
            
        mean[t] = float(np.mean(vals))
        median[t] = float(np.median(vals))
        std[t] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

    steps = np.arange(max_len, dtype=np.int32)
    return steps, mean, median, std
