from typing import Optional, Tuple

import numpy as np


def agreement_with_baseline(
    perturbed_ids: np.ndarray,
    lengths: np.ndarray,
    baseline_ids: np.ndarray,
    prompt_len: int,
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    max_len = min(int(lengths.max()) if len(lengths) else 0, int(len(baseline_ids)))
    if max_steps is not None:
        max_len = min(max_len, prompt_len + int(max_steps))

    steps = np.arange(prompt_len, max_len, dtype=np.int32)
    rates = np.full((len(steps),), np.nan, dtype=np.float32)

    for idx, pos in enumerate(steps):
        active = lengths > pos
        if not active.any():
            continue
        tokens = perturbed_ids[active, pos]
        rates[idx] = float(np.mean(tokens == baseline_ids[pos]))

    return steps - prompt_len, rates


def agreement_all_pairs(
    perturbed_ids: np.ndarray,
    lengths: np.ndarray,
    prompt_len: int,
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    max_len = int(lengths.max()) if len(lengths) else 0
    if max_steps is not None:
        max_len = min(max_len, prompt_len + int(max_steps))

    steps = np.arange(prompt_len, max_len, dtype=np.int32)
    rates = np.full((len(steps),), np.nan, dtype=np.float32)

    for idx, pos in enumerate(steps):
        active = lengths > pos
        tokens = perturbed_ids[active, pos]
        n = tokens.shape[0]
        if n < 2:
            continue
        _, counts = np.unique(tokens, return_counts=True)
        agree_pairs = np.sum(counts * (counts - 1) // 2)
        total_pairs = n * (n - 1) // 2
        rates[idx] = float(agree_pairs / total_pairs) if total_pairs > 0 else np.nan

    return steps - prompt_len, rates
