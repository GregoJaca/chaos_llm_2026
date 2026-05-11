from typing import Optional, Tuple

import numpy as np


def _to_index(pos: int, prompt_len: int, index_reference: str) -> int:
    if index_reference == "absolute":
        return pos
    return pos - prompt_len


def divergence_any_pair(
    perturbed_ids: np.ndarray,
    lengths: np.ndarray,
    prompt_len: int,
    include_baseline: bool,
    baseline_ids: Optional[np.ndarray],
    index_reference: str,
) -> int:
    max_len = int(lengths.max()) if len(lengths) else 0
    if include_baseline and baseline_ids is not None:
        max_len = max(max_len, int(len(baseline_ids)))

    for pos in range(prompt_len, max_len):
        active = lengths > pos
        tokens = perturbed_ids[active, pos] if active.any() else np.array([], dtype=np.int32)
        if include_baseline and baseline_ids is not None and len(baseline_ids) > pos:
            tokens = np.concatenate([tokens, baseline_ids[pos:pos + 1]])
        if tokens.size < 2:
            continue
        if np.any(tokens != tokens[0]):
            return _to_index(pos, prompt_len, index_reference)
    return -1


def divergence_vs_baseline(
    perturbed_ids: np.ndarray,
    lengths: np.ndarray,
    baseline_ids: np.ndarray,
    prompt_len: int,
    index_reference: str,
) -> np.ndarray:
    out = np.full((perturbed_ids.shape[0],), -1, dtype=np.int32)
    base_len = len(baseline_ids)

    for i in range(perturbed_ids.shape[0]):
        seq_len = int(lengths[i])
        max_pos = min(seq_len, base_len)
        if max_pos <= prompt_len:
            continue
        slice_ids = perturbed_ids[i, prompt_len:max_pos]
        base_slice = baseline_ids[prompt_len:max_pos]
        mismatch = np.nonzero(slice_ids != base_slice)[0]
        if mismatch.size:
            pos = prompt_len + int(mismatch[0])
            out[i] = _to_index(pos, prompt_len, index_reference)
    return out
