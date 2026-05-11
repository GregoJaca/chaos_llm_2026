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


def _pairwise_divergence_index(
    seq_a: np.ndarray,
    len_a: int,
    seq_b: np.ndarray,
    len_b: int,
    prompt_len: int,
    index_reference: str,
) -> int:
    min_len = min(len_a, len_b)
    if min_len > prompt_len:
        slice_a = seq_a[prompt_len:min_len]
        slice_b = seq_b[prompt_len:min_len]
        mismatch = np.nonzero(slice_a != slice_b)[0]
        if mismatch.size:
            pos = prompt_len + int(mismatch[0])
            return _to_index(pos, prompt_len, index_reference)
    if len_a != len_b:
        return _to_index(min_len, prompt_len, index_reference)
    return -1


def divergence_pairwise(
    perturbed_ids: np.ndarray,
    lengths: np.ndarray,
    prompt_len: int,
    index_reference: str,
    max_pairs: Optional[int] = None,
) -> np.ndarray:
    n = perturbed_ids.shape[0]
    results = []
    count = 0
    for i in range(n - 1):
        seq_a = perturbed_ids[i]
        len_a = int(lengths[i])
        for j in range(i + 1, n):
            seq_b = perturbed_ids[j]
            len_b = int(lengths[j])
            div_idx = _pairwise_divergence_index(
                seq_a=seq_a,
                len_a=len_a,
                seq_b=seq_b,
                len_b=len_b,
                prompt_len=prompt_len,
                index_reference=index_reference,
            )
            results.append(div_idx)
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return np.array(results, dtype=np.int32)
    return np.array(results, dtype=np.int32)
