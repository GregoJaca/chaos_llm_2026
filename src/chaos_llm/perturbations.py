from typing import Iterable, Tuple

import numpy as np
import torch


def build_simplex(num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be >= 2 for a simplex")
    eye = np.eye(num_points, dtype=np.float32)
    ones = np.ones((num_points, num_points), dtype=np.float32) / num_points
    vertices = eye - ones
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    return vertices


def select_subspace_indices(
    total_dim: int,
    num_points: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_points > total_dim:
        raise ValueError(
            f"num_conditions ({num_points}) must be <= subspace_dim ({total_dim})"
        )
    if mode == "first":
        return np.arange(num_points, dtype=np.int64)
    if mode == "random":
        return rng.choice(total_dim, size=num_points, replace=False).astype(np.int64)
    raise ValueError(f"Unknown subspace_mode: {mode}")


def iter_simplex_perturbations(
    base_embeds: torch.Tensor,
    simplex: np.ndarray,
    magnitude: float,
    subspace_indices: np.ndarray,
) -> Iterable[torch.Tensor]:
    device = base_embeds.device
    dtype = base_embeds.dtype

    flat_size = base_embeds.numel()
    index_tensor = torch.as_tensor(subspace_indices, device=device, dtype=torch.long)

    for i in range(simplex.shape[0]):
        row = torch.as_tensor(simplex[i], device=device, dtype=dtype) * magnitude
        delta = torch.zeros((flat_size,), device=device, dtype=dtype)
        delta.index_copy_(0, index_tensor, row)
        yield delta.view_as(base_embeds)
