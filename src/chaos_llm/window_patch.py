"""Monkeypatch a hard sliding-window mask into any HF decoder (Qwen2/3, Phi3)."""
import types
import torch


def install_sliding_window(model, window):
    inner = None
    for name in ("model", "transformer"):
        cand = getattr(model, name, None)
        if cand is not None and hasattr(cand, "_update_causal_mask"):
            inner = cand
            break
    if inner is None:
        raise RuntimeError("no _update_causal_mask found; cannot install sliding window")
    W = int(window)

    def patched(self, attention_mask, input_tensor, cache_position, past_key_values=None,
                output_attentions=False):
        L = input_tensor.shape[1]
        device = input_tensor.device
        dtype = input_tensor.dtype
        minv = torch.finfo(dtype).min
        idx = torch.arange(L, device=device)
        band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - W)
        m = torch.full((L, L), minv, device=device, dtype=dtype).masked_fill(band, 0.0)
        return m[None, None]

    inner._update_causal_mask = types.MethodType(patched, inner)
    return True
