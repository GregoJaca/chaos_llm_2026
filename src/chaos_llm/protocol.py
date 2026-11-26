"""Shared inference protocol for decoder-only LLMs (FP32 <=2.5B, bf16 above).

Rules: in-batch r=0 reference, fixed-length generation, one precision per model.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_model_path(model):
    p = os.path.expanduser(model)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
        return p
    if os.path.isdir(p):
        snaps = sorted(glob.glob(os.path.join(p, "snapshots", "*", "config.json")))
        if snaps:
            return os.path.dirname(snaps[-1])
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    cand = os.path.join(cache, "models--" + model.replace("/", "--"))
    snaps = sorted(glob.glob(os.path.join(cand, "snapshots", "*", "config.json")))
    if snaps:
        return os.path.dirname(snaps[-1])
    return p


def estimate_params(cfg):
    """Rough parameter count from config, good enough to pick a dtype."""
    H = getattr(cfg, "hidden_size", 1536)
    L = getattr(cfg, "num_hidden_layers", 28)
    F = getattr(cfg, "intermediate_size", 4 * H)
    V = getattr(cfg, "vocab_size", 150000)
    kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 12))
    attn = 4 * H * H if kv == getattr(cfg, "num_attention_heads", 12) else 2 * H * H + 2 * H * (H * kv // max(1, getattr(cfg, "num_attention_heads", 12)))
    return V * H + L * (attn + 2 * H * F) + 2 * H


def model_info(model, dtype=None):
    """Config-only description; does not load weights."""
    path = resolve_model_path(model)
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    tc = getattr(cfg, "text_config", cfg)
    H = getattr(tc, "hidden_size", getattr(cfg, "hidden_size", 1536))
    L = getattr(tc, "num_hidden_layers", getattr(cfg, "num_hidden_layers", 28))
    F = getattr(tc, "intermediate_size", 4 * H)
    V = getattr(tc, "vocab_size", getattr(cfg, "vocab_size", 150000))
    params = V * H + L * (4 * H * H + 2 * H * F) + 2 * H
    if dtype is None or dtype == "auto":
        dtype = os.environ.get("CHAOS_DTYPE", "")
    if dtype in (None, "", "auto"):
        dtype = "float32" if params * 4 <= 9.5e9 else "float16"
    return dict(path=path, params=params, dtype=dtype, hidden_size=H, num_layers=L,
                vocab_size=V, sliding_window=getattr(tc, "sliding_window", None),
                use_sliding_window=getattr(cfg, "use_sliding_window", None))


def load_model(model="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B", attn="sdpa", dtype="auto"):
    path = resolve_model_path(model)
    info = model_info(model, dtype)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    td = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[info["dtype"]]
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=td, local_files_only=True,
            attn_implementation=attn, low_cpu_mem_usage=True,
        ).cuda().eval()
    except ValueError:
        # some architectures (e.g. Mamba) have no sdpa path; fall back to eager
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=td, local_files_only=True,
            attn_implementation="eager", low_cpu_mem_usage=True,
        ).cuda().eval()
    m._chaos_info = info
    return m, tok, path


def set_window(model, window):
    """Set the sliding window if the architecture supports it; warn otherwise."""
    cfg = model.config
    if hasattr(cfg, "sliding_window"):
        cfg.sliding_window = int(window)
        if hasattr(cfg, "use_sliding_window"):
            cfg.use_sliding_window = True
        return True
    print("WARNING: this config has no sliding_window attribute; window is ignored", flush=True)
    return False


def band_mask4d(L, W, device="cuda", dtype=torch.float32):
    """Causal sliding-window mask [1,1,L,L] (0 inside the window, -inf outside)."""
    idx = torch.arange(L, device=device)
    band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - int(W))
    m = torch.zeros((L, L), device=device, dtype=dtype).masked_fill(~band, torch.finfo(dtype).min)
    return m[None, None]


def get_layers(model):
    """Return the list of decoder blocks, robust to the wrapper name."""
    for path in ("model.layers", "backbone.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("cannot find decoder layers on this model")


def prompt_ids(tokenizer, text):
    return tokenizer(text, return_tensors="pt").input_ids.cuda()


def build_simplex(k):
    if k < 2:
        raise ValueError("num_directions must be >= 2 (k=1 gives a zero perturbation)")
    eye = np.eye(k, dtype=np.float32)
    v = eye - np.ones((k, k), dtype=np.float32) / k
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def subspace_indices(total_dim, k, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(total_dim, size=k, replace=False).astype(np.int64)


def make_directions(num_directions, prompt_len, dim, seed=69):
    simplex = build_simplex(num_directions)
    sub = subspace_indices(prompt_len * dim, num_directions, seed)
    return simplex, sub


@torch.inference_mode()
def generate(model, embeds, max_new_tokens, batch_meta=None):
    B, P, _ = embeds.shape
    mask = torch.ones((B, P), dtype=torch.long, device=embeds.device)
    out = model.generate(
        inputs_embeds=embeds, attention_mask=mask, max_new_tokens=int(max_new_tokens),
        do_sample=False, eos_token_id=None, pad_token_id=0,
    )
    return out


@torch.inference_mode()
def forward(model, embeds, output_hidden_states=False):
    B, L, _ = embeds.shape
    mask = torch.ones((B, L), dtype=torch.long, device=embeds.device)
    return model(inputs_embeds=embeds, attention_mask=mask,
                 output_hidden_states=output_hidden_states, use_cache=False)


def full_embeddings(model, prompt_ids_t, gen_ids, prompt_delta=None):
    """Embed prompt+generated tokens; optionally add a [P,D] or [B,P,D] prompt delta."""
    ids = torch.cat([prompt_ids_t, gen_ids], dim=1)
    E = model.get_input_embeddings()(ids)
    if prompt_delta is not None:
        d = prompt_delta
        if d.dim() == 2:
            d = d.unsqueeze(0)
        E[:, :d.shape[1]] += d
    return E


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

