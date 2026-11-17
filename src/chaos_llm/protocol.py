"""Shared FP32 inference protocol for DeepSeek-1.5B.

Rules: in-batch r=0 reference, fixed-length generation, no mixed precision.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_model(model="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B", attn="sdpa"):
    path = resolve_model_path(model)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation=attn, low_cpu_mem_usage=True,
    ).cuda().eval()
    return model, tok, path


def set_window(model, window):
    model.config.sliding_window = int(window)
    model.config.use_sliding_window = True


def prompt_ids(tokenizer, text):
    return tokenizer(text, return_tensors="pt").input_ids.cuda()


def build_simplex(k):
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
