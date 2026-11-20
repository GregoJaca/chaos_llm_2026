"""High-throughput FP32 inference engine for large models on the DGX Spark (GB10).

The GB10 is bandwidth-bound (~273 GB/s unified memory): every forward pass sweeps
the whole weight matrix, so the way to be fast is to (a) batch many independent
rows into one forward, and (b) prefer full-sequence teacher-forced forwards
(1 weight sweep for P+T positions) over autoregressive generation
(T weight sweeps). This module implements both.

Numerical rules carried over from the T4 protocol (docs/DECISIONS.md):
- FP32 only, one precision per model, never mixed.
- In-batch r=0 reference rows; two identical r=0 rows must be bit-identical.
- No TF32 (mantissa truncation would fake margins); assert it is off.

Memory rules (shared 128 GB unified memory, colleagues hold ~49 GB):
- check_mem() reads /proc/meminfo and refuses to run below a floor.
- Hooks only ever hold the selected layers; everything else is released.
- torch.cuda.empty_cache() between batches.
"""
import glob
import json
import os

import numpy as np
import torch


# --------------------------------------------------------------------- memory

def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return -1.0


def check_mem(min_gb, label=""):
    avail = mem_available_gb()
    if avail < min_gb:
        raise RuntimeError(
            f"memory guard: MemAvailable {avail:.1f} GB < {min_gb:.1f} GB required ({label}); "
            "refusing to run (shared machine, colleagues' servers must not be disturbed)")
    return avail


def configure_torch():
    """Deterministic fp32 matmul settings; call before loading the model."""
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 must stay off (margin science)"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    inc = os.path.expanduser("~/LLM/pydev/extracted/usr/include")
    pyinc = os.path.join(inc, "python3.12")
    if os.path.exists(os.path.join(pyinc, "Python.h")):
        # Python.h itself lives in python3.12/, but its platform pyconfig.h is
        # included as <aarch64-linux-gnu/python3.12/pyconfig.h>, i.e. relative
        # to the base include dir; both must be on the search path.
        os.environ["C_INCLUDE_PATH"] = pyinc + os.pathsep + inc
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------- model

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


def load_big(model, dtype="float32", attn="sdpa", load_guard_gb=8.0):
    """Load a large model in one fixed precision. Returns (model, tokenizer, path, info).

    Refuses to load when the unified-memory pool cannot hold the weights plus
    `load_guard_gb` of headroom (shared machine; colleagues' servers must not
    be disturbed).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    configure_torch()
    path = resolve_model_path(model)
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    tc = getattr(cfg, "text_config", cfg)
    H = getattr(tc, "hidden_size", getattr(cfg, "hidden_size", None))
    L = getattr(tc, "num_hidden_layers", getattr(cfg, "num_hidden_layers", None))
    V = getattr(tc, "vocab_size", getattr(cfg, "vocab_size", None))
    F = getattr(tc, "intermediate_size", 4 * (H or 2560))
    kv = getattr(tc, "num_key_value_heads", getattr(tc, "num_attention_heads", 1) or 1)
    nh = getattr(tc, "num_attention_heads", kv)
    hd = getattr(tc, "head_dim", (H or 2560) // (nh or 1))
    # q/k/v/o projections (GQA) + SwiGLU MLP (gate, up, down = 3*H*F)
    attn_params = 2 * H * (nh * hd) + 2 * H * (kv * hd)
    emb = V * H * (1 if getattr(tc, "tie_word_embeddings", False) else 2)
    params = emb + L * (attn_params + 3 * H * F) + 2 * H
    bytes_per = {"float32": 4, "bfloat16": 2, "float16": 2}[dtype]
    weights_gb = params * bytes_per / 1e9
    if mem_available_gb() < weights_gb + load_guard_gb:
        raise RuntimeError(
            f"memory guard: need ~{weights_gb:.0f} GB weights + {load_guard_gb} GB headroom, "
            f"only {mem_available_gb():.1f} GB available; refusing to load")
    info = dict(path=path, dtype=dtype, hidden_size=H, num_layers=L, vocab_size=V,
                params=params, weights_gb=round(weights_gb, 1),
                sliding_window=getattr(tc, "sliding_window", None),
                arch=cfg.architectures[0] if getattr(cfg, "architectures", None) else "?")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    td = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    # GB10 unified memory: CPU and GPU share one physical pool, so a CPU-side
    # materialisation followed by .cuda() would transiently need 2x the weights
    # (118 GB for a 14B fp32) and OOM the shared machine. device_map streams
    # shards tensor-by-tensor straight onto the GPU instead.
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=td, local_files_only=True, attn_implementation=attn,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    m.eval()
    m._chaos_info = info
    return m, tok, path, info


def set_window(model, window):
    """Same semantics as protocol.set_window (config-level sliding window)."""
    cfg = model.config
    if hasattr(cfg, "sliding_window"):
        cfg.sliding_window = int(window)
        if hasattr(cfg, "use_sliding_window"):
            cfg.use_sliding_window = True
        return True
    print("WARNING: config has no sliding_window; window ignored", flush=True)
    return False


def base_model(model):
    """The transformer trunk (layers + final norm), robust to wrapper names."""
    for path in ("model", "transformer"):
        if hasattr(model, path) and hasattr(getattr(model, path), "layers"):
            return getattr(model, path)
    raise RuntimeError("cannot find the decoder trunk on this model")


def final_norm(model):
    b = base_model(model)
    for name in ("norm", "final_layernorm", "final_norm"):
        if hasattr(b, name):
            return getattr(b, name)
    raise RuntimeError("cannot find the final norm")


# --------------------------------------------------------------------- hooks

class Capture:
    """Forward hooks that keep only the selected states (references, not copies).

    Layer index follows the transformers `output_hidden_states` convention:
    state 0 = embedding output, state l (>=1) = output of block l-1, plus the
    final-norm output under key -1 (the lm_head input).
    """

    def __init__(self, model, layers):
        self.store = {}
        self.handles = []
        b = base_model(model)
        for l in layers:
            mod = b.embed_tokens if l == 0 else b.layers[l - 1]
            self.handles.append(mod.register_forward_hook(self._wrap(l)))
        self.handles.append(final_norm(model).register_forward_hook(self._wrap(-1)))

    def _wrap(self, key):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.store[key] = t
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.store.clear()


# --------------------------------------------------------------------- directions

def build_direction_blocks(num_directions, prompt_len, dim, block=20, seed=69):
    """num_directions unit directions as independent simplex blocks.

    Each block of `block` directions lives on its own random set of `block`
    coordinates of the prompt embedding (same geometry as the original 20-dir
    protocol); different blocks use disjoint coordinates, so directions from
    different blocks are orthogonal. Returns (simplex [K, S], subspace [S])
    such that delta = simplex[di] * r scattered into subspace reproduces the
    perturbation exactly (same convention as a1.npz).
    """
    nblocks = int(np.ceil(num_directions / block))
    S = nblocks * block
    rng = np.random.default_rng(seed)
    coords = rng.choice(prompt_len * dim, size=S, replace=False).astype(np.int64)
    eye = np.eye(block, dtype=np.float32)
    vert = eye - np.ones((block, block), dtype=np.float32) / block
    vert = vert / np.linalg.norm(vert, axis=1, keepdims=True)
    simplex = np.zeros((num_directions, S), dtype=np.float32)
    for b in range(nblocks):
        k0, k1 = b * block, min((b + 1) * block, num_directions)
        simplex[k0:k1, b * block:(b + 1) * block] = vert[:k1 - k0]
    return simplex, coords


# --------------------------------------------------------------------- forward

@torch.inference_mode()
def baseline_pass(model, Efull, P, T, layers, capture):
    """Single-row baseline forward. Returns margins, top1, top2 (numpy) and
    h0: dict layer -> GPU tensor [T, D] of the baseline states at generated
    positions, plus 'final' -> lm_head input states."""
    L = Efull.shape[1]
    mask = torch.ones((1, L), dtype=torch.long, device="cuda")
    model(inputs_embeds=Efull, attention_mask=mask, use_cache=False)
    hf = capture.store[-1][0, P - 1:P - 1 + T]  # logits input at generated steps
    z = model.lm_head(hf)  # [T, V]
    top2 = torch.topk(z, k=2, dim=-1)
    margins = (top2.values[:, 0] - top2.values[:, 1]).detach().float().cpu().numpy()
    h0 = {}
    for l in layers:
        # state 0 = the embeddings themselves (embed_tokens is skipped with inputs_embeds)
        h0[l] = (Efull[0, P:P + T] if l == 0 else capture.store[l][0, P:P + T]).detach().clone()
    h0["final"] = hf.detach().clone()
    return margins, top2.indices[:, 0].cpu().numpy(), top2.indices[:, 1].cpu().numpy(), h0


@torch.inference_mode()
def teacher_batch(model, E, P, T, layers, h0, wdiff, capture, vector_rows=()):
    """One batched full-sequence forward of B perturbed rows.

    Returns dict of numpy arrays:
      dh_norm [B, nl, T], cos [B, nl, T], dm [B, T],
      dh_final_rows: {row_idx: [T, D] npz-able np.float32} for rows in vector_rows.
    Never materialises [B, T, V] logits: dm uses the lm_head row difference
    wdiff = W[top1] - W[top2] of shape [T, D], so dm = (h' - h0) . wdiff.
    """
    B, L, _ = E.shape
    mask = torch.ones((B, L), dtype=torch.long, device="cuda")
    model(inputs_embeds=E, attention_mask=mask, use_cache=False)
    dhf = capture.store[-1][:, P - 1:P - 1 + T] - h0["final"][None]  # [B, T, D]
    dm = torch.einsum("btd,td->bt", dhf, wdiff).detach().float().cpu().numpy()
    out = dict(dm=dm)
    dh_norm = np.zeros((B, len(layers), T), dtype=np.float32)
    cos = np.zeros((B, len(layers), T), dtype=np.float32)
    for li, l in enumerate(layers):
        h = E[:, P:P + T, :] if l == 0 else capture.store[l][:, P:P + T, :]  # [B, T, D]
        base = h0[l][None]
        diff = h - base
        dh_norm[:, li] = diff.norm(dim=-1).detach().float().cpu().numpy()
        cos[:, li] = torch.nn.functional.cosine_similarity(
            h, base.expand_as(h), dim=-1).detach().float().cpu().numpy()
        del h, diff
    out["dh_norm"] = dh_norm
    out["cos"] = cos
    if vector_rows:
        vr = {}
        for j in vector_rows:
            vr[int(j)] = dhf[j].detach().float().cpu().numpy()
        out["dh_final_rows"] = vr
    capture.store.clear()
    return out


def wdiff_tensor(model, top1, top2):
    """[T, D] rows W[top1_t] - W[top2_t]; dm = dh . wdiff exactly."""
    W = model.lm_head.weight
    return (W[torch.as_tensor(top1, device="cuda")] -
            W[torch.as_tensor(top2, device="cuda")]).detach().clone()


@torch.inference_mode()
def free_generate(model, E, max_new_tokens):
    """Batched greedy generation from embedding inputs. Returns [B, T] ids."""
    B, P, _ = E.shape
    mask = torch.ones((B, P), dtype=torch.long, device="cuda")
    out = model.generate(
        inputs_embeds=E, attention_mask=mask, max_new_tokens=int(max_new_tokens),
        do_sample=False, eos_token_id=None, pad_token_id=0)
    return out


def perturb_rows(Efull, P, D, simplex, subspace, row_indices, r):
    """Clone Efull to [B, L, D] and add r * simplex[di] scattered on subspace."""
    B = len(row_indices)
    E = Efull.expand(B, -1, -1).clone()
    sub_t = torch.as_tensor(subspace, device="cuda", dtype=torch.long)
    for j, di in enumerate(row_indices):
        delta = torch.zeros(P * D, device="cuda", dtype=torch.float32)
        delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device="cuda") * float(r))
        E[j, :P] += delta.view(P, D)
    return E


def full_embeddings(model, prompt_ids_t, gen_ids):
    ids = torch.cat([prompt_ids_t, gen_ids], dim=1)
    return model.get_input_embeddings()(ids)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
