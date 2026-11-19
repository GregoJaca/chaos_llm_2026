"""protocol_general.py -- model-agnostic replacement for chaos_llm/protocol.py.

Backward compatible: keeps protocol.py's public functions with the same names and
signatures (resolve_model_path, load_model, set_window, prompt_ids, build_simplex,
subspace_indices, make_directions, generate, forward, full_embeddings, save_json),
and adds:

  * automatic dtype: fp32 if params <= 2.5B else bf16 (override with dtype=...)
  * model_layers(model)   -- robust decoder-block discovery (works for Qwen2/Qwen3/
                             Phi3/Qwen3_5, nested multimodal wrappers included)
  * model_info(model)     -- hidden_size / num_layers / vocab_size / dtype / ...
  * set_window(...)       -- sets the window through the config with per-module
                             propagation, and WARNS instead of silently no-op'ing
  * window_is_effective() -- empirical end-to-end check that the window clips attention
  * hard_window_mask(...) -- dtype-aware 4D window mask (bf16 safe)
  * to_np()               -- tensor -> float32 numpy (bf16 tensors crash .numpy())

Protocol decisions (documented, identical for every model):
  * raw completion: no chat template, tokenizer called with add_special_tokens=False
  * greedy generation with eos_token_id=None (never stops early), fixed max_new_tokens
  * everything under torch.inference_mode()

Verified against transformers 4.49.0 / 4.51.3 modeling sources for Qwen2, Qwen3,
Phi3, and the Qwen3.5 (transformers 5.x) docs. Anything that cannot be verified
statically is checked at runtime with a clear error message.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:  # inside the chaos_llm package
    from .model_plan import (
        FP32_MAX_PARAMS, causal_lm_class_name, classify_attention, default_layer_indices,
        is_local_model_dir, layer_path_candidates, parse_dtype, pick_dtype,
        plan_from_config, read_safetensors_param_counts, resolve_model_path,
        resolve_text_config, text_param_estimate, _get, _has,
    )
    _MODEL_PLAN_SOURCE = "chaos_llm.model_plan"
except ImportError:  # files dropped side by side / sys.path
    from model_plan import (  # type: ignore
        FP32_MAX_PARAMS, causal_lm_class_name, classify_attention, default_layer_indices,
        is_local_model_dir, layer_path_candidates, parse_dtype, pick_dtype,
        plan_from_config, read_safetensors_param_counts, resolve_model_path,
        resolve_text_config, text_param_estimate, _get, _has,
    )
    _MODEL_PLAN_SOURCE = "model_plan (local)"

# path resolution is stdlib-only and lives in model_plan; kept importable from here
_is_local = is_local_model_dir

logger = logging.getLogger("chaos_llm.protocol_general")

DEFAULT_MODEL = "~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def default_device():
    return DEVICE


_TORCH_DTYPES = {
    "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
}


def to_np(t):
    """tensor -> float32 numpy. .numpy() alone crashes on bf16 tensors."""
    return t.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# parameter estimation without loading weights
# ---------------------------------------------------------------------------


def estimate_params_from(path, cfg=None, verbose=True):
    """(n_params, source) for the TEXT backbone, without loading weights.

    1) safetensors headers when local files exist (stdlib parse; excludes
       vision/mtp groups so multimodal checkpoints count only the LM);
    2) otherwise a meta-device instantiation from the config.
    Raises with an actionable message if both fail.
    """
    if os.path.isdir(path):
        total, groups = read_safetensors_param_counts(path)
        est = text_param_estimate(groups)
        if total:
            n = est if est is not None else total
            if verbose:
                top = sorted(groups.items(), key=lambda kv: -kv[1])[:4]
                logger.info("params from safetensors headers: total=%s text~=%s groups(top)=%s",
                            f"{total:,}", f"{n:,}",
                            {k: f"{v:,}" for k, v in top})
            return n, "safetensors-headers(local)"
    if cfg is None:
        cfg = AutoConfig.from_pretrained(path)
    try:
        n = _meta_param_count(cfg)
        return n, "meta-init(config)"
    except Exception as exc:  # pragma: no cover - depends on transformers version
        raise RuntimeError(
            f"could not estimate the parameter count of {path!r} ({exc}). "
            "Pass dtype explicitly (load_model(..., dtype='float32'|'bfloat16'))."
        ) from exc


def _meta_param_count(cfg):
    """Instantiate the model on the meta device and count parameters (no weights read)."""
    text_cfg = resolve_text_config(cfg)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(text_cfg)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def _from_pretrained_causal_lm(path, torch_dtype, attn, trust_remote_code, local_files_only):
    kwargs = dict(
        torch_dtype=torch_dtype, attn_implementation=attn, low_cpu_mem_usage=True,
        local_files_only=local_files_only, trust_remote_code=trust_remote_code,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except Exception as exc:
        # Nested/multimodal configs (e.g. Qwen3.5) may not map in AutoModelForCausalLM;
        # retry with the explicit text-only class derived from the config's model_type.
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(
                path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        except Exception:
            raise exc
        tried = []
        text_cfg = resolve_text_config(cfg)
        for mt in (_get(text_cfg, "model_type"), _get(cfg, "model_type")):
            name = causal_lm_class_name(mt)
            if not name or name in tried:
                continue
            tried.append(name)
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(path, **kwargs)
                logger.info("loaded %s via explicit class %s (AutoModelForCausalLM failed: %s)",
                            path, name, exc)
                return model
            except Exception:
                continue
        raise RuntimeError(
            f"AutoModelForCausalLM failed for {path!r} ({exc}) and no explicit text-only "
            f"class worked (tried {tried}; transformers {transformers.__version__}, "
            f"model_type={_get(cfg, 'model_type')!r}, architectures={_get(cfg, 'architectures')}). "
            "If this is a multimodal-era checkpoint, load its text-only *ForCausalLM class "
            "from a transformers version that supports it."
        ) from exc


def load_model(model=DEFAULT_MODEL, attn="sdpa", dtype="auto", device=None,
               trust_remote_code=False, local_files_only=None, verbose=True):
    """Load (model, tokenizer, path).

    dtype: 'auto' (default) -> float32 if the text-backbone params <= 2.5B else
    bfloat16; or an explicit 'float32' | 'bfloat16' | 'float16' override.
    Default behavior for DeepSeek-1.5B is unchanged (fp32, local_files_only=True).
    """
    path = resolve_model_path(model)
    is_local = _is_local(path)
    if local_files_only is None:
        local_files_only = bool(is_local)
    cfg = AutoConfig.from_pretrained(
        path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)

    if dtype is None or parse_dtype(dtype) == "auto":
        n_params, psrc = estimate_params_from(path, cfg, verbose=verbose)
        torch_dtype = _TORCH_DTYPES[pick_dtype(n_params)]
        if verbose:
            logger.info("dtype auto -> %s (%s; %s)", str(torch_dtype).replace("torch.", ""), psrc,
                        "fp32 if params <= 2.5e9 else bf16")
    else:
        n_params = None
        torch_dtype = _TORCH_DTYPES[parse_dtype(dtype)]

    model = _from_pretrained_causal_lm(path, torch_dtype, attn, trust_remote_code, local_files_only)
    model.to(device or DEVICE).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(
        path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    if verbose:
        info = model_info(model, tok=tok, model_path=path)
        logger.info("loaded %s", json.dumps(info, default=str))
    return model, tok, path


# ---------------------------------------------------------------------------
# decoder-layer discovery
# ---------------------------------------------------------------------------


def _resolve_path(obj, path):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def model_layers(model):
    """The list/ModuleList of decoder blocks, resolved robustly.

    Tries known paths (Qwen2/Qwen3/Phi3: model.layers; multimodal wrappers:
    model.language_model.layers; ...) and cross-checks the length against
    config.num_hidden_layers (text config when nested). Raises a clear error
    listing what it found if nothing matches.
    """
    try:
        expected = int(_get(resolve_text_config(model.config), "num_hidden_layers"))
    except Exception:
        expected = None
    candidates = layer_path_candidates(model.config)
    matches = []
    for path in candidates:
        obj = _resolve_path(model, path)
        if obj is None:
            continue
        if isinstance(obj, (torch.nn.ModuleList, list)) and len(obj) > 0:
            matches.append((path, obj))
    if not matches:
        layout = {
            "model attributes": [a for a in dir(model) if not a.startswith("_")][:60],
            "model.model attributes": [a for a in dir(getattr(model, "model", None))
                                       if not a.startswith("_")][:60],
            "config keys": sorted(getattr(model.config, "__dict__", {}).keys())[:80],
            "expected num_hidden_layers": expected,
        }
        raise RuntimeError(
            "could not locate the decoder-layer list (tried " + ", ".join(candidates) +
            "). " + json.dumps(layout, default=str))
    if expected is not None:
        for path, obj in matches:
            if len(obj) == expected:
                return obj
        path, obj = matches[0]
        logger.warning("layer list %s has %d entries but config.num_hidden_layers=%d; "
                       "using it anyway -- check the checkpoint", path, len(obj), expected)
        return obj
    return matches[0][1]


# ---------------------------------------------------------------------------
# window handling
# ---------------------------------------------------------------------------


def set_window(model, window):
    """Set (window > 0) or disable (window <= 0) the sliding window.

    Qwen2/Qwen3/Phi3 build the sdpa/eager causal mask from config.sliding_window at
    forward time, so mutating the config works; attention modules that captured the
    value in __init__ (flash-attention paths) are updated too. Architectures without
    a sliding-window mechanism (e.g. Qwen3.5 hybrid linear attention) log a warning
    and report applied=False instead of silently doing nothing.

    Returns an info dict; does not raise for unsupported models (the operator
    decides whether to skip window runs or use --hard-window).
    """
    window = int(window)
    cfg = resolve_text_config(model.config)
    attention = classify_attention(model.config)
    info = dict(window=window, applied=False, mechanism=attention,
                notes=list(attention.get("notes", [])))

    if not attention.get("window_supported"):
        for w in attention.get("warnings", []):
            info["notes"].append(w)
        logger.warning(
            "set_window(window=%d) is a NO-OP for this model: %s", window,
            " ".join(info["notes"]))
        if window <= 0:
            info["applied"] = True  # "disable" is trivially satisfied
        return info

    if window > 0:
        cfg.sliding_window = window
        if _has(cfg, "use_sliding_window"):
            cfg.use_sliding_window = True
        n_mod = 0
        try:
            blocks = model_layers(model)
        except RuntimeError:
            blocks = []
        for block in blocks:
            for sub in block.modules():
                if hasattr(sub, "sliding_window"):
                    sub.sliding_window = window
                    n_mod += 1
        info.update(applied=True, n_modules_patched=n_mod)
        info["notes"].append(
            "sdpa/eager read config.sliding_window at forward time (effective immediately); "
            f"{n_mod} attention modules also updated for flash-attention paths")
        if attention.get("max_window_layers") is not None and _has(cfg, "use_sliding_window"):
            info["notes"].append(
                f"Qwen legacy gate: init-time per-module windows used "
                f"'layer_idx >= max_window_layers={attention['max_window_layers']}'; "
                "post-load propagation above bypasses that gate for all layers")
    else:
        cfg.sliding_window = None
        if _has(cfg, "use_sliding_window"):
            cfg.use_sliding_window = False
        try:
            blocks = model_layers(model)
        except RuntimeError:
            blocks = []
        for block in blocks:
            for sub in block.modules():
                if hasattr(sub, "sliding_window"):
                    sub.sliding_window = None
        info.update(applied=True, n_modules_patched=None)
        info["notes"].append("window disabled (full attention restored)")
    logger.info("set_window -> %s", json.dumps({k: v for k, v in info.items() if k != "mechanism"}))
    return info


@torch.inference_mode()
def window_is_effective(model, window=100, seq_len=None, device=None, seed=0):
    """Empirical check: two sequences that differ ONLY before the last `window`
    tokens must produce identical final hidden states if the window clips attention.

    Returns dict(effective, windowed_max_diff, full_attention_max_diff, note).
    Restores the previous config window afterwards.
    """
    dev = device or next(model.parameters()).device
    w = int(window)
    n = seq_len or (2 * w + 32)
    if n <= w + 2:
        raise ValueError(f"seq_len ({n}) must exceed window ({w})")
    cfg = resolve_text_config(model.config)
    orig_window = _get(cfg, "sliding_window")

    vocab = int(model.get_input_embeddings().weight.shape[0])
    hi = max(1000, min(vocab - 1000, 50000))
    gen = torch.Generator().manual_seed(seed)
    head_a = torch.randint(1000, hi, (1, n - w), generator=gen)
    head_b = (head_a + 12345) % (hi - 1000) + 1000  # same shape, different content
    tail = torch.randint(1000, hi, (1, w), generator=gen)
    ids_a = torch.cat([head_a, tail], dim=1).to(dev)
    ids_b = torch.cat([head_b, tail], dim=1).to(dev)
    mask = torch.ones((1, n), dtype=torch.long, device=dev)

    def final_state(ids):
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        return out.hidden_states[-1][0, -1] if out.hidden_states is not None \
            else out.logits[0, -1]

    set_window(model, w)
    h_a = final_state(ids_a)
    h_b = final_state(ids_b)
    d_win = (h_a.float() - h_b.float()).abs().max().item()
    del h_a, h_b

    set_window(model, 0)  # disable -> full attention
    h_a = final_state(ids_a)
    h_b = final_state(ids_b)
    d_full = (h_a.float() - h_b.float()).abs().max().item()
    del h_a, h_b

    # restore
    if isinstance(orig_window, int) and orig_window > 0:
        set_window(model, orig_window)
    else:
        set_window(model, 0)

    if d_full == 0.0:
        note = ("INCONCLUSIVE: the two prefixes already give identical final states under "
                "full attention (test too weak); try a longer seq_len")
        effective = False
    else:
        effective = d_win == 0.0
        note = ("window clips attention: final states identical under the window"
                if effective else
                f"window NOT effective (final states differ by {d_win:g}); "
                "use --hard-window or investigate the attention implementation")
    return dict(window=w, seq_len=n, effective=effective,
                windowed_max_diff=d_win, full_attention_max_diff=d_full, note=note)


def hard_window_mask(L, W, dtype=None, device=None):
    """4D additive mask that hard-clips attention to the last W tokens.

    dtype must match the model's hidden dtype (sdpa rejects mixed-dtype masks);
    defaults to float32 for fp32 models like the original run_a1 helper.
    """
    dt = dtype or torch.float32
    m = torch.full((1, 1, L, L), torch.finfo(dt).min, dtype=dt, device=device or DEVICE)
    for i in range(L):
        m[0, 0, i, max(0, i - W + 1):i + 1] = 0
    return m


# ---------------------------------------------------------------------------
# tokenizer / directions / generation (protocol core)
# ---------------------------------------------------------------------------


def prompt_ids(tokenizer, text, add_special_tokens=False, device=None):
    """Token ids for the RAW-COMPLETION protocol: no chat template, no BOS/EOS.

    add_special_tokens=False is explicit so every tokenizer behaves the same;
    for the Qwen2 tokenizer of DeepSeek this is identical to the old default.
    """
    return tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens
                     ).input_ids.to(device or DEVICE)


def build_simplex(k):
    """Rows of a regular (k-1)-simplex in R^k, centered at 0, unit norm.

    k must be >= 2: k=1 gives the zero vector, whose 'normalization' is NaN and
    downstream produced the NaN runs documented in docs/GPU_GUIDE.md.
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"build_simplex needs k >= 2 (got {k}); k=1 is degenerate "
                         "(zero direction -> NaN); use 4-20 directions")
    eye = np.eye(k, dtype=np.float32)
    v = eye - np.ones((k, k), dtype=np.float32) / k
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def subspace_indices(total_dim, k, seed):
    k = int(k)
    if k > total_dim:
        raise ValueError(f"cannot place {k} directions in a {total_dim}-dimensional "
                          "prompt subspace; increase prompt length or reduce directions")
    rng = np.random.default_rng(seed)
    return rng.choice(total_dim, size=k, replace=False).astype(np.int64)


def make_directions(num_directions, prompt_len, dim, seed=69):
    """(simplex, subspace indices) as in protocol.py, but refuses k < 2."""
    if int(num_directions) < 2:
        raise ValueError("num_directions must be >= 2 (degenerate simplex -> NaN); "
                         "see docs/GPU_GUIDE.md rule 5")
    simplex = build_simplex(num_directions)
    sub = subspace_indices(prompt_len * dim, num_directions, seed)
    return simplex, sub


@torch.inference_mode()
def generate(model, embeds, max_new_tokens, batch_meta=None):
    """Greedy fixed-length generation from embeddings.

    eos_token_id=None => never stops early (the trajectory runs the full horizon,
    even past the model's own EOS); pad_token_id=0 is bookkeeping only because
    no row can finish. Asserts the returned length so a silent early stop can
    never pass unnoticed.
    """
    B, P, _ = embeds.shape
    mask = torch.ones((B, P), dtype=torch.long, device=embeds.device)
    out = model.generate(
        inputs_embeds=embeds, attention_mask=mask, max_new_tokens=int(max_new_tokens),
        do_sample=False, eos_token_id=None, pad_token_id=0,
    )
    if out.shape[1] != int(max_new_tokens):
        raise RuntimeError(
            f"generate returned {out.shape[1]} tokens, expected {max_new_tokens}. "
            "The installed transformers stopped early or changed the inputs_embeds "
            "return convention; the protocol requires fixed-length greedy runs.")
    return out


@torch.inference_mode()
def forward(model, embeds, output_hidden_states=False):
    B, L, _ = embeds.shape
    mask = torch.ones((B, L), dtype=torch.long, device=embeds.device)
    return model(inputs_embeds=embeds, attention_mask=mask,
                 output_hidden_states=output_hidden_states, use_cache=False)


@torch.inference_mode()
def full_embeddings(model, prompt_ids_t, gen_ids, prompt_delta=None):
    """Embed prompt+generated tokens; optionally add a [P,D] or [B,P,D] prompt delta.

    Wrapped in inference_mode (bug #1: an unwrapped embedding lookup keeps the
    autograd graph alive). Perturbation arithmetic convention used across the
    patched scripts: accumulate in fp32, round ONCE to the embedding dtype.
    """
    ids = torch.cat([prompt_ids_t, gen_ids], dim=1)
    E = model.get_input_embeddings()(ids)
    if prompt_delta is not None:
        d = prompt_delta
        if d.dim() == 2:
            d = d.unsqueeze(0)
        n = d.shape[1]
        E[:, :n] = (E[:, :n].float() + d.float()).to(E.dtype)
    return E


# ---------------------------------------------------------------------------
# introspection / io
# ---------------------------------------------------------------------------


def model_info(model, tok=None, model_path=None):
    """Everything model-specific the scripts need (replaces hardcoded 1536/27/28)."""
    cfg = resolve_text_config(model.config)
    try:
        blocks = model_layers(model)
        n_layers = len(blocks)
        layers_path = next((p for p in layer_path_candidates(model.config)
                            if _resolve_path(model, p) is blocks), None)
    except RuntimeError as exc:
        logger.warning("model_layers failed: %s", exc)
        n_layers = int(_get(cfg, "num_hidden_layers"))
        layers_path = None
    params = sum(p.numel() for p in model.parameters())
    first = next(model.parameters())
    return {
        "model_class": type(model).__name__,
        "model_path": model_path,
        "hidden_size": int(_get(cfg, "hidden_size")),
        "num_layers": n_layers,
        "n_hidden_states": n_layers + 1,
        "vocab_size": int(_get(cfg, "vocab_size")),
        "embedding_rows": int(model.get_input_embeddings().weight.shape[0]),
        "dtype": str(first.dtype).replace("torch.", ""),
        "params": int(params),
        "layers_path": layers_path,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "sliding_window": _get(cfg, "sliding_window"),
        "use_sliding_window": bool(_get(cfg, "use_sliding_window", False)),
        "layer_types": _get(cfg, "layer_types"),
        "tie_word_embeddings": bool(_get(cfg, "tie_word_embeddings", False)),
        "device": str(first.device),
        "tokenizer_class": type(tok).__name__ if tok is not None else None,
        "tokenizer_bos": getattr(tok, "bos_token_id", None) if tok is not None else None,
        "tokenizer_eos": getattr(tok, "eos_token_id", None) if tok is not None else None,
        "tokenizer_pad": getattr(tok, "pad_token_id", None) if tok is not None else None,
    }


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


__all__ = [
    "DEFAULT_MODEL", "DEVICE", "FP32_MAX_PARAMS",
    "resolve_model_path", "load_model", "estimate_params_from",
    "model_layers", "model_info", "set_window", "window_is_effective",
    "hard_window_mask", "prompt_ids", "build_simplex", "subspace_indices",
    "make_directions", "generate", "forward", "full_embeddings", "save_json",
    "to_np", "default_device", "parse_dtype", "pick_dtype",
    "plan_from_config", "resolve_text_config", "default_layer_indices",
    "read_safetensors_param_counts",
]
