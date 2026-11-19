"""model_plan.py -- stdlib-only planning core for the multi-model protocol.

No torch, no transformers imports: this module only inspects configs (PreTrainedConfig
objects or plain dicts) and safetensors headers, so it can be unit-tested and used on
a CPU-only / no-dependency machine (see selftest_cpu.py).

protocol_general.py builds on these helpers; keep the two files in sync.
"""
from __future__ import annotations

import glob
import json
import os
import struct

# ---------------------------------------------------------------------------
# model path resolution (local dirs, HF-style cache dirs, HF hub ids)
# ---------------------------------------------------------------------------


def _hub_cache_roots():
    """Candidate HF cache directories (HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE
    and the default), each followed by its 'hub' subdir, de-duplicated in order."""
    roots, seen = [], set()

    def add(p):
        if p and p not in seen:
            seen.add(p)
            roots.append(p)

    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        add(os.path.expanduser(os.environ.get(var, "")))
    add(os.path.expanduser("~/.cache/huggingface"))
    out = []
    for r in roots:
        out.append(r)
        out.append(os.path.join(r, "hub"))
    return [o for o in out if o]


def _latest_snapshot(repo_dir, log=None):
    """Snapshot dir of a hub-style repo dir; prefers refs/main over lexicographic sort."""
    ref = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref):
        try:
            rev = open(ref, encoding="utf-8").read().strip()
            snap = os.path.join(repo_dir, "snapshots", rev)
            if os.path.isdir(snap):
                return snap
        except OSError:
            pass
    snaps = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*", "config.json")))
    if snaps:
        if log is not None:
            log("no refs/main in %s; using lexicographically last snapshot "
                "(may not be the newest revision)" % repo_dir)
        return os.path.dirname(snaps[-1])
    return None


def resolve_model_path(model):
    """Expand ~, accept plain local dirs, hub-style cache dirs, and HF hub ids.

    Returns a local path when one exists, otherwise the (possibly nonexistent)
    input string, which then acts as an HF hub id. Never raises.
    """
    p = os.path.expanduser(str(model))
    if os.path.isdir(p):
        if os.path.isfile(os.path.join(p, "config.json")):
            return p
        if os.path.isdir(os.path.join(p, "snapshots")):  # bare repo dir, e.g. ~/LLM/models--X--Y
            snap = _latest_snapshot(p)
            if snap:
                return snap
    cand = "models--" + str(model).replace("/", "--")
    for root in _hub_cache_roots():
        repo = os.path.join(root, cand)
        if os.path.isdir(repo):
            snap = _latest_snapshot(repo)
            if snap:
                return snap
    return p


def is_local_model_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


# ---------------------------------------------------------------------------
# generic attribute access (works on PreTrainedConfig and plain dict)
# ---------------------------------------------------------------------------


def _has(cfg, key):
    if cfg is None:
        return False
    if isinstance(cfg, dict):
        return key in cfg
    return hasattr(cfg, key)


def _get(cfg, key, default=None):
    """Value of `key`; `default` only when the key is absent (None counts as present)."""
    if not _has(cfg, key):
        return default
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def resolve_text_config(cfg):
    """Unwrap multimodal nesting: return the text-backbone config (Qwen3_5 etc.).

    Falls back to `cfg` itself for flat causal-LM configs (Qwen2/Qwen3/Phi3).
    """
    cur = cfg
    for _ in range(4):  # depth guard
        if cur is None or not _has(cur, "text_config"):
            return cur
        nxt = _get(cur, "text_config")
        if nxt is None:
            return cur
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
# dtype decision
# ---------------------------------------------------------------------------

FP32_MAX_PARAMS = 2_500_000_000  # 2.5B: fp32 weights ~10 GB; fits the 15 GB T4 only below this

_DTYPE_ALIASES = {
    "float32": "float32", "fp32": "float32", "float": "float32", "32": "float32",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16", "half": "float16", "16": "float16",
}


def parse_dtype(name):
    """'float32' | 'bfloat16' | 'float16' (aliases accepted). Raises on unknown names."""
    if name is None:
        raise ValueError("dtype is None; use 'auto', 'float32', 'bfloat16' or 'float16'")
    key = str(name).strip().lower().replace("torch.", "")
    if key in ("auto", ""):
        return "auto"
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"unknown dtype {name!r}; use auto, float32 (fp32), bfloat16 (bf16) or float16 (fp16)")
    return _DTYPE_ALIASES[key]


def pick_dtype(params, override=None):
    """fp32 if params <= 2.5B else bf16; explicit override wins."""
    ov = parse_dtype(override) if override is not None else None
    if ov is not None and ov != "auto":
        return ov
    if params is None:
        raise ValueError("cannot pick dtype automatically: parameter count unknown; pass an override")
    return "float32" if params <= FP32_MAX_PARAMS else "bfloat16"


# ---------------------------------------------------------------------------
# layer paths and default measurement layers
# ---------------------------------------------------------------------------

_LAYER_PATH_CANDIDATES = [
    "model.layers", "model.language_model.layers", "model.model.layers",
    "backbone.layers", "transformer.h", "decoder.layers", "gpt_neox.layers",
]

# architectures we verified in the transformers source (4.49-5.x)
_ARCH_LAYER_PATHS = {
    "Qwen2ForCausalLM": ["model.layers"],
    "Qwen2_5ForCausalLM": ["model.layers"],
    "Qwen3ForCausalLM": ["model.layers"],
    "Phi3ForCausalLM": ["model.layers"],
    "LlamaForCausalLM": ["model.layers"],
    "MistralForCausalLM": ["model.layers"],
    "Qwen3_5ForCausalLM": ["model.layers"],
    "Qwen3_5ForConditionalGeneration": [
        "model.language_model.layers", "model.layers", "model.model.layers"],
    "Qwen2VLForConditionalGeneration": ["model.layers"],
    "Gemma3ForConditionalGeneration": ["model.language_model.layers"],
}


def causal_lm_class_name(model_type):
    """'qwen3_5_text'/'qwen3_5' -> 'Qwen3_5ForCausalLM'; 'phi3' -> 'Phi3ForCausalLM'."""
    mt = str(model_type or "").strip()
    if mt.endswith("_text"):
        mt = mt[:-len("_text")]
    if not mt:
        return None
    return mt[0].upper() + mt[1:] + "ForCausalLM"


def layer_path_candidates(cfg):
    archs = _get(cfg, "architectures") or _get(resolve_text_config(cfg), "architectures") or []
    out = []
    for a in archs:
        for p in _ARCH_LAYER_PATHS.get(a, []):
            if p not in out:
                out.append(p)
    for p in _LAYER_PATH_CANDIDATES:
        if p not in out:
            out.append(p)
    return out


def default_layer_indices(n):
    """Spread 5 measurement layers over n blocks; reproduces '0,7,14,21,27' for n=28."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {n}")
    if n <= 5:
        return list(range(n))
    return sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))


# ---------------------------------------------------------------------------
# sliding-window support detection
# ---------------------------------------------------------------------------

_NON_TEXT_GROUPS = (
    "visual", "vision", "mtp", "vision_tower",
    "model.visual", "model.vision", "model.mtp", "model.vision_tower",
)


def classify_attention(cfg):
    """What mechanism (if any) controls the attention window for this checkpoint.

    Returns a dict; `window_supported` False means set_window() is a no-op and
    any window study must use an explicit 4D mask (--hard-window) or be skipped.
    """
    tc = resolve_text_config(cfg)
    notes, warnings = [], []
    lt = _get(tc, "layer_types")
    if lt:
        kinds = sorted(set(str(x) for x in lt))
        counts = {k: sum(1 for x in lt if str(x) == k) for k in kinds}
        if "linear_attention" in kinds:
            return {
                "window_supported": False, "window_key": None,
                "hybrid": "linear_attention", "layer_types": counts,
                "notes": ["hybrid linear-attention (Gated DeltaNet) + full-attention stack: "
                          "the sliding-window protocol is NOT defined for this model; "
                          "set_window will be a no-op (linear layers carry unbounded recurrent state)"],
            }
        if "sliding_attention" in kinds:
            warnings.append(
                "Qwen3-style hybrid attention: only 'sliding_attention' layers honour "
                f"config.sliding_window ({counts}); 'full_attention' layers stay unwindowed")
    if not _has(tc, "sliding_window"):
        return {
            "window_supported": False, "window_key": None, "hybrid": None,
            "layer_types": None,
            "notes": ["config has no `sliding_window` attribute: this architecture has no "
                      "sliding-window mechanism; set_window is a no-op and window runs are invalid"],
        }
    info = {
        "window_supported": True, "window_key": "sliding_window", "hybrid": None,
        "sliding_window": _get(tc, "sliding_window"),
        "use_sliding_window": bool(_get(tc, "use_sliding_window", False)),
        "max_window_layers": _get(tc, "max_window_layers"),
        "layer_types": None, "notes": notes, "warnings": warnings,
    }
    if _has(tc, "use_sliding_window"):
        notes.append(
            "Qwen2/Qwen3-style: sdpa/eager causal masks are rebuilt from config.sliding_window "
            "at every forward, so post-load mutation takes effect; flash_attention_2 instead uses "
            "per-attention copies captured in __init__ (protocol_general.set_window propagates to "
            "those modules too, but prefer sdpa for window runs)")
    if _has(tc, "full_attn_mod"):
        notes.append(
            f"Phi3-style: full_attn_mod={_get(tc, 'full_attn_mod')}; native modeling applies "
            "config.sliding_window uniformly at forward time (this checkpoint ships a huge "
            "'sliding_window' value, i.e. effectively full attention; setting a small window is "
            "the same *protocol intervention* as on DeepSeek)")
    return info


# ---------------------------------------------------------------------------
# parameter counting without loading weights
# ---------------------------------------------------------------------------


def read_safetensors_param_counts(model_dir):
    """(total_numel, {group: numel}) from safetensors headers, stdlib only.

    Groups are keyed by the first two path components, e.g. 'model.layers',
    'model.embed_tokens', 'lm_head', 'model.visual'.
    """
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    total, groups = 0, {}
    for fp in files:
        with open(fp, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n).decode("utf-8"))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            shape = meta.get("shape") if isinstance(meta, dict) else None
            numel = 1
            for s in shape or []:
                numel *= int(s)
            parts = str(key).split(".")
            grp = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            groups[grp] = groups.get(grp, 0) + int(numel)
            total += int(numel)
    return total, groups


def text_param_estimate(groups):
    """Total minus vision/mtp groups; None when `groups` is empty."""
    if not groups:
        return None
    non_text = 0
    for k, v in groups.items():
        if any(k == p or k.startswith(p + ".") for p in _NON_TEXT_GROUPS):
            non_text += v
    return sum(groups.values()) - non_text


# ---------------------------------------------------------------------------
# the plan: everything the operator needs before touching the GPU
# ---------------------------------------------------------------------------


def plan_from_config(cfg, model_dir=None, params=None, dtype_override=None):
    """Build the per-model execution plan from a config alone (no weights).

    `params` may be supplied (e.g. from a meta-device count); otherwise local
    safetensors headers are used when `model_dir` is given.
    """
    tc = resolve_text_config(cfg)
    n_layers = _get(tc, "num_hidden_layers")
    if n_layers is None:
        raise ValueError(
            "config has no num_hidden_layers; this does not look like a decoder-only LM config. "
            f"Found keys: {sorted(_get(tc, '__dict__', {}) or [])}"
            if not isinstance(tc, dict) else
            "config has no num_hidden_layers; this does not look like a decoder-only LM config.")
    attention = classify_attention(cfg)
    params_source = "provided"
    groups = None
    if params is None and model_dir and os.path.isdir(model_dir):
        total, groups = read_safetensors_param_counts(model_dir)
        est = text_param_estimate(groups)
        if total:
            params = est if est is not None else total
            params_source = f"safetensors headers (total={total:,}, text~={params:,})"
    dtype, dtype_reason = None, "not decided"
    ov = parse_dtype(dtype_override) if dtype_override is not None else None
    if ov is not None and ov != "auto":
        dtype = ov
        dtype_reason = "explicit override"
    elif params is not None:
        dtype = pick_dtype(params)
        dtype_reason = (f"params {params:,} <= {FP32_MAX_PARAMS:,} -> float32"
                        if dtype == "float32" else
                        f"params {params:,} > {FP32_MAX_PARAMS:,} -> bfloat16")
    else:
        dtype_reason = ("unknown: run selftest where the model files (or meta-init) are "
                        "available, or pass --dtype explicitly")
    return {
        "model_type": _get(cfg, "model_type"),
        "text_model_type": _get(tc, "model_type"),
        "architectures": _get(cfg, "architectures"),
        "hidden_size": _get(tc, "hidden_size"),
        "num_layers": int(n_layers),
        "n_hidden_states": int(n_layers) + 1,
        "vocab_size": _get(tc, "vocab_size"),
        "tie_word_embeddings": bool(_get(tc, "tie_word_embeddings", False)),
        "config_torch_dtype": _get(tc, "torch_dtype") or _get(tc, "dtype"),
        "num_attention_heads": _get(tc, "num_attention_heads"),
        "num_key_value_heads": _get(tc, "num_key_value_heads"),
        "max_position_embeddings": _get(tc, "max_position_embeddings"),
        "bos_token_id": _get(cfg, "bos_token_id", _get(tc, "bos_token_id")),
        "eos_token_id": _get(cfg, "eos_token_id", _get(tc, "eos_token_id")),
        "pad_token_id": _get(cfg, "pad_token_id", _get(tc, "pad_token_id")),
        "attention": attention,
        "window_key": attention.get("window_key"),
        "window_supported": attention.get("window_supported"),
        "layer_path_candidates": layer_path_candidates(cfg),
        "default_layers": default_layer_indices(n_layers),
        "params_estimate": params,
        "param_groups": groups,
        "params_source": params_source,
        "dtype": dtype,
        "dtype_reason": dtype_reason,
        "prompt_protocol": "raw completion: tokenizer(text, add_special_tokens=False), no chat template",
        "generation_protocol": "greedy, eos_token_id=None, pad_token_id=0, fixed max_new_tokens",
    }
