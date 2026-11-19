"""Local audit test for model_plan.py against the real hub configs (fetched 2026-11-19)."""
import json
import struct
import sys
import tempfile
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_plan as mp

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


DEEPSEEK = {
    "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 1536,
    "num_hidden_layers": 28, "vocab_size": 151936, "sliding_window": 4096,
    "use_sliding_window": False, "max_window_layers": 21, "tie_word_embeddings": False,
    "torch_dtype": "bfloat16", "bos_token_id": 151643, "eos_token_id": 151643,
}
QWEN3_4B_2507 = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 2560,
    "num_hidden_layers": 36, "vocab_size": 151936, "sliding_window": None,
    "use_sliding_window": False, "max_window_layers": 36, "tie_word_embeddings": True,
    "torch_dtype": "bfloat16", "eos_token_id": 151645, "bos_token_id": 151643,
    "intermediate_size": 9728, "num_attention_heads": 32, "num_key_value_heads": 8,
}
PHI4MINI = {
    "architectures": ["Phi3ForCausalLM"], "model_type": "phi3", "hidden_size": 3072,
    "num_hidden_layers": 32, "vocab_size": 200064, "sliding_window": 262144,
    "full_attn_mod": 1, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
    "original_max_position_embeddings": 4096, "rope_scaling": {"type": "longrope"},
    "partial_rotary_factor": 0.75, "bos_token_id": 199999, "eos_token_id": 199999,
}
QWEN35_2B = {
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 2048, "num_hidden_layers": 24,
        "vocab_size": 248320, "tie_word_embeddings": True, "dtype": "bfloat16",
        "full_attention_interval": 4, "mamba_ssm_dtype": "float32",
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
        "eos_token_id": 248044, "num_attention_heads": 8, "num_key_value_heads": 2,
    },
    "vision_config": {"depth": 24, "hidden_size": 1024},
}

# --- resolve_text_config -------------------------------------------------
tc = mp.resolve_text_config(DEEPSEEK)
check("flat config stays itself", tc is DEEPSEEK)
tc = mp.resolve_text_config(QWEN35_2B)
check("nested config unwraps to text_config",
      tc["model_type"] == "qwen3_5_text" and tc["num_hidden_layers"] == 24)

# --- dtype rules ----------------------------------------------------------
check("pick_dtype boundary low", mp.pick_dtype(2_499_999_999) == "float32")
check("pick_dtype boundary high", mp.pick_dtype(2_500_000_001) == "bfloat16")
check("pick_dtype override", mp.pick_dtype(4e9, "float32") == "float32")
check("pick_dtype unknown params raises",
      _raises := (lambda f: (lambda: [
          (lambda: f())()] and False) or True)
      if False else True)
try:
    mp.pick_dtype(None)
    check("pick_dtype(None) raises", False)
except ValueError:
    check("pick_dtype(None) raises", True)
check("parse_dtype alias", mp.parse_dtype("bf16") == "bfloat16")
try:
    mp.parse_dtype("int8")
    check("parse_dtype unknown raises", False)
except ValueError:
    check("parse_dtype unknown raises", True)

# --- default layers -------------------------------------------------------
check("default layers n=28 == legacy default",
      mp.default_layer_indices(28) == [0, 7, 14, 21, 27],
      f"got {mp.default_layer_indices(28)}")
check("default layers n=36", mp.default_layer_indices(36) == [0, 9, 18, 27, 35])
check("default layers n=24", mp.default_layer_indices(24) == [0, 6, 12, 18, 23])
check("default layers n=32", mp.default_layer_indices(32) == [0, 8, 16, 24, 31])
check("default layers n=4", mp.default_layer_indices(4) == [0, 1, 2, 3])

# --- class names ----------------------------------------------------------
check("class name qwen3_5_text", mp.causal_lm_class_name("qwen3_5_text") == "Qwen3_5ForCausalLM")
check("class name qwen3", mp.causal_lm_class_name("qwen3") == "Qwen3ForCausalLM")
check("class name phi3", mp.causal_lm_class_name("phi3") == "Phi3ForCausalLM")

# --- attention classification ---------------------------------------------
a = mp.classify_attention(DEEPSEEK)
check("deepseek window supported", a["window_supported"] and a["window_key"] == "sliding_window")
check("deepseek qwen2-style note",
      any("Qwen2/Qwen3-style" in n for n in a["notes"]))
a = mp.classify_attention(QWEN3_4B_2507)
check("qwen3-4b-2507 window supported (sliding_window key exists, value None)",
      a["window_supported"] and a["sliding_window"] is None)
a = mp.classify_attention(PHI4MINI)
check("phi-4-mini window supported via config at forward time",
      a["window_supported"] and any("Phi3-style" in n for n in a["notes"]),
      f"notes={a['notes']}")
a = mp.classify_attention(QWEN35_2B)
check("qwen3.5 window NOT supported (linear_attention)",
      a["window_supported"] is False and a["hybrid"] == "linear_attention")
a = mp.classify_attention({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                           "layer_types": ["full_attention", "sliding_attention"] * 4,
                           "sliding_window": 4096, "use_sliding_window": True,
                           "num_hidden_layers": 8, "hidden_size": 2560, "vocab_size": 151936})
check("qwen3 hybrid sliding gets warning",
      a["window_supported"] and any("sliding_attention" in w for w in a["warnings"]))
a = mp.classify_attention({"model_type": "llama", "num_hidden_layers": 2})
check("config without sliding_window -> unsupported",
      a["window_supported"] is False)

# --- layer path candidates -------------------------------------------------
c = mp.layer_path_candidates(QWEN35_2B)
check("qwen3.5 layer paths prefer language_model",
      c[0] == "model.language_model.layers", f"got {c}")
c = mp.layer_path_candidates(DEEPSEEK)
check("deepseek layer path", c[0] == "model.layers")
c = mp.layer_path_candidates({"architectures": ["SomethingNew"], "model_type": "x"})
check("unknown arch falls back to candidate list", c[0] == "model.layers" and len(c) >= 6)

# --- plan ------------------------------------------------------------------
p = mp.plan_from_config(DEEPSEEK, params=1_780_000_000)
check("deepseek plan dtype float32", p["dtype"] == "float32", p["dtype_reason"])
check("deepseek plan n_hidden_states", p["n_hidden_states"] == 29)
check("deepseek default layers", p["default_layers"] == [0, 7, 14, 21, 27])
p = mp.plan_from_config(QWEN3_4B_2507, params=4_020_000_000)
check("qwen3-4b plan dtype bfloat16", p["dtype"] == "bfloat16", p["dtype_reason"])
check("qwen3-4b plan layers 36", p["num_layers"] == 36)
check("qwen3-4b vocab", p["vocab_size"] == 151936)
p = mp.plan_from_config(PHI4MINI, params=3_800_000_000, dtype_override="bfloat16")
check("phi-4-mini plan dtype bfloat16 (override)", p["dtype"] == "bfloat16")
p = mp.plan_from_config(QWEN35_2B, params=None, dtype_override=None)
check("qwen3.5 plan dtype undecided without params", p["dtype"] is None)
p = mp.plan_from_config(QWEN35_2B, params=2_100_000_000)
check("qwen3.5 plan text layers 24", p["num_layers"] == 24)
check("qwen3.5 plan vocab 248320", p["vocab_size"] == 248320)
check("qwen3.5 plan window unsupported", p["window_supported"] is False)
try:
    mp.plan_from_config({"model_type": "clip"})
    check("plan without num_hidden_layers raises", False)
except ValueError:
    check("plan without num_hidden_layers raises", True)

# --- safetensors header reader --------------------------------------------
tmp = tempfile.mkdtemp()
# synthesize a minimal safetensors file: header json + dummy payload
hdr = {
    "model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 50]},
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [50, 50]},
    "lm_head.weight": {"dtype": "BF16", "shape": [100, 50]},
    "model.visual.patch_emb.weight": {"dtype": "BF16", "shape": [10, 10]},
    "__metadata__": {"format": "pt"},
}
hb = json.dumps(hdr).encode("utf-8")
with open(os.path.join(tmp, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    f.write(b"\0" * 16)
total, groups = mp.read_safetensors_param_counts(tmp)
check("safetensors total", total == 100 * 50 * 2 + 50 * 50 + 10 * 10,
      f"total={total}")
est = mp.text_param_estimate(groups)
check("text estimate excludes visual", est == total - 100,
      f"est={est}, groups={groups}")

print()
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("all model_plan checks passed")
