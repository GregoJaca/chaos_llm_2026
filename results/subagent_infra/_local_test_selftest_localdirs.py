"""Build local model dirs (real hub config.json + synthetic safetensors headers)
and run selftest_cpu.py against them, verifying the printed plan and dtype decisions.

Runs entirely without torch/transformers: validates section (c) of the selftest.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_plan as mp

CONFIGS = {
    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B": {
        "config": {
            "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
            "hidden_size": 1536, "num_hidden_layers": 28, "vocab_size": 151936,
            "sliding_window": 4096, "use_sliding_window": False, "max_window_layers": 21,
            "tie_word_embeddings": False, "torch_dtype": "bfloat16",
            "bos_token_id": 151643, "eos_token_id": 151643,
        },
        "tensors": {
            "model.embed_tokens.weight": [151936, 1536],
            "lm_head.weight": [151936, 1536],
            "model.layers.0.big.weight": [1_313_651_968],
        },
        "expect_dtype": "float32",   # 1.78B
    },
    "models--Qwen--Qwen3-4B-Instruct-2507": {
        "config": {
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151936,
            "sliding_window": None, "use_sliding_window": False, "max_window_layers": 36,
            "tie_word_embeddings": True, "torch_dtype": "bfloat16",
            "eos_token_id": 151645, "bos_token_id": 151643,
        },
        "tensors": {
            "model.embed_tokens.weight": [151936, 2560],
            "lm_head.weight": [151936, 2560],
            "model.layers.0.big.weight": [3_240_887_680],
        },
        "expect_dtype": "bfloat16",  # 4.02B
    },
    "models--microsoft--Phi-4-mini-instruct": {
        "config": {
            "architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
            "hidden_size": 3072, "num_hidden_layers": 32, "vocab_size": 200064,
            "sliding_window": 262144, "full_attn_mod": 1, "tie_word_embeddings": True,
            "torch_dtype": "bfloat16", "original_max_position_embeddings": 4096,
            "rope_scaling": {"type": "longrope"}, "partial_rotary_factor": 0.75,
        },
        "tensors": {
            "model.embed_tokens.weight": [200064, 3072],
            "lm_head.weight": [200064, 3072],
            "model.layers.0.big.weight": [2_570_806_784],
        },
        "expect_dtype": "bfloat16",  # 3.8B
    },
    "models--Qwen--Qwen3.5-2B": {
        "config": {
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": 2048,
                "num_hidden_layers": 24, "vocab_size": 248320,
                "tie_word_embeddings": True, "dtype": "bfloat16",
                "full_attention_interval": 4, "mamba_ssm_dtype": "float32",
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
                "eos_token_id": 248044,
            },
            "vision_config": {"depth": 24, "hidden_size": 1024},
        },
        "tensors": {
            "model.embed_tokens.weight": [248320, 2048],
            "lm_head.weight": [248320, 2048],
            "model.layers.0.big.weight": [1_182_881_280],
            "model.visual.big.weight": [300_000_000],   # must be excluded from text estimate
            "mtp.0.big.weight": [50_000_000],          # must be excluded from text estimate
        },
        "expect_dtype": "float32",   # text ~2.2B <= 2.5B despite vision/mtp extras
    },
}


def build_dir(root, name, spec):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(spec["config"], f)
    header = {k: {"dtype": "BF16", "shape": v} for k, v in spec["tensors"].items()}
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode("utf-8")
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
    return d


def main():
    tmp = tempfile.mkdtemp(prefix="chaos_selftest_models_")
    dirs = {name: build_dir(tmp, name, spec) for name, spec in CONFIGS.items()}

    # sanity: text_param_estimate must drop the vision/mtp groups for Qwen3.5
    total, groups = mp.read_safetensors_param_counts(dirs["models--Qwen--Qwen3.5-2B"])
    est = mp.text_param_estimate(groups)
    assert est == 2_200_000_000 and total == 2_550_000_000, (est, total)
    print(f"qwen3.5 groups: total={total:,} text_estimate={est:,} (vision/mtp excluded)")

    cmd = [sys.executable, os.path.join(HERE, "selftest_cpu.py"),
           "--skip-tokenizers", "--models"] + list(dirs.values())
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        print("SELFTEST RETURNED", r.returncode)
        sys.exit(1)

    # verify the plan output per model
    ok = True
    lines = r.stdout.splitlines()
    for name, spec in CONFIGS.items():
        d = dirs[name]
        block_start = next(i for i, ln in enumerate(lines) if f"MODEL: {d}" in ln)
        block = "\n".join(lines[block_start:block_start + 40])
        want = spec["expect_dtype"]
        got = None
        for ln in block.splitlines():
            stripped = ln.strip()
            if stripped.startswith("dtype ") or stripped.startswith("dtype:"):
                got = stripped.split(":", 1)[1].strip()
                break
        cfg = CONFIGS[name]["config"]
        tc = cfg.get("text_config", cfg)
        exp_layers = mp.default_layer_indices(tc["num_hidden_layers"])
        print(f"-> {name}: dtype_line={got!r} expected={want!r}; "
              f"default_layers={exp_layers} present={str(exp_layers) in block}")
        if got is None or want not in got:
            ok = False
        if str(exp_layers) not in block:
            ok = False
    if not ok:
        print("MISMATCH: see above")
        sys.exit(1)
    print("ALL LOCAL selftest-with-model-dirs CHECKS PASSED")


if __name__ == "__main__":
    main()
