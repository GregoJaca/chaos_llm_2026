import json, glob, os
paths = {
 "qwen3_4b": "/home/grego/LLM/models--Qwen--Qwen3-4B-Instruct-2507/config.json",
 "phi4mini": "/home/grego/LLM/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/config.json",
 "qwen3_5": "/home/grego/LLM/models--Qwen--Qwen3.5-2B/config.json",
}
for name, p in paths.items():
    c = json.load(open(p))
    tc = c.get("text_config", c) if isinstance(c.get("text_config"), dict) else c
    print("==", name, c.get("model_type"))
    for k in ["sliding_window", "use_sliding_window", "max_window_layers", "layer_types",
              "num_hidden_layers", "hidden_size", "attention_types"]:
        v = tc.get(k, c.get(k))
        if isinstance(v, list):
            v = (v[:6], "len", len(v))
        print("   ", k, "=", v)
