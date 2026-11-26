import numpy as np, json, os
base = os.path.expanduser("~/LLM/chaos_llm_2026")
for p in ["results/a1_ds_w100_layers/a1.npz",
          "results/models/qwen3_4b/a1/a1.npz",
          "results/models/qwen25_3b/a1/a1.npz",
          "results/models/phi4mini/a1/a1.npz",
          "results/models/qwen3_5/a1/a1.npz"]:
    try:
        z = np.load(os.path.join(base, p), allow_pickle=True)
        print(p, z["dh_norm"].shape, z["layers"], z["radii"])
    except Exception as e:
        print(p, "ERR", e)
