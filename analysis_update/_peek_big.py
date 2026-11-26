import numpy as np, os
base = os.path.expanduser("~/LLM/chaos_llm_2026")
for p in ["results_big/qwen314/a1_inter_w0/a1.npz",
          "results_big/phi4/a1_inter_w0/a1.npz",
          "results_big/ds14/a1_inter_w100/a1.npz"]:
    z = np.load(os.path.join(base, p), allow_pickle=True)
    print(p)
    for k in z.files:
        print("  ", k, z[k].shape, z[k].dtype)
