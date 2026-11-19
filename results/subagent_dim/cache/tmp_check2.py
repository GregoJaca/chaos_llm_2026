import numpy as np
ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
for tag, Lexp in [("deepseek", 28), ("qwen3_4b", 36)]:
    a = np.load(ROOT + r"\dim" + "\\" + tag + r"\acts.npz")
    acts = a["acts"]
    for p in range(3):
        h_last = acts[p, Lexp, ::40].astype(np.float32)   # extra index
        h_prev = acts[p, Lexp - 1, ::40].astype(np.float32)
        c = (h_last * h_prev).sum(1) / (np.linalg.norm(h_last, axis=1) * np.linalg.norm(h_prev, axis=1))
        print(tag, "prompt", p, "cos(h_extra, h_prev) mean", round(float(c.mean()), 3), "min", round(float(c.min()), 3))
