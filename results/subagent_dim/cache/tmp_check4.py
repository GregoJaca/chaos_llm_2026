import numpy as np
ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
for tag, Lp in [("deepseek", 28), ("qwen3_4b", 36)]:
    a = np.load(ROOT + r"\dim" + "\\" + tag + r"\acts.npz")
    acts = a["acts"][0].astype(np.float32)  # [L+1, T, D]
    print("===", tag)
    for l in range(1, Lp + 1):
        h1, h0 = acts[l, ::20], acts[l - 1, ::20]
        c = (h1 * h0).sum(1) / (np.linalg.norm(h1, axis=1) * np.linalg.norm(h0, axis=1))
        print(f"  cos(h[{l}],h[{l-1}]) mean {c.mean():.3f}")
