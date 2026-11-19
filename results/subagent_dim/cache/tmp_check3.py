import numpy as np
ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
for tag, Lexp in [("deepseek", 28), ("qwen3_4b", 36)]:
    a = np.load(ROOT + r"\dim" + "\\" + tag + r"\acts.npz")
    acts = a["acts"]
    h_last = acts[0, Lexp, ::50].astype(np.float32)
    h_prev = acts[0, Lexp - 1, ::50].astype(np.float32)
    # ratio h_last/h_prev per channel for several positions: is it the same vector
    # (=> h_last = diag(g) h_prev / rms, i.e. RMSNorm of the SAME stream)?
    cors = []
    n = h_last.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            r1 = h_last[i] / np.maximum(np.abs(h_prev[i]), 1e-3)
            r2 = h_last[j] / np.maximum(np.abs(h_prev[j]), 1e-3)
            cors.append(np.corrcoef(r1, r2)[0, 1])
    print(tag, "corr of channel ratios across position pairs: mean", round(float(np.mean(cors)), 3),
          "min", round(float(np.min(cors)), 3))
    # also compare to ratios between two raw block states (control)
    hA = acts[0, 20, ::50].astype(np.float32)
    hB = acts[0, 19, ::50].astype(np.float32)
    cors0 = []
    for i in range(n):
        for j in range(i + 1, n):
            r1 = hA[i] / np.maximum(np.abs(hB[i]), 1e-3)
            r2 = hA[j] / np.maximum(np.abs(hB[j]), 1e-3)
            cors0.append(np.corrcoef(r1, r2)[0, 1])
    print(tag, "control (block20/block19 ratios): mean", round(float(np.mean(cors0)), 3))
