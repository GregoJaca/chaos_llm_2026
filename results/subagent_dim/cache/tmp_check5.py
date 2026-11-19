import numpy as np
ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
tag = "deepseek"
v = np.load(ROOT + r"\dim" + "\\" + tag + r"\vocab.npz")
a = np.load(ROOT + r"\dim" + "\\" + tag + r"\acts.npz")
lg = np.load(ROOT + r"\dim" + "\\" + tag + r"\logits.npz")
GW = v["GW"].astype(np.float64)
Lg = lg["logits"][:100].astype(np.float32)
rn = np.linalg.norm(Lg, axis=1)  # recorded row norms
h_top = a["acts"][0, 28, :100].astype(np.float64)   # story B: post-norm state
h_prev = a["acts"][0, 27, :100].astype(np.float64)
for name, h in [("h28 as postnorm (B)", h_top), ("h27 raw (A-ish control)", h_prev)]:
    for shift in [0, 1]:
        hh = h[: 100 - shift] if shift == 0 else h[shift:]
        rr = rn[shift:] if shift else rn
        # predicted ||W_U h||^2 = h^T GW h
        pred = np.einsum("ti,ij,tj->t", hh, GW, hh)
        c = np.corrcoef(pred, rr**2)[0, 1]
        ratio = (rr**2).mean() / pred.mean()
        print(f"{name} shift={shift}: corr(pred, ||logits||^2) = {c:.4f}, mean ratio = {ratio:.3g}")
# also: does a plain scale fix it? story B + scale s: s^2 * pred = ||logits||^2
pred = np.einsum("ti,ij,tj->t", h_top, GW, h_top)
s2 = (rn**2).mean() / pred.mean()
print("story B implied scale s =", np.sqrt(s2), " std of s per row:", (rn**2 / pred).std() / (rn**2 / pred).mean())
