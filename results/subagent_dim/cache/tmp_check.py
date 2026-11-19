import numpy as np
ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
for name, p in [
    ("big/deepseek", ROOT + r"\big\all_layers\a1.npz"),
    ("w100/deepseek", ROOT + r"\w100_layers\a1.npz"),
    ("q3_4b", ROOT + r"\models\qwen3_4b\a1.npz"),
    ("phi", ROOT + r"\models\phi\a1.npz"),
    ("qwen35", ROOT + r"\models\qwen35\a1.npz"),
]:
    d = np.load(p)
    nh0 = d["norm_h0"]
    lay = d["layers"]
    print(
        name, "norm_h0", nh0.shape,
        "row0[:3]", np.round(nh0[0, :3], 2),
        "rowmid[:3]", np.round(nh0[nh0.shape[0] // 2, :3], 2),
        "rowlast[:3]", np.round(nh0[-1, :3], 2),
        "layers[:3]", lay[:3], "layers[-3:]", lay[-3:],
    )
a = np.load(ROOT + r"\dim\deepseek\acts.npz"); n = a["norms"]
print("deepseek acts mean norms L0..L5:", np.round(n.mean(axis=(0, 2))[:6], 2), " L27,L28:", np.round(n.mean(axis=(0, 2))[27:29], 2))
a2 = np.load(ROOT + r"\dim\qwen3_4b\acts.npz"); n2 = a2["norms"]
print("qwen acts mean norms L0..L5:", np.round(n2.mean(axis=(0, 2))[:6], 2), " L35,L36:", np.round(n2.mean(axis=(0, 2))[35:37], 2))
