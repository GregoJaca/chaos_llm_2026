"""Cross-model comparison normalised by the state norm."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
RUNS = [
    ("DeepSeek-1.5B", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz", 27),
    ("Qwen3-4B", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models\qwen3_4b\a1.npz", 32),
    ("Phi-4-mini", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models\phi\a1.npz", 28),
    ("Qwen3.5-2B", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models\qwen35\a1.npz", 23),
]

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.5))
for name, path, layer in RUNS:
    d = np.load(path, allow_pickle=True)
    L = list(d["layers"]); li = L.index(layer) if layer in L else len(L) - 1
    radii = d["radii"]; ri = int(np.argmin(np.abs(radii - 0.1)))
    a = np.median(d["dh_norm"][ri, :, li, :], axis=0)
    h = d["norm_h0"][li]
    q = a / np.maximum(h, 1e-9)
    axs[0].semilogy(np.arange(len(q)), np.maximum(q, 1e-9), lw=1.2, label=name)
    g = np.median(d["dh_norm"][ri, :, li, :50] / radii[ri], axis=1)
    g = g[g > 0]
    axs[1].hist(np.log10(g), bins=16, alpha=0.45, label=name)
axs[0].set_xlabel(r"$t$")
axs[0].set_ylabel(r"$a(t,\ell)/\|h_{t,\ell}\|$")
axs[0].set_title("(a) Relative response at the last layer")
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
axs[1].set_xlabel(r"$\log_{10}$ gain per direction")
axs[1].set_ylabel("directions")
axs[1].set_title("(b) Direction spread")
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_comparison_rel.png"), dpi=190)
print("saved")
for name, path, layer in RUNS:
    d = np.load(path, allow_pickle=True); L = list(d["layers"]); li = L.index(layer) if layer in L else len(L) - 1
    radii = d["radii"]; ri = int(np.argmin(np.abs(radii - 0.1)))
    a = np.median(d["dh_norm"][ri, :, li, :], axis=0); h = d["norm_h0"][li]
    print(name, "q(t=10)=%.3e q(t=50)=%.3e" % (a[10] / h[10], a[50] / h[50]))
