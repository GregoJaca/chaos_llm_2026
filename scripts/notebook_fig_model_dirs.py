"""Direction-gain spread: DeepSeek vs Qwen3-4B."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
QW = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models\qwen3_4b\a1.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
for ax, (path, name, layer) in zip(axs, [(DS, "DeepSeek-1.5B", 27), (QW, "Qwen3-4B", 32)]):
    d = np.load(path, allow_pickle=True)
    L = list(d["layers"]); li = L.index(layer)
    radii = d["radii"]; ri = int(np.argmin(np.abs(radii - 0.1)))
    g = np.median(d["dh_norm"][ri, :, li, :50] / radii[ri], axis=1)
    g = g[g > 0]
    ax.hist(np.log10(g), bins=18, color="tab:blue")
    ax.set_xlabel(r"$\log_{10}$ gain per direction (median over 50 tokens)")
    ax.set_ylabel("directions")
    ax.set_title(f"{name}, layer {layer}, $r=0.1$\n"
                 f"spread ×{np.percentile(g,95)/np.percentile(g,5):.2f} (5–95%)")
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_direction_spread.png"), dpi=190)
print("saved")
