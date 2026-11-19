"""Cross-prompt depth gain: 6 prompts x 28 layers."""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\big\all_layers"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

curves = []
for p in sorted(glob.glob(os.path.join(B, "p*", "a1.npz"))):
    d = np.load(p, allow_pickle=True)
    ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    g = np.median(d["dh_norm"][ri, :, :, :50] / d["radii"][ri], axis=(0, 2))
    curves.append(g)
C = np.stack(curves)                      # [prompts, layers]
C = C[:, 1:]                              # drop layer 0 (response is zero there)
layers = np.arange(C.shape[1]) + 1

fig, ax = plt.subplots(figsize=(9, 4.8))
for i, c in enumerate(C):
    ax.semilogy(layers, np.maximum(c, 1e-8), lw=0.9, color="0.7")
ax.semilogy(layers, np.median(C, axis=0), lw=2.2, color="tab:blue", label="median over 6 prompts")
ax.fill_between(layers, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                color="tab:blue", alpha=0.18, label="25-75%")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"median $a/r$ over the first 50 tokens")
ax.set_title("Depth gain is the same shape for six prompts\n(only the overall scale differs)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_gain_6prompts.png"), dpi=190)
print("saved")
print("L1, L7, L14, L21, L27 medians:",
      [round(float(np.median(C[:, i])), 3) for i in [0, 6, 13, 20, 26]])
print("spread L27 across prompts: x%.2f" % (C[:, 26].max() / C[:, 26].min()))
