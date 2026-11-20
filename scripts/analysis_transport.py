"""Transport kernel: response of generated token t to a perturbation of prompt position p."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "transport")
OUT = os.path.join(ROOT, "notebook", "figures", "transport")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})


def kernel(tag):
    f = os.path.join(B, tag, "transport.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    dh = z["dh"]                      # [pos, dir, layer, T]
    r = float(z["r"])
    A = np.median(dh[:, :, -1, :], axis=1) / r      # last layer, [pos, T]
    return A, z["positions"], z["layers"]


fig, axs = plt.subplots(1, 2, figsize=(13.4, 5), gridspec_kw={"width_ratios": [1.25, 1]})
A, pos, layers = kernel("deepseek")

ax = axs[0]
im = ax.imshow(np.log10(np.maximum(A, 1e-3)), aspect="auto", origin="lower", cmap="magma",
               extent=[0, A.shape[1], pos[0], pos[-1]])
ax.plot(np.argmax(A, axis=1), pos, "c-", lw=1.2, label="peak generated token")
ax.set_xlabel("generated token $t$"); ax.set_ylabel("perturbed prompt position $p$")
ax.set_title("DeepSeek: response to one prompt position\n(last layer, median over 8 dirs)")
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(r"$\log_{10}\|\delta h\|/r$")
ax.legend(loc="upper left")

ax = axs[1]
for p in [0, 16, 32, 48, 64]:
    i = int(np.argmin(np.abs(pos - p)))
    ax.semilogy(np.arange(A.shape[1]), np.maximum(A[i], 1e-4), lw=1.2, label=f"$p={int(pos[i])}$")
ax.set_xlabel("generated token $t$"); ax.set_ylabel(r"$\|\delta h_{t,L}\|/r$")
ax.set_title("Rows of the kernel")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kernel_deepseek.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

peak = np.argmax(A, axis=1)
json.dump(dict(peak_by_position=dict(zip(pos.tolist(), peak.tolist())),
               max_response=float(A.max())),
          open(os.path.join(ROOT, "results", "transport_deepseek.json"), "w"), indent=1)
print("saved; peak range", int(peak.min()), int(peak.max()))
