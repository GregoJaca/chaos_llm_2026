"""KV-cache kicks (DeepSeek) vs state kicks (Mamba): the symmetric memory
experiment.  Reads results/ctrl_local/ds_kv/kv_kick.npz and the Mamba
A-experiment curves from jac130.npz.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebook", "figures_controls")
KV = os.path.join(ROOT, "results", "ctrl_local", "ds_kv", "kv_kick.npz")
JM = np.load(os.path.join(ROOT, "results", "ctrl_local", "mamba130", "jac130.npz"), allow_pickle=True)

C_DS = "#D55E00"
C_MAM = "#0072B2"

z = np.load(KV, allow_pickle=True)
keys = [k for k in z.files if not k.endswith("_layers") and k not in ("margins", "prompt_len", "T")]

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axs[0]
# DeepSeek KV kick, all-layer, eps=0.1, t0=70/150
for key, ls in [("eps0.1_t070_lall", "-"), ("eps0.1_t0150_lall", "--")]:
    if key in keys:
        dh = np.median(z[key], axis=0)
        ax.plot(np.arange(len(dh)), dh, ls, color=C_DS, lw=1.4, label=f"DeepSeek KV kick {key.split('_')[1]}")
ax.set_yscale("log")
ax.set_xlabel(r"tokens after the cache kick")
ax.set_ylabel(r"$\|\delta h_L(t)\|$ (last layer, median of 8 dirs)")
ax.set_title("(a) DeepSeek KV-cache kicks ($\\varepsilon=0.1$, all layers, live\nwindow): bounded, then decay to zero with the window")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

ax = axs[1]
# Mamba state kicks (A experiment) for comparison
dstate = JM["A_dstate"].item()
for t0 in sorted(dstate.keys(), key=int):
    D = dstate[t0]
    med = np.median(D, axis=0)
    ax.plot(np.arange(len(med)), med, lw=1.4, color=C_MAM, label=f"Mamba state kick $t_0$={t0}")
ax.set_yscale("log")
ax.set_xlabel(r"tokens after the state kick ($\varepsilon=10^{-2}$)")
ax.set_ylabel(r"$\|\delta\mathrm{state}(t)\|$ (all layers, median)")
ax.set_title("(b) Mamba SSM-state kicks (Part 4): the same shape -\nbounded transient, no growth; memory kicks do not explode in either arch")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_kv_kick.png"), dpi=190); plt.close(fig)

# layer attribution figure
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for key in sorted(keys):
    if "_l" not in key or key.endswith("_lall"):
        continue
    if not key.startswith("eps0.1_t070"):
        continue
    dh = np.median(z[key], axis=0)
    lbl = key.split("_l")[1]
    ax.plot(np.arange(len(dh)), dh, lw=1.0, label=f"cache layer {lbl}")
dh = np.median(z["eps0.1_t070_lall"], axis=0)
ax.plot(np.arange(len(dh)), dh, "k-", lw=2.0, label="all layers")
ax.set_yscale("log")
ax.set_xlabel(r"tokens after the cache kick ($\varepsilon=0.1$, $t_0=70$)")
ax.set_ylabel(r"$\|\delta h_L(t)\|$")
ax.set_title("KV-cache layer attribution: single-layer cache kicks decay;\nthe all-layer kick is the sum - no runaway layer")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_kv_layers.png"), dpi=190); plt.close(fig)

for key in sorted(keys):
    dh = np.median(z[key], axis=0)
    print("%-22s dh(0)=%.4g +20=%.4g end=%.4g  peak=%.4g" % (
        key, dh[0], dh[min(20, len(dh) - 1)], dh[-1], dh.max()))
print("wrote", OUT)
