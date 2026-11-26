import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = {"DeepSeek-1.5B": "results/local_data/w100_layers/a1.npz",
          "Qwen2.5-3B": "results/update/fp32_qwen25_a1.npz",
          "Qwen3.5-2B": "results/update/fp32_qwen35_a1.npz"}
fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
ax = axes[0, 0]
for n, p in MODELS.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    r = z["radii"].astype(float)
    y = np.median(z["dh_norm"][:, :, -1, 50], axis=1)
    ax.plot(r, y, "o-", ms=4, color=COL[n], label=n)
ax.plot(np.logspace(-4, 0.3, 40), np.logspace(-4, 0.3, 40), "k:", lw=1.0, label=r"$y=r$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\|\delta h_{t=50,L}\|$")
ax.set_title("(a) fp32 response vs radius", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
for n, p in MODELS.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    r = z["radii"].astype(float)
    y = np.median(z["dh_norm"][:, :, -1, 50], axis=1)
    beta = np.diff(np.log(y)) / np.diff(np.log(r))
    ax.plot(np.sqrt(r[:-1] * r[1:]), beta, "o-", ms=4, color=COL[n], label=n)
ax.axhline(1, color="k", ls=":", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\beta=\mathrm{d}\log\|\delta h\|/\mathrm{d}\log r$")
ax.set_title(r"(b) $\beta\to1$ at small $r$", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)

pts = []
for root in ["deepseek_r0.003", "deepseek_r0.01", "deepseek_r0.03", "deepseek",
             "deepseek_p2", "deepseek_p3", "deepseek_r003"]:
    p = os.path.join(HERE, "results/superposition", root, "superposition.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    if "err_L27" not in z.files:
        continue
    r = float(z["r"]); P = int(z["prompt_len"])
    pts.append((r * np.sqrt(P), np.nanmedian(np.abs(z["err_L27"])),
                np.nanmedian(z["cos_L27"]), root))
pts.sort()
ax = axes[1, 0]
ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#1f77b4")
ax.set_xscale("log")
ax.set_xlabel(r"total norm $r\sqrt{P}$")
ax.set_ylabel(r"$\|A-S\|/\|A\|$")
ax.set_title("(c) Superposition error", loc="left")
ax.grid(True, which="both", alpha=0.3)
ax = axes[1, 1]
ax.plot([p[0] for p in pts], [p[2] for p in pts], "o-", color="#d62728")
ax.set_xscale("log")
ax.set_xlabel(r"total norm $r\sqrt{P}$")
ax.set_ylabel(r"$\cos(A,S)$")
ax.set_title("(d) The summed response has the same direction", loc="left")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_response_vs_radius.png")
print("done")

