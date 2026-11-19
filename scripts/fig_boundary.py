"""P1: the decision boundary at large radius. Margin coordinate u = dm/m, flips at u = -1."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "bigr")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9})

MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Phi-4-mini", "phi4mini", "tab:green")]

fig, axs = plt.subplots(1, 3, figsize=(17, 4.7), sharey=True)
for ax, (name, tag, c) in zip(axs, MODELS):
    d = np.load(os.path.join(B, tag, "p1", "a1.npz"))
    m = d["margins"].astype(np.float64)
    ok = m > 0.02
    u = np.where(ok[None, None, :], d["dm"] / np.where(ok, m, 1.0), np.nan)
    radii = d["radii"]
    for i, r in enumerate(radii):
        y = np.nanmin(u[i], axis=1)               # closest approach per direction
        ax.scatter(np.full_like(y, r), y, s=22, color=c, alpha=0.55, edgecolor="none")
    lin = np.nanmin(u[0], axis=1)[:, None] * (radii / radii[0])[None, :]
    ax.plot(radii, np.nanmedian(lin, axis=0), "k--", lw=1.0, label=r"linear in $r$ (from $r=0.3$)")
    ax.plot(radii, np.nanmedian(np.nanmin(u, axis=2), axis=1), "o-", color=c, lw=1.4, label="measured")
    ax.axhline(-1, color="r", ls="--", lw=1.2, label=r"flip at $u=-1$")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel(r"radius $r$")
    ax.set_title(name)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="lower left", fontsize=8)
axs[0].set_ylabel(r"$\min_t u_t$ per direction")
fig.suptitle(r"Approach to the decision boundary: the coordinate is linear to $r\approx0.3$, "
             r"then saturates (normalisation bounds the response)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "boundary_large_r.png"), dpi=190, bbox_inches="tight")
print("saved")

for name, tag, c in MODELS:
    d = np.load(os.path.join(B, tag, "p1", "a1.npz"))
    m = d["margins"].astype(np.float64)
    ok = m > 0.02
    u = np.where(ok[None, None, :], d["dm"] / np.where(ok, m, 1.0), np.nan)
    print(name, "min u per r", [round(float(np.nanmin(u[i])), 2) for i in range(len(d["radii"]))],
          "frac dirs crossing", [round(float(np.nanmin(u[i], axis=1) < -1), 2)
                                 for i in range(len(d["radii"]))])
