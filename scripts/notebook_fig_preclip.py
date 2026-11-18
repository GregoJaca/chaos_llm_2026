"""No effect of the window before the context reaches it."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\preclip.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(B)
P = int(d["P"]); T = int(d["T"])
pos = np.arange(P + T)

fig, ax = plt.subplots(figsize=(10.5, 4.5))
for W, c in [(50, "tab:red"), (100, "tab:blue"), (200, "tab:green")]:
    ref = d["W4096_l27"]
    cur = d[f"W{W}_l27"]
    rel = np.linalg.norm(cur - ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-9)
    ax.semilogy(pos - P, np.maximum(rel, 1e-12), lw=1.3, color=c, label=fr"$W={W}$")
    ax.axvline(W - P, color=c, ls="--", lw=0.9)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel(r"token index $t$ (negative = prompt, 0 = first generated token)")
ax.set_ylabel(r"relative hidden-state difference vs $W=4096$")
ax.set_title("With identical inputs and no perturbation, nothing changes until $t=W-P$")
ax.set_xlim(-P - 2, T)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_preclip_null.png"), dpi=190)
print("saved")
