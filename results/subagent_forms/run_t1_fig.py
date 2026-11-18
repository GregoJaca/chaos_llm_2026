"""Task 1 figure: 3 example full-curve fits + winner-count summary."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitlib import fit_all_full

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)

examples = [(0.1, 3, 27, "rising (spike train)"), (0.01, 0, 14, "mild rise + decay"),
            (0.3, 7, 7, "decay-only (shallow layer)")]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.0))
for ax, (r, di, L, tag) in zip(axs[:3], examples):
    ri = int(np.argmin(np.abs(radii - r))); li = int(np.where(layers == L)[0][0])
    y = dh[ri, di, li, :]
    t = np.arange(len(y)); m = (t >= 1) & (y > 0)
    fits = fit_all_full(t[m], y[m])
    ax.semilogy(t[m], y[m], "k.", ms=1.6, label="data")
    tt = t[m]
    f = fits["pw-logt"]
    x = np.log(tt); xs = np.log(f["params"]["tstar"])
    ax.semilogy(tt, np.exp(f["params"]["a0"] + f["params"]["b1"] * np.minimum(x, xs)
                           + f["params"]["b2"] * np.maximum(0, x - xs)), "-", lw=1.3,
                color="tab:red", label=f"pw-logt (BIC {f['bic']:.0f}, R2 {f['r2']:.2f})")
    f = fits["two-exp"]
    ts2 = f["params"]["tstar"]
    ax.semilogy(tt, np.exp(f["params"]["a0"] + f["params"]["lam1"] * np.minimum(tt, ts2)
                           + f["params"]["lam2"] * np.maximum(0, tt - ts2)), "--", lw=1.3,
                color="tab:orange", label=f"two-exp (BIC {f['bic']:.0f}, R2 {f['r2']:.2f})")
    f = fits["exp"]
    ax.semilogy(tt, np.exp(f["params"]["a0"] + f["params"]["lam"] * tt), ":", lw=1.4,
                color="tab:blue", label=f"exp (BIC {f['bic']:.0f}, R2 {f['r2']:.2f})")
    f = fits["lognormal"]
    if f["params"]:
        p = f["params"]
        ax.semilogy(tt, p["A"] * np.exp(-(np.log(tt) - p["mu"]) ** 2 / (2 * p["sigma"] ** 2)),
                    "-.", lw=1.2, color="tab:green", label=f"lognormal (R2 {f['r2']:.2f})")
    f = fits["exp+floor"]
    A, tau, C = f["params"]["A"], f["params"]["tau"], f["params"]["C"]
    ax.semilogy(tt, A * np.exp(-(tt - tt[0]) / tau) + C, "-", lw=1.1, color="tab:purple",
                alpha=.8, label=f"exp+floor (R2 {f['r2']:.2f})")
    ax.set_xlabel("t (tokens)"); ax.set_title(f"r={r:g} dir={di} L={L}: {tag}", fontsize=9)
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=6.5)
axs[0].set_ylabel(r"$a(t)=\|\delta h_t\|$")

summ = json.load(open(os.path.join(OUT, "t1_summary.json")))
MODELS = ["exp", "power", "exp+floor", "lognormal", "pw-logt", "two-exp"]
ax = axs[3]
xpos = np.arange(len(MODELS))
vals = [summ["winners"][mm]["bic"] for mm in MODELS]
ax.bar(xpos, vals, color=["tab:blue", "tab:cyan", "tab:purple", "tab:green", "tab:red", "tab:orange"])
for x, v in zip(xpos, vals):
    ax.text(x, v + 5, str(v), ha="center", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(MODELS, rotation=25, fontsize=8)
ax.set_ylabel("BIC winner count (of 700)")
ax.set_title("piecewise forms win; smooth forms never", fontsize=9)
ax.grid(alpha=.3, axis="y")
fig.suptitle("Task 1: functional-form fits to $a(t)$, r$\\leq$0.3, 20 dirs, layers {3,7,14,21,27}, w100")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_fits_examples.png"), dpi=160)
print("saved fig_fits_examples.png")
