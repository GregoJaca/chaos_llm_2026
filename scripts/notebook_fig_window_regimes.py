"""Window regimes on two prompts."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
R = 0.1

PROMPTS = [("ocean currents", os.path.join(BASE, "window_sweep")),
           ("cybersecurity", os.path.join(BASE, "window_sweep_cyber"))]
WINS = [50, 100, 200]
COLS = {50: "tab:red", 100: "tab:blue", 200: "tab:green", 4096: "0.5"}

fig, axs = plt.subplots(2, 2, figsize=(13, 8.2), sharex="col")

for row, (name, folder) in enumerate(PROMPTS):
    curves = {}
    P = None
    for w in WINS:
        d = np.load(os.path.join(folder, f"w{w}", "a1.npz"), allow_pickle=True)
        P = int(d["prompt_ids"].shape[0])
        L = list(d["layers"]); li = L.index(27)
        ri = int(np.argmin(np.abs(d["radii"] - R)))
        curves[w] = np.maximum(np.median(d["dh_norm"][ri, :, li, :], axis=0), 1e-12)

    ax = axs[row, 0]
    for w in WINS:
        ax.semilogy(np.arange(300), curves[w], lw=1.2, color=COLS[w], label=f"$W={w}$")
    ax.axvline(P, color="k", ls=":", lw=0.8)
    ax.text(P + 2, ax.get_ylim()[0] * 2, f"$W=P$\n({P})", fontsize=8)
    ax.set_ylabel(r"$a(t,27)=\|\delta h_{t,27}\|_2$")
    ax.set_title(f"({chr(97 + 2*row)}) {name} ($P={P}$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axs[row, 1]
    tt = np.arange(1, 300)
    for w in WINS:
        a = curves[w]
        ax.semilogx(tt, np.log(a[1:] / a[1]) / tt, lw=1.2, color=COLS[w], label=f"$W={w}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$\lambda(t)=\frac{1}{t}\log\frac{a(t)}{a(1)}$")
    ax.set_title(f"({chr(98 + 2*row)}) {name}: finite-time rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

for ax in axs[1]:
    ax.set_xlabel(r"generated token $t$")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_regimes.png"), dpi=190)
print("saved")
