"""Does the attention window change the dynamical regime?"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_sweep"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
WINS = [50, 100, 200, 400, 800, 4096]
R = 0.1
P = 63

curves = {}
for w in WINS:
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27)
    ri = int(np.argmin(np.abs(d["radii"] - R)))
    a = np.maximum(np.median(d["dh_norm"][ri, :, li, :], axis=0), 1e-12)
    curves[w] = a

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})

ax = axs[0]
for w, c in zip([50, 100, 200, 400], ["tab:red", "tab:blue", "tab:green", "tab:orange"]):
    ax.semilogy(np.arange(300), curves[w], lw=1.2, color=c, label=f"$W={w}$")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t,27)=\|\delta h_{t,27}\|_2$")
ax.set_title("(a) $W=50<P$: no rise at all.\n$W>P$: rise, then decay.")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axs[1]
tt = np.arange(1, 300)
for w, c in zip([50, 100], ["tab:red", "tab:blue"]):
    a = curves[w]
    lam_t = np.log(a[1:] / a[1]) / tt
    ax.semilogx(tt, lam_t, lw=1.4, color=c, label=f"$W={w}$")
ax.text(0.35, 0.55, "for $W\\geq100$ the curves are\nidentical up to $t=W-P$", transform=ax.transAxes, fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"horizon $t$ (log)")
ax.set_ylabel(r"finite-time rate $\lambda(t)=\frac{1}{t}\log\frac{a(t)}{a(1)}$")
ax.set_title("(b) The rate is positive early, crosses zero,\nand stays negative: no sustained stretching")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

ax = axs[2]
early, late = [], []
for w in WINS:
    a = curves[w]
    early.append(np.log(a[20] / a[1]) / 19)
    late.append(np.log(a[299] / a[200]) / 99)
x = np.arange(len(WINS))
ax.bar(x - 0.2, early, 0.4, label="rate over $t\\in[1,20]$", color="tab:green")
ax.bar(x + 0.2, late, 0.4, label="rate over $t\\in[200,299]$", color="tab:red")
ax.set_xticks(x); ax.set_xticklabels([f"{w}" for w in WINS])
ax.set_xlabel(r"window $W$")
ax.set_ylabel(r"finite-time rate $\lambda$ (nats/token)")
ax.set_title("(c) Early growth only for $W>P$;\nlate decay for every $W$")
ax.axhline(0, color="k", lw=0.8)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_regimes.png"), dpi=190)
print("saved")
for w in WINS:
    a = curves[w]
    print(f"W={w}: a0={a[0]:.3f} a20={a[20]:.3f} a299={a[-1]:.4f} early={early[WINS.index(w)]:+.4f} late={late[WINS.index(w)]:+.4f}")
