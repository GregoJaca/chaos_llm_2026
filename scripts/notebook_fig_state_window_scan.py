"""State perturbation under a window scan: dose-response across W."""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\big\state"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

runs = {}
for p in glob.glob(os.path.join(B, "w*", "state2.npz")):
    w = int(os.path.basename(os.path.dirname(p))[1:])
    runs[w] = np.load(p, allow_pickle=True)

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.6))

ax = axs[0]
for w, c in [(50, "tab:red"), (70, "tab:orange"), (90, "tab:green"), (4096, "0.4")]:
    d = runs[w]
    steps = list(d["steps"]); radii = d["radii"]; layers = list(d["layers"])
    ri = int(np.argmin(np.abs(radii - 0.1))); li = layers.index(27)
    si = steps.index(16)
    a = np.median(d["dh"][ri, si, :, li, :], axis=0)
    ax.semilogy(np.arange(300 - 16), np.maximum(a[16:], 1e-12), lw=1.3, color=c, label=f"$W={w}$")
ax.set_xlim(0, 120)
ax.set_ylim(0.5, 30)
ax.set_xlabel(r"lag $\tau=t-s$")
ax.set_ylabel(r"$a(s+\tau,27)$")
ax.set_title("(a) Decay of a single-position perturbation ($s=16$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

wins, imm, half = [], [], []
for w, d in sorted(runs.items()):
    steps = list(d["steps"]); radii = d["radii"]; layers = list(d["layers"])
    ri = int(np.argmin(np.abs(radii - 0.1))); li = layers.index(27)
    si = steps.index(16)
    a = np.median(d["dh"][ri, si, :, li, :], axis=0)
    seg = a[16:]
    below = np.where(seg < 0.5 * seg[0])[0]
    wins.append(w); imm.append(float(a[16])); half.append(int(below[0]) if len(below) else -1)

ax = axs[1]
ax.semilogx(wins, imm, "o-", color="tab:blue", label="immediate response $a(16,27)$")
ax.set_xlabel(r"window $W$")
ax.set_ylabel(r"$a(16,27)$", color="tab:blue")
ax.grid(alpha=0.25, which="both")
ax2 = ax.twinx()
ax2.semilogx(wins, half, "s--", color="tab:red", label="half-life (tokens)")
ax2.set_ylabel("half-life (tokens)", color="tab:red")
ax.set_title("(b) Dose–response: the effect saturates for $W\\gtrsim90$")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "state_window_scan.png"), dpi=190)
print("saved", list(zip(wins, [round(x, 1) for x in imm], half)))
