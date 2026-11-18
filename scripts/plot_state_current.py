"""Current-state perturbation: a kick at one generated position, then watch it
propagate forward in time (t) and depth (layer).  r-scan and 8 injection steps.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\state_deep_emb"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

d = np.load(os.path.join(BASE, "state2.npz"), allow_pickle=True)
steps = d["steps"]; radii = d["radii"]; layers = list(d["layers"])
dh = d["dh"]; cs = d["cos"]; dm = d["dm"]; margins = d["margins"]
li = layers.index(27)
T = dh.shape[-1]
t = np.arange(T)

# pick a radius in the linear range
RIDX = int(np.argmin(np.abs(radii - 0.03)))
print("radius used", radii[RIDX])

med = np.median(dh[RIDX, :, :, li, :], axis=1)   # [n_s, T]

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.26)

# (a) field
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(np.log10(np.maximum(med, 1e-6)), aspect="auto", origin="lower",
               extent=(0, T, -0.5, len(steps) - 0.5), cmap="magma")
ax.set_yticks(range(len(steps))); ax.set_yticklabels([str(s) for s in steps])
ax.set_xlabel("generated token t (when the response is measured)")
ax.set_ylabel("step s at which the kick is applied")
ax.set_title(f"(a) Where and when a one-token kick acts (r={radii[RIDX]:g}, last layer)")
cb = plt.colorbar(im, ax=ax); cb.set_label("log10 response size")
ax.axline((0, 0), slope=1, color="cyan", lw=1, ls="--")
ax.text(200, len(steps) - 1.6, "no effect before the kick", color="cyan", fontsize=8)

# (b) decay with lag
ax = fig.add_subplot(gs[0, 1])
for si, s in enumerate(steps):
    lag = t - s
    ok = lag >= 0
    ax.semilogy(lag[ok], np.maximum(med[si, ok], 1e-6), lw=1.1, label=f"kick at s={s}")
ax.set_xlabel("lag: tokens after the kick (t - s)")
ax.set_ylabel("response size (last layer)")
ax.set_title("(b) The response decays with lag, but the decay rate\nand the initial size depend strongly on where the kick lands")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25)

# (c) immediate size and half-life vs s
ax = fig.add_subplot(gs[1, 0])
imm = med[np.arange(len(steps)), steps]
half = []
for si, s in enumerate(steps):
    seg = med[si, s:]
    peak = seg[0]
    below = np.where(seg < 0.5 * peak)[0]
    half.append(int(below[0]) if len(below) else T - s)
ax.bar(np.arange(len(steps)) - 0.2, imm, width=0.4, color="tab:blue", label="response right after the kick, t=s")
ax.set_xticks(range(len(steps))); ax.set_xticklabels([str(s) for s in steps])
ax.set_xlabel("kick step s")
ax.set_ylabel("response size at t=s")
ax.set_title("(c) Where the kick lands matters more than how big it is\n(13x spread across positions)")
ax.grid(alpha=0.25, axis="y")
ax2 = ax.twinx()
ax2.plot(np.arange(len(steps)), half, "o-", color="tab:red", lw=1.2, label="tokens until the response halves")
ax2.set_ylabel("tokens until the response falls to half", color="tab:red")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

# (d) linearity in r
ax = fig.add_subplot(gs[1, 1])
slopes = []
for si, s in enumerate(steps):
    vals = [np.median(dh[ri, si, :, li, s]) for ri in range(len(radii))]
    ax.loglog(radii, vals, "o-", lw=1.1, label=f"s={s}")
    p = np.polyfit(np.log(radii), np.log(vals), 1)[0]
    slopes.append(p)
ax.set_xlabel("kick size r (norm added to one token's embedding)")
ax.set_ylabel("response size right after the kick")
ax.set_title("(d) Linear in r: slope 1 at every position\n(fitted slopes %.2f-%.2f)" % (min(slopes), max(slopes)))
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=7, ncol=2)

fig.suptitle("Current-state (single-token) perturbation: immediate response, memory, and linearity", fontsize=12)
fig.savefig(os.path.join(OUT, "state_current_field.png"), dpi=190, bbox_inches="tight")
print("saved")
print("immediate response by s:", [round(float(v), 3) for v in imm])
print("half-life by s:", half)
print("slopes:", [round(float(x), 3) for x in slopes])
print("cos at t=s (L27):", [round(float(np.median(cs[RIDX, si, :, li, s])), 4) for si, s in enumerate(steps)])
