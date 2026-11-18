"""Current-state perturbation: depth propagation and effect on the generated text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\state_deep_emb"
A1 = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

d = np.load(os.path.join(BASE, "state2.npz"), allow_pickle=True)
a1 = np.load(A1, allow_pickle=True)
steps = d["steps"]; radii = d["radii"]; layers = list(d["layers"])
dh, cs, dm, margins = d["dh"], d["cos"], d["dm"], d["margins"]
norm = a1["norm_h0"]; a1layers = list(a1["layers"])
P = 65
T = dh.shape[-1]
RIDX = int(np.argmin(np.abs(radii - 0.03)))

fig = plt.figure(figsize=(14, 8.8))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.27)

# (a) response vs layer at the kick and 10 tokens later
ax = fig.add_subplot(gs[0, 0])
for s in [16, 50, 100, 200]:
    si = list(steps).index(s)
    for lag, ls in [(0, "-"), (10, "--")]:
        t = s + lag
        if t >= T:
            continue
        vals = [np.median(dh[RIDX, si, :, li, t]) for li in range(len(layers))]
        ax.semilogy(layers, vals, ls, marker="o", ms=3,
                    label=f"s={s}, lag={lag}")
ax.set_xlabel("layer where the response is measured")
ax.set_ylabel("response size")
ax.set_title("(a) A one-token kick is amplified into the last layers")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25)

# (b) same, but divided by the baseline state length at that layer/token
ax = fig.add_subplot(gs[0, 1])
for s in [16, 50, 100, 200]:
    si = list(steps).index(s)
    vals = []
    for li, l in enumerate(layers):
        ai = a1layers.index(l)
        vals.append(np.median(dh[RIDX, si, :, li, s]) / norm[ai, s])
    ax.semilogy(layers, vals, "o-", ms=3, label=f"kick at s={s}")
ax.set_xlabel("layer")
ax.set_ylabel("response size / baseline state length at that layer")
ax.set_title("(b) Relative to the model's own scale, the kick shrinks with depth\n(the same 'radial scale' effect as for prompt perturbations)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# (c) would the kick change the next token?
ax = fig.add_subplot(gs[1, 0])
for ri, r in enumerate(radii):
    frac = []
    for si, s in enumerate(steps):
        hit = (dm[ri, si] <= -margins[None, :]).any(axis=1)
        frac.append(hit.mean())
    ax.plot(range(len(steps)), frac, "o-", lw=1.2, label=f"r={r:g}")
ax.set_xticks(range(len(steps))); ax.set_xticklabels([str(s) for s in steps])
ax.set_xlabel("kick step s")
ax.set_ylabel("fraction of the 24 directions that\nchange a later top-1 token")
ax.set_title("(c) A single-token kick can change the text,\nbut only for larger r and at sensitive positions")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.set_ylim(0, 1.02)

# (d) how long a kick stays latent before it changes a token
ax = fig.add_subplot(gs[1, 1])
ri = int(np.argmin(np.abs(radii - 0.1)))
events = []
for si, s in enumerate(steps):
    hit = dm[ri, si] <= -margins[None, :]
    for di in range(hit.shape[0]):
        if hit[di].any():
            tch = int(hit[di].argmax())
            events.append((s, tch, margins[tch]))
ax.plot([0, T], [0, T], color="grey", lw=0.8, ls=":")
labels = []
for k, (s, tch, mg) in enumerate(events):
    ax.plot([s, tch], [k, k], "-", color="tab:red", lw=2, zorder=5)
    ax.plot([s], [k], "o", color="tab:red", ms=6, zorder=6)
    ax.plot([tch], [k], "X", color="tab:red", ms=8, zorder=6)
    ax.text(s - 4, k + 0.16, f"kick at s={s}", fontsize=8, color="tab:red", ha="right")
    ax.text(tch + 4, k + 0.16, f"token changes at t={tch}\n({tch - s} tokens later)", fontsize=8, color="tab:red")
ax.set_yticks([])
ax.set_xlim(0, T)
ax.set_ylim(-0.6, max(1.4, len(events) - 0.2))
ax.set_xlabel("generated token t")
ax.set_title("(d) The kick can sit latent for ~100 tokens\n(r=0.1; every direction that eventually changes a token)")
ax.grid(alpha=0.25, axis="x")
ax2 = ax.twinx()
ax2.semilogy(np.arange(T), np.maximum(margins, 1e-4), color="lightgrey", lw=0.8, zorder=0)
ax2.set_ylabel("baseline top1-top2 gap (grey, log)", color="grey", fontsize=8)
ax2.set_zorder(0)
ax.set_zorder(1)
ax.patch.set_visible(False)

fig.suptitle("Current-state perturbation: amplification with depth and effect on the generated text", fontsize=12)
fig.savefig(os.path.join(OUT, "state_current_depth.png"), dpi=190, bbox_inches="tight")
print("saved")
print("fraction flipping by s (r=0.1):", [round(float((dm[3, si] <= -margins[None, :]).any(axis=1).mean()), 2) for si in range(len(steps))])
