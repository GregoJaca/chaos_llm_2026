"""Clean depth + latency figures for the notebook (single-position perturbation)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(os.path.join(B, "state_deep_emb", "state2.npz"), allow_pickle=True)
a1 = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)
steps = d["steps"]; radii = d["radii"]; layers = list(d["layers"])
dh = d["dh"]; dm = d["dm"]; margins = d["margins"]
norm = a1["norm_h0"]; a1layers = list(a1["layers"])
RIDX = int(np.argmin(np.abs(radii - 0.03)))
T = dh.shape[-1]

# ---------- depth: absolute vs relative ----------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
for s in [16, 50, 100, 200]:
    si = list(steps).index(s)
    ab = [np.median(dh[RIDX, si, :, li, s]) for li in range(len(layers))]
    rel = [ab[li] / norm[a1layers.index(layers[li]), s] for li in range(len(layers))]
    axs[0].semilogy(layers, ab, "o-", ms=4, label=f"$s={s}$")
    axs[1].semilogy(layers, rel, "o-", ms=4, label=f"$s={s}$")
axs[0].set_ylabel(r"$a(s,\ell)=\|\delta h_{s,\ell}\|_2$  (absolute)")
axs[0].set_title("(a) Absolute size grows with depth")
axs[1].set_ylabel(r"$q(s,\ell)=a(s,\ell)/\|h_{s,\ell}\|_2$  (relative to the state length)")
axs[1].set_title("(b) Relative size shrinks with depth,\nexcept at the sensitive position $s=16$")
for ax in axs:
    ax.set_xlabel(r"layer $\ell$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "state_depth_two_views.png"), dpi=190)
plt.close(fig)

# ---------- latency timeline ----------
ri = int(np.argmin(np.abs(radii - 0.1)))
events = []
for si, s in enumerate(steps):
    hit = dm[ri, si] <= -margins[None, :]
    for di in range(hit.shape[0]):
        if hit[di].any():
            events.append((int(s), int(hit[di].argmax()), int(di)))

fig, ax = plt.subplots(figsize=(10.5, 3.8))
for k, (s, t, di) in enumerate(events):
    ax.plot([s, t], [k, k], "-", color="tab:red", lw=2.2, zorder=4)
    ax.plot([s], [k], "o", color="tab:red", ms=7, zorder=5)
    ax.plot([t], [k], "X", color="tab:red", ms=9, zorder=5)
    ax.text(s - 4, k + 0.18, f"$s={s}$, dir {di}", fontsize=8, color="tab:red", ha="right")
    ax.text(t + 4, k + 0.18, f"changes token at $t={t}$ ({t-s} later)", fontsize=8, color="tab:red")
ax.set_yticks([])
ax.set_ylim(-0.6, len(events) - 0.2)
ax.set_xlim(0, T)
ax.set_xlabel(r"generated token $t$")
ax.set_title(r"Perturbation at one position ($r=0.1$): when it changes the output")
ax.grid(alpha=0.25, axis="x")
ax2 = ax.twinx()
ax2.semilogy(np.arange(T), np.maximum(margins, 1e-4), color="0.75", lw=0.8, zorder=0)
ax2.set_ylabel("baseline gap $m_t$ (grey, log)", color="0.45", fontsize=8)
ax2.set_zorder(0); ax.set_zorder(1); ax.patch.set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "state_latency_timeline.png"), dpi=190)
print("saved; events", events)
