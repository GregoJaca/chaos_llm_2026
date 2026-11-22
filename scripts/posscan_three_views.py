"""Three ways to visualise the same position-scan data (so the user can pick one).

Data: probe2/<model>/pos<s>/layer_probe.npz
  dh[inj_block, direction, block, generated_token] = ||dh||
  perturbation: one vector of norm r at generated position s, injected at block `inj_block`.
All panels use injection block 0 and the LAST block as the readout.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "probe2")
OUT = os.path.join(ROOT, "notebook", "figures", "positions")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

MODELS = [("DeepSeek-1.5B", "deepseek", 0.1, "tab:blue"),
          ("Phi-4-mini", "phi4mini", 0.3, "tab:green")]


def load_grid(tag):
    """Return (positions, M) with M[s, tau] = median over directions of ||dh_{s+tau,L}||/r."""
    pos, rows = [], []
    d0 = os.path.join(B, tag)
    if not os.path.isdir(d0):
        return None, None
    for sub in sorted(os.listdir(d0)):
        if not sub.startswith("pos"):
            continue
        f = os.path.join(d0, sub, "layer_probe.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        p = int(z["pos"]); r = float(z["r"])
        dh = z["dh"][0][:, -1, :]                 # [dir, T]
        tau = np.arange(1, 51)
        v = dh[:, p + 1:p + 51].max(axis=1) / r   # per-direction max over 50 lags
        rows.append((v, dh[:, p:p + 51].max(axis=1) / r))
        pos.append(p)
    if not pos:
        return None, None
    o = np.argsort(pos)
    return np.array(pos)[o], [rows[i] for i in o]


# ---------------------------------------------------------------- A. heatmap (s x lag)
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
for ax, (name, tag, r, c) in zip(axs, MODELS):
    pos, rows = load_grid(tag)
    if pos is None:
        ax.axis("off"); continue
    M = np.zeros((len(pos), 50))
    for i, sub in enumerate(sorted(os.listdir(os.path.join(B, tag)))):
        f = os.path.join(os.path.join(B, tag), sub, "layer_probe.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); p = int(z["pos"]); rr = float(z["r"])
        M[np.where(pos == p)[0][0]] = np.median(z["dh"][0][:, -1, p + 1:p + 51], axis=0) / rr
    im = ax.imshow(np.log10(M), aspect="auto", origin="lower", cmap="magma",
                   extent=[1, 50, pos[0], pos[-1]])
    ax.set_xlabel("lag $\\tau$ after the perturbation")
    ax.set_ylabel("position $s$ of the perturbation")
    ax.set_title(f"{name}, $r={r}$, injected at block 0, read at block $L$")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(r"$\log_{10}\|\delta h_{s+\tau,L}\|/r$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "posscan_A_heatmap.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- B. violins
fig, ax = plt.subplots(figsize=(11, 4.8))
width = 0.36
for k, (name, tag, r, c) in enumerate(MODELS):
    pos, rows = load_grid(tag)
    if pos is None:
        continue
    data = [rows[i][0] for i in range(len(pos))]
    parts = ax.violinplot(data, positions=pos + (k - 0.5) * width * 3, widths=width * 2.6,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(c); pc.set_alpha(0.5)
    parts["cmedians"].set_color(c)
ax.set_yscale("log")
ax.set_xlabel("position $s$ of the perturbation")
ax.set_ylabel(r"per-direction $\max_{1\leq\tau\leq50}\|\delta h_{s+\tau,L}\|/r$")
ax.set_title("Distribution over 8 directions, per position (violin = full distribution)")
ax.grid(alpha=0.25, which="both", axis="y")
handles = [plt.Line2D([], [], color=c, lw=3, label=n) for n, _, _, c in MODELS]
ax.legend(handles=handles)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "posscan_B_violin.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- C. trajectories vs lag
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
for ax, (name, tag, r, c) in zip(axs, MODELS):
    pos, rows = load_grid(tag)
    if pos is None:
        ax.axis("off"); continue
    cmap = plt.get_cmap("viridis")
    for i, p in enumerate(pos):
        f = os.path.join(B, tag, f"pos{p}", "layer_probe.npz")
        z = np.load(f); rr = float(z["r"])
        curve = np.median(z["dh"][0][:, -1, p + 1:p + 51], axis=0) / rr
        ax.semilogy(np.arange(1, 51), np.maximum(curve, 1e-3), lw=1.1,
                    color=cmap(i / max(len(pos) - 1, 1)), alpha=0.9)
    ax.set_xlabel("lag $\\tau$ after the perturbation")
    ax.set_title(f"{name}, $r={r}$; colour = position $s$")
    ax.grid(alpha=0.25, which="both")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(pos[0], pos[-1]))
    cb = fig.colorbar(sm, ax=ax, pad=0.01); cb.set_label("position $s$")
axs[0].set_ylabel(r"$\|\delta h_{s+\tau,L}\|/r$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "posscan_C_trajectories.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved A (heatmap), B (violin), C (trajectories)")
