"""Shared loaders / helpers for the direction-equivalence analysis.

Data semantics (verified in explore.py / explore2.py):
- w100_layers/a1.npz : teacher-forced prompt-embedding perturbations.
    dh_norm[R=12, D=20, L=8, T=300]  ||delta h_t|| (hidden-state norm change) at
        radius radii[r], direction d, layer layers[l], generated-token position t.
        Layer 0 is identically 0 (embedding layer at generated positions).
    dm[R, D, T]   change of the top1-top2 logit margin (signed; <0 = margin shrinks).
    cos[R, D, L, T]  cosine(perturbed h, baseline h).
    margins[T]    baseline logit margin (logit top1 - logit top2, in logits).
    r_c[R, D, T]  predicted flip radius = margins*r/(-dm) (inf-huge if dm>=0).
    pred_T[R, D]  first t with r_c < r, else -1.
    dh_last[R, 3, T, 1536] delta-h vectors at layer 27 for directions 0,1,2.
    simplex[20,20] coordinates of the 20 directions in an orthonormalized
        20-token-embedding subspace (subspace[i] = token id of basis vector i).
    norm_h0[8, 300]  baseline hidden-state norms per layer/token.
- root a1.npz (=w200) and w500/a1.npz: SAME prompt and SAME 20 directions but a
  different sampled baseline continuation -> cross-rollout replication.
- state.npz : current-state (hidden) injection at layer 27, r=1e-3, 32 Gaussian
  directions, s in {16,50,100,150}; dh[s, d, t] = 0 for t < s.
- state_mid.npz : layer-14 injection, 8 directions, s in {16,50,100}.
- fr_w100_large.npz : observed free-run flip steps obs[3 radii, 20 dirs]
  (first generated-token index where text differs from baseline; -1 = never),
  radii 0.5, 1, 2; pred = pred_T at those radii.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"

# w100_layers indices
LIN_R = [0, 1, 2, 3]            # radii 1e-3 .. 3e-2 (linear regime)
MID_R = 4                        # r = 0.1
SAT_R = [8, 9, 10, 11]           # radii 0.7, 1, 1.5, 2
LARGE_R = [6, 7, 8, 9, 10, 11]   # radii 0.3 .. 2 (exponent fit)
L14, L21, L27 = 5, 6, 7          # layer indices for layers 14, 21, 27
LAYS = [1, 2, 3, 4, 5, 6, 7]     # layers with nonzero response (skip layer 0)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})


def load_w100():
    z = np.load(BASE + r"\w100_layers\a1.npz")
    d = {k: z[k] for k in z.files}
    z.close()
    d["dh_norm"] = d["dh_norm"].astype(np.float64)
    d["dm"] = d["dm"].astype(np.float64)
    d["cos"] = d["cos"].astype(np.float64)
    d["margins"] = d["margins"].astype(np.float64)
    d["r_c"] = d["r_c"].astype(np.float64)
    d["norm_h0"] = d["norm_h0"].astype(np.float64)
    return d


def load_state():
    z = np.load(BASE + r"\state.npz")
    d = {k: z[k] for k in z.files}
    z.close()
    d["dh"] = d["dh"].astype(np.float64)
    d["dm"] = d["dm"].astype(np.float64)
    return d


def load_state_mid():
    z = np.load(BASE + r"\state_mid.npz")
    d = {k: z[k] for k in z.files}
    z.close()
    d["dh"] = d["dh"].astype(np.float64)
    d["dm"] = d["dm"].astype(np.float64)
    return d


def load_fr():
    z = np.load(BASE + r"\fr_w100_large.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files if k not in ("ref_ids", "gen_ids")}
    z.close()
    return d


def load_rollouts():
    """Same prompt + same 20 directions, different baseline continuations."""
    out = {}
    for name, rel in [("w100", r"w100_layers\a1.npz"),
                      ("w200", r"a1.npz"),
                      ("w500", r"w500\a1.npz")]:
        z = np.load(BASE + "\\" + rel)
        d = {k: z[k] for k in z.files}
        z.close()
        d["dh_norm"] = d["dh_norm"].astype(np.float64)
        out[name] = d
    return out


def gain(dh, t_mask=None):
    """dh[..., T] -> median over token axis. [R, D, L] for w100 layout."""
    if t_mask is None:
        return np.median(dh, axis=-1)
    return np.ma.median(np.ma.masked_array(dh, ~t_mask), axis=-1).filled(np.nan)


def logd(x):
    """safe log for arrays containing zeros"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.where(x > 0, x, np.nan))


def spear(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic)


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def save_json(name, obj):
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if not isinstance(o, np.integer) else int(o)
        return o
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(conv(obj), f, indent=1)
    print("saved", path)


def annot_heat(ax, M, xlab, ylab, title, cmap="RdYlGn", vmin=0.0, vmax=1.0,
               fmt="{:.2f}", fs=6.0, cbar=True):
    """Small annotated matrix (avoids unreadable heatmaps)."""
    n, m = M.shape
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(m)); ax.set_xticklabels(xlab, fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(ylab, fontsize=fs)
    for i in range(n):
        for j in range(m):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color="black" if (vmax - v) > (vmax - vmin) * 0.45 else "white")
            else:
                ax.text(j, i, "nan", ha="center", va="center", fontsize=fs - 1, color="gray")
    ax.grid(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=fs)
    ax.set_title(title)


def pick_examples(values, labels_):
    """Deterministic hi / mid / lo example directions by gain `values`."""
    order = np.argsort(values)
    hi, mid, lo = int(order[-1]), int(order[len(order) // 2]), int(order[0])
    return hi, mid, lo
