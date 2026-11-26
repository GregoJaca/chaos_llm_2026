"""Shared loaders and plotting style for the 2026-11 update analysis.

Everything here reads *measured* arrays from the repository results tree.
No synthetic curves are generated.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "update_figures")
os.makedirs(FIGDIR, exist_ok=True)

# ----------------------------------------------------------------------------- style
COL = {
    "DeepSeek-1.5B": "#1f77b4",
    "DeepSeek-14B": "#08306b",
    "Qwen2.5-3B": "#e377c2",
    "Qwen3-4B": "#ff7f0e",
    "Qwen3-14B": "#8c564b",
    "Qwen3.5-2B": "#bcbd22",
    "Qwen3.5-9B": "#2ca02c",
    "Phi-4-mini": "#17becf",
    "Phi-4": "#9467bd",
    "Mamba-130m": "#555555",
    "Pythia-410m": "#8c8c8c",
    "randQwen": "#111111",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.8,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e6e6e6",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "mathtext.fontset": "dejavusans",
})


def save(fig, name, dpi=170):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ----------------------------------------------------------------------------- a1 data
# All-layer prompt-perturbation runs.  Arrays:
#   dh_norm[r, dir, layer, token]   ||h'(t,l) - h(t,l)||
#   norm_h0[layer, token]           ||h(t,l)|| of the baseline
#   cos[r, dir, layer, token]       cos between response at layer l and at the last saved layer
#   dm[r, dir, token]               change of top-2 logit margin
#   margins[token]                  baseline top-2 logit gap
A1 = {
    "DeepSeek-1.5B": "results/local_data/big/all_layers/a1.npz",
    "Qwen3-4B": "results/local_data/models/qwen3_4b/a1.npz",
    "Qwen2.5-3B": "results/local_data/models/qwen25_3b/a1.npz",
    "Phi-4-mini": "results/local_data/models/phi/a1.npz",
    "Qwen3.5-2B": "results/local_data/models/qwen35/a1.npz",
}

# depth-probe runs: lam[token, block, dir] = log local gain, layers[] block index
DEPTH = {
    "DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
    "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
    "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
    "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
    "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz",
    "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
}

MODEL_SIZE = {
    "Phi-4-mini": 3.8, "Qwen2.5-3B": 3.1, "Qwen3-4B": 4.0, "Qwen3.5-2B": 2.0,
    "DeepSeek-1.5B": 1.5, "Qwen3.5-9B": 9.0, "DeepSeek-14B": 14.0,
    "Qwen3-14B": 14.8, "Phi-4": 14.7, "Mamba-130m": 0.13,
}


def load_a1(name):
    z = np.load(os.path.join(HERE, A1[name]), allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["_path"] = A1[name]
    return out


def load_depth(name):
    z = np.load(os.path.join(HERE, DEPTH[name]), allow_pickle=True)
    return {k: z[k] for k in z.files}


def rstar_stats(name, tie=0.02):
    """Pooled r* over the five rstar prompts for one small model."""
    rs, tstar, mstar, sstar, raw = [], [], [], [], []
    for i in range(1, 6):
        p = os.path.join(HERE, f"results/rstar/{name}/p{i}/a1.npz")
        z = np.load(p, allow_pickle=True)
        m = z["margins"].astype(float)
        rc = z["r_c"][0].astype(float)
        T = len(m)
        bad = m <= tie
        rc[:, bad] = np.inf
        j = np.argmin(rc, axis=1)
        r = rc[np.arange(rc.shape[0]), j]
        rs.append(r)
        tstar.append(j)
        mstar.append(m[j])
        raw.append((m, rc))
    return dict(rstar=np.concatenate(rs), tstar=np.concatenate(tstar),
                mstar=np.concatenate(mstar), per_prompt=raw)


# ----------------------------------------------------------------------------- softloop
SOFT = {  # tag -> folder holding T*/softloop.npz
    "DeepSeek-1.5B": "results/softloop/ds15v2",
    "Qwen2.5-3B": "results/softloop/qwen25_3b",
    "Qwen3-4B": "results/softloop/qwen3_4b",
    "Qwen3.5-2B": "results/softloop/qwen3_5",
    "Qwen3.5-9B": "results/softloop/qwen35_9b",
    "Phi-4-mini": "results/softloop/phi4mini",
    "Mamba-130m": "results/softloop/mamba130",
    "Pythia-410m": "results/softloop/pythia410",
    "randQwen": "results/softloop/randqwen",
}


def softloop_runs(name, mode="loop"):
    """Yield dict(T, path, data, meta) for each temperature of one model."""
    base = os.path.join(HERE, SOFT[name])
    for Tdir in sorted(os.listdir(base)):
        d = os.path.join(base, Tdir)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "softloop.npz" if mode == "loop" else "replay/softloop.npz")
        if not os.path.exists(p):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        yield dict(T=float(meta["temp"]), path=p,
                   data=dict(np.load(p, allow_pickle=True)), meta=meta)


def loglog_slope(x, y):
    """OLS slope and R^2 of y on x."""
    A = np.polyfit(x, y, 1)
    pred = np.polyval(A, x)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return float(A[0]), float(A[1]), float(r2)
