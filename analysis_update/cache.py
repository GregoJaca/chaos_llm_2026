"""Single source of truth for the 2026-11-26 figure pack.

Every function reads a raw `.npz`/`.json` file.  No numerical data are hard-coded.
"""
from __future__ import annotations
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
RB = os.path.join(HERE, "results_big_small")

# model -> (a1 path with all layers, dtype, n_layers)
SMALL = {
    "DeepSeek-1.5B": ("results/local_data/big/all_layers/a1.npz", "fp32", 28),
    "Qwen2.5-3B":    ("results/local_data/models/qwen25_3b/a1.npz", "fp32", 36),
    "Qwen3-4B":      ("results/local_data/models/qwen3_4b/a1.npz", "fp32", 36),
    "Qwen3.5-2B":    ("results/local_data/models/qwen35/a1.npz", "fp32", 24),
    "Phi-4-mini":    ("results/local_data/models/phi/a1.npz", "fp32", 32),
}
BIG = {
    "DeepSeek-14B": ("results_big_small/big_DeepSeek_14B", 48),
    "Qwen3-14B":    ("results_big_small/big_Qwen3_14B", 40),
    "Phi-4":        ("results_big_small/big_Phi_4", 40),
}
ANGLE = {
    "DeepSeek-1.5B": "results/update/angle_ds15/angle.npz",
    "Qwen2.5-3B":    "results/update/angle_qwen25/angle.npz",
    "Qwen3-4B":      "results/update/angle_qwen34/angle.npz",
    "Phi-4-mini":    "results/update/angle_phi/angle.npz",
    "Qwen3.5-2B":    "results/update/angle_qwen35/angle.npz",
    "Mamba-130m":    "results/update/angle_mamba130/angle.npz",
    "Pythia-410m":   "results/update/angle_pythia410/angle.npz",
    "randQwen":      "results/update/angle_randqwen/angle.npz",
}
RSTAR_SMALL = {"deepseek": "DeepSeek-1.5B", "phi4mini": "Phi-4-mini",
               "qwen25_3b": "Qwen2.5-3B", "qwen3_4b": "Qwen3-4B",
               "qwen3_5": "Qwen3.5-2B"}
RSTAR_BIG = {"DeepSeek-14B": "DeepSeek_14B", "Qwen3-14B": "Qwen3_14B", "Phi-4": "Phi_4"}
SOFT = {
    "DeepSeek-1.5B": "results/softloop/ds15v2",
    "Qwen2.5-3B":    "results/softloop/qwen25_3b",
    "Qwen3-4B":      "results/softloop/qwen3_4b",
    "Qwen3.5-2B":    "results/softloop/qwen3_5",
    "Qwen3.5-9B":    "results/softloop/qwen35_9b",
    "Phi-4-mini":    "results/softloop/phi4mini",
    "Mamba-130m":    "results/softloop/mamba130",
    "Pythia-410m":   "results/softloop/pythia410",
    "randQwen":      "results/softloop/randqwen",
}
COL = {
    "DeepSeek-1.5B": "#1f77b4", "DeepSeek-14B": "#08306b",
    "Qwen2.5-3B": "#e377c2", "Qwen3-4B": "#ff7f0e", "Qwen3-14B": "#8c564b",
    "Qwen3.5-2B": "#bcbd22", "Qwen3.5-9B": "#2ca02c",
    "Phi-4-mini": "#17becf", "Phi-4": "#9467bd",
    "Mamba-130m": "#555555", "Pythia-410m": "#9a9a9a", "randQwen": "#111111",
}
SIZE = {"Phi-4-mini": 3.8, "Qwen2.5-3B": 3.1, "Qwen3-4B": 4.0, "Qwen3.5-2B": 2.0,
        "DeepSeek-1.5B": 1.5, "Qwen3.5-9B": 9.0, "DeepSeek-14B": 14.0,
        "Qwen3-14B": 14.8, "Phi-4": 14.7, "Mamba-130m": 0.13, "Pythia-410m": 0.41}


def load(rel):
    return dict(np.load(os.path.join(HERE, rel), allow_pickle=True))


def a1(name):
    """Compact summary of a small-model A1 run (all saved layers, all radii)."""
    p, dt, nL = SMALL[name]
    z = load(p)
    dh = z["dh_norm"].astype(np.float32)      # (r, dir, layer, T)
    hn = z["norm_h0"].astype(np.float32)      # (layer, T)
    med = np.median(dh, axis=1)               # (r, layer, T)
    lo, hi = np.percentile(dh, [25, 75], axis=1)
    rel = med / np.maximum(hn, 1e-30)[None]
    return dict(dh=dh, med=med, lo=lo, hi=hi, hn=hn, rel=rel,
                radii=z["radii"].astype(float), layers=z["layers"].astype(int),
                margins=z["margins"].astype(float), dm=z["dm"].astype(float),
                dtype=dt, nL=nL, T=dh.shape[-1])


def big(name, tag="inter"):
    z = load(os.path.join(BIG[name][0] + ("_" + tag if os.path.exists(os.path.join(HERE, BIG[name][0] + "_" + tag + ".npz")) else "") + ".npz"))
    return z


def big_tags(name):
    out = []
    for tag in ["inter", "childhood", "cyber"]:
        if os.path.exists(os.path.join(HERE, BIG[name][0] + "_" + tag + ".npz")):
            out.append(tag)
    return out


def angle(name):
    if name in ANGLE:
        z = load(ANGLE[name])
        return dict(angle=z["angle"].astype(float), cos=z["cos"].astype(float),
                    dh=z["dh_norm"].astype(float), hn=z["norm_h"].astype(float),
                    radii=z["radii"].astype(float), layers=z["layers"].astype(int),
                    T=z["angle"].shape[-1], n=z["angle"].shape[1])
    if name in BIG:
        z = big(name)
        return dict(angle=z["angle"].astype(float), cos=z["cosang"].astype(float),
                    radii=z["radii"].astype(float), T=z["angle"].shape[-1],
                    n=z["angle"].shape[0], layers=np.asarray([z["layers"][-1]]))
    return None


def rstar(name):
    """Pooled r*, controlling step, m*, s* for one model (5 small prompts / 3 big)."""
    rs, ts, ms, ss = [], [], [], []
    if name in RSTAR_SMALL.values():
        tag = {v: k for k, v in RSTAR_SMALL.items()}[name]
        for i in range(1, 6):
            z = load(f"results/rstar/{tag}/p{i}/a1.npz")
            m = z["margins"].astype(float)
            rc = z["r_c"][0].astype(float)
            dm = z["dm"][0].astype(float)
            r0 = float(z["radii"][0])
            rc[:, m <= 0.02] = np.inf
            j = np.argmin(rc, axis=1)
            r = rc[np.arange(rc.shape[0]), j]
            ok = np.isfinite(r)
            rs.append(r[ok]); ts.append(j[ok]); ms.append(m[j][ok])
            ss.append(np.abs(dm[np.arange(dm.shape[0]), j])[ok] / r0)
    elif name in RSTAR_BIG:
        for tag in big_tags(name):
            z = big(name, tag)
            m = z["margins"].astype(float)
            rc = np.full((z["dm"].shape[0], len(m)), np.inf)
            # r_c is not saved in the compact file; recompute from dm and m
            rc = np.where(z["dm"] / float(z["radii"][0]) < 0, m[None] * float(z["radii"][0]) / np.maximum(-z["dm"], 1e-30), np.inf)
            rc[:, m <= 0.02] = np.inf
            j = np.argmin(rc, axis=1)
            r = rc[np.arange(rc.shape[0]), j]
            ok = np.isfinite(r)
            rs.append(r[ok]); ts.append(j[ok]); ms.append(m[j][ok])
            ss.append(np.abs(z["dm"][np.arange(z["dm"].shape[0]), j])[ok] / float(z["radii"][0]))
    else:
        return None
    return dict(r=np.concatenate(rs), t=np.concatenate(ts),
                m=np.concatenate(ms), s=np.concatenate(ss))


def soft_runs(name, mode="loop"):
    base = os.path.join(HERE, SOFT[name])
    if not os.path.isdir(base):
        return []
    out = []
    for Tdir in sorted(os.listdir(base)):
        d = os.path.join(base, Tdir)
        p = os.path.join(d, "softloop.npz" if mode == "loop" else "replay/softloop.npz")
        if os.path.exists(p):
            meta = json.load(open(os.path.join(d, "meta.json")))
            out.append((float(meta["temp"]), load(p), meta))
    return out


def all_names():
    return ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Phi-4-mini",
            "DeepSeek-14B", "Qwen3-14B", "Phi-4"]


def pctl(a, ps=(5, 25, 50, 75, 95), axis=0):
    return np.percentile(a, ps, axis=axis)


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(1, len(x) + 1) / len(x)
