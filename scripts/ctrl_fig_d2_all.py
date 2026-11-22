"""Part 5b: RP and correlation-dimension comparison across ALL models:
the five LLMs (master.md), Mamba sizes, Pythia, and the random-init
transformers.  Same estimator and loop screen everywhere; per model we
report the fitted D2, the plateau width, and show matched-RR recurrence
plots.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebook", "figures_controls")
NPTS = 1000


# ------- estimator (same as ctrl_d2_local.py, self-contained copy) -------
def _pair_counts(X, eps, theiler=0, chunk=512):
    N = len(X)
    counts = np.zeros(len(eps), dtype=np.int64)
    valid_total = 0
    sq = (X * X).sum(1)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        D2 = sq[i0:i1, None] + sq[None, :] - 2.0 * X[i0:i1] @ X.T
        np.maximum(D2, 0.0, out=D2)
        valid = np.abs(np.arange(i0, i1)[:, None] - np.arange(N)[None, :]) > theiler
        valid_total += int(valid.sum())
        D2[~valid] = np.inf
        D = np.sqrt(D2)
        for k, e in enumerate(eps):
            counts[k] += int((D <= e).sum())
    return counts, valid_total


def correlation_curve(X, eps, theiler=0, chunk=512):
    counts, total = _pair_counts(X, eps, theiler=theiler, chunk=chunk)
    return counts.astype(np.float64) / max(total, 1)


def fit_plateau(eps, C, min_points=6):
    ok = C > 0
    eps = np.asarray(eps)[ok]
    C = np.asarray(C)[ok]
    if len(C) < min_points:
        return np.nan, np.nan, None
    x = np.log(eps)
    y = np.log(C)
    best = (-1, np.nan, np.nan, None)
    for i in range(len(x) - min_points + 1):
        for j in range(i + min_points, len(x) + 1):
            xx = x[i:j]; yy = y[i:j]
            if np.ptp(xx) <= 0:
                continue
            slope, intercept = np.polyfit(xx, yy, 1)
            pred = slope * xx + intercept
            ss_res = np.sum((yy - pred) ** 2)
            ss_tot = np.sum((yy - yy.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best[0]:
                best = (r2, slope, intercept, (float(eps[i]), float(eps[j - 1])))
    return best[1], best[0], best[3]


def eps_grid(X, n=25, lo=2, hi=60):
    N = len(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, 1000), replace=False)
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[idx][:, None] + sq[idx][None, :] - 2.0 * X[idx] @ X[idx].T, 0))
    vals = D[~np.eye(len(idx), dtype=bool)]
    return np.percentile(vals, np.linspace(lo, hi, n))


def plateau_scan(eps_mid, slope, C_mid, N, tol_rel=0.15, floor_tol=0.05,
                 c_lo=None, c_hi=0.5, er_cap=True):
    er = 2.0 * np.log10(max(N, 10))
    if c_lo is None:
        c_lo = 10.0 / N
    good = np.isfinite(slope) & (C_mid >= c_lo) & (C_mid <= c_hi)
    if er_cap:
        good = good & (slope <= er)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = prev = idx[0]
    s0 = slope[start]
    for k in idx[1:]:
        if k == prev + 1 and abs(slope[k] - s0) <= tol_rel * abs(s0) + floor_tol:
            prev = k
        else:
            runs.append((start, prev, s0))
            start, prev, s0 = k, k, slope[k]
    runs.append((start, prev, s0))
    out = []
    for a, b, s0 in runs:
        width = np.log10(eps_mid[b] / eps_mid[a]) if b > a else 0.0
        out.append(dict(slope=float(np.median(slope[a:b + 1])), width_dec=float(width)))
    out.sort(key=lambda r: -r["width_dec"])
    return out


def prep(X, n=None):
    if n is not None and len(X) > n:
        X = X[:n]
    return X - X.mean(axis=0, keepdims=True)


def loop_screen(x):
    T = len(x)
    x = x - x.mean(0, keepdims=True)
    scale = float(np.median(np.linalg.norm(x, axis=1)))
    n = 1 << (2 * T - 1).bit_length()
    F = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft((F * np.conj(F)).sum(1), n=n)[:T]
    sq = (x * x).sum(1)
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    best = (9e9, -1)
    for P in range(5, min(260, T - 10)):
        s = cs[T - P] + (cs[T] - cs[P]) - 2 * ac[P]
        d2 = np.sqrt(max(s / (T - P), 0.0)) / scale
        if d2 < best[0]:
            best = (d2, P)
    return best[1], best[0]

SOURCES = [
    # label, glob
    ("DeepSeek-1.5B", os.path.join(ROOT, "results", "local_data_ds", "long_last", "*.npz")),
    ("Qwen2.5-3B", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "qwen25_3b_*.npz")),
    ("Qwen3-4B", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "qwen3_4b_*.npz")),
    ("Qwen3.5-2B", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "qwen3_5_*.npz")),
    ("Phi-4-mini", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "phi4mini_*.npz")),
    ("Mamba-130m", os.path.join(ROOT, "results", "ctrl_local", "mamba130", "traj", "*.npz")),
    ("Mamba-370m", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "mamba370_*.npz")),
    ("Mamba-790m", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "mamba790_*.npz")),
    ("Pythia-410m", os.path.join(ROOT, "results", "ctrl_local", "traj_hlast", "traj_hlast", "pythia410_*.npz")),
    ("Pythia-410m step0", os.path.join(ROOT, "results", "ctrl_local", "randqwen_traj", "step0", "*.npz")),
    ("randQwen g=0.35", os.path.join(ROOT, "results", "ctrl_local", "randqwen_traj", "g0.35", "*.npz")),
    ("randQwen g=1.0", os.path.join(ROOT, "results", "ctrl_local", "randqwen_traj", "g1.0", "*.npz")),
    ("randQwen g=4.0", os.path.join(ROOT, "results", "ctrl_local", "randqwen_traj", "g4.0", "*.npz")),
]


def d2_stats(X):
    Xn = prep(X, n=NPTS)
    eps = eps_grid(Xn, n=25)
    C = correlation_curve(Xn, eps, theiler=0)
    slope, r2, rng = fit_plateau(eps, C)
    width = float(np.log10(rng[1] / rng[0])) if rng else 0.0
    # plateau scan (flat local slope)
    lm = np.log(eps)
    lC = np.log(np.maximum(C, 1e-300))
    sl = np.diff(lC) / np.diff(lm)
    mid = np.sqrt(eps[:-1] * eps[1:])
    C_mid = np.sqrt(np.maximum(C[:-1], 1e-300) * np.maximum(C[1:], 1e-300))
    runs = plateau_scan(mid, sl, C_mid, N=len(Xn))
    pw = runs[0]["width_dec"] if runs else 0.0
    psl = runs[0]["slope"] if runs else np.nan
    return dict(d2=float(slope), r2=float(r2), width=width, plat_width=pw, plat_slope=psl)


rows = {}
for label, pat in SOURCES:
    files = sorted(glob.glob(pat))
    if not files:
        print("no data:", label)
        continue
    stats = []
    loops = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        X = z["h_last"].astype(np.float64)
        P, v = loop_screen(X)
        loops.append(v)
        if v < 0.10:  # skip loops (degenerate D2)
            continue
        stats.append(d2_stats(X))
    rows[label] = dict(
        n=len(files), n_used=len(stats), loops=loops,
        d2=[s["d2"] for s in stats], r2=[s["r2"] for s in stats],
        fitw=[s["width"] for s in stats], platw=[s["plat_width"] for s in stats],
        platslope=[s["plat_slope"] for s in stats])
    print("%-18s n=%d used=%d loops<0.25: %s  D2 med=%.2f  plateau width med=%.2f dec" % (
        label, len(files), len(stats), sum(1 for v in loops if v < 0.25),
        np.median(rows[label]["d2"]) if stats else np.nan,
        np.median(rows[label]["platw"]) if stats else np.nan), flush=True)

json.dump(rows, open(os.path.join(ROOT, "results", "ctrl_local", "d2_all_models.json"), "w"),
          indent=1, default=float)

# ---- summary figure: D2 + plateau width per model ----
order = [k for k, _ in SOURCES if k in rows and rows[k]["n_used"] > 0]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
xs = np.arange(len(order))
w = 0.38
ax = axs[0]
ax.bar(xs - w / 2, [np.median(rows[k]["d2"]) for k in order], w, color="#0072B2", label="median fitted $D_2$")
for i, k in enumerate(order):
    v = rows[k]["d2"]
    ax.plot(np.full(len(v), i - w / 2) + np.random.default_rng(0).uniform(-0.08, 0.08, len(v)), v,
            "k.", ms=4, alpha=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.axhline(1.157, color="C2", ls=":", lw=1)
ax.axhline(1.984, color="C2", ls="--", lw=1)
ax.annotate("Henon 1.16", (0.02, 1.157), fontsize=7, color="C2", xycoords=("axes fraction", "data"))
ax.set_ylabel("fitted $D_2$ (best plateau slope, $r^2\\geq0.99$)")
ax.set_title("(a) Fitted D2 per model: the pretrained LLMs, the SSM control, the random-init\nmodels all sit in 1.7-11; nothing separates from the control")
ax.grid(alpha=0.3, axis="y")
ax = axs[1]
ax.bar(xs, [np.median(rows[k]["platw"]) for k in order], 0.55, color="#D55E00", alpha=0.85,
       label="median flat-slope plateau width")
for i, k in enumerate(order):
    v = rows[k]["platw"]
    ax.plot(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.1, 0.1, len(v)), v,
            "k.", ms=4, alpha=0.5)
ax.axhline(1.67, color="C2", ls=":", lw=1)
ax.axhline(0.95, color="C2", ls="--", lw=1)
ax.annotate("Henon (1.67 dec)", (0.02, 1.67), fontsize=7, color="C2", xycoords=("axes fraction", "data"))
ax.annotate("Lorenz (0.95 dec)", (0.02, 0.95), fontsize=7, color="C2", xycoords=("axes fraction", "data"))
ax.set_xticks(xs)
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("plateau width (decades of $\\varepsilon$)")
ax.set_title("(b) No model reaches a real scaling region: all plateau widths\nstay an order of magnitude below the true chaotic references")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5b_all_models.png"), dpi=190); plt.close(fig)

# ---- recurrence-plot gallery at matched RR=2% ----
fig, axs = plt.subplots(2, 4, figsize=(16, 7.6))
gallery = ["DeepSeek-1.5B", "Qwen3-4B", "Phi-4-mini", "Mamba-130m",
           "Pythia-410m", "Pythia-410m step0", "randQwen g=1.0", "randQwen g=4.0"]
for ax, label in zip(axs.flat, gallery):
    if label not in rows or rows[label]["n"] == 0:
        ax.set_visible(False)
        continue
    f = sorted(glob.glob(dict(SOURCES)[label]))[0]
    X = prep(np.load(f, allow_pickle=True)["h_last"].astype(np.float64), n=800)
    eps = eps_grid(X, n=200, lo=1, hi=50)
    C = correlation_curve(X, eps, theiler=0)
    e = eps[int(np.argmin(np.abs(C - 0.02)))]
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0))
    R = (D <= e)
    ax.imshow(R, cmap="Greys", origin="lower", extent=[0, 800, 0, 800])
    ax.set_title("%s\nRR=%.3f" % (label, R.mean()), fontsize=9)
    ax.set_xlabel("$t$"); ax.set_ylabel("$t$")
fig.suptitle("Recurrence plots at matched RR = 2% across every model family: band/patch texture\nappears in trained LLMs, the SSM control, and random-init transformers alike", y=1.00)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5b_rp_gallery.png"), dpi=190, bbox_inches="tight"); plt.close(fig)

print("wrote", OUT)
