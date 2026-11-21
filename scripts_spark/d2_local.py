#!/usr/bin/env python
"""Correlation dimension (D2) on precomputed distance matrices — laptop side.

The GPU job ``scripts_spark/d2_distances.py`` writes one full ``[N,N]``
Euclidean distance matrix per trajectory (``dist_<name>_h_last.npy``, fp32)
plus ``summary.json``.  This script consumes those matrices and produces the
D2 / TwoNN / MLE numbers, the repetition screen and the figures.  No GPU, no
torch, no model.

Usage
-----
    python scripts_spark/d2_local.py --dir results_big/ds14/d2 \
        --theiler 0,50,500 --out notebook_big/figures --tag ds14

    # validation on the 1.5B trajectories that exist only as .pt / .npz
    python scripts_spark/d2_local.py --pt <files-or-dirs> --out notebook_big/figures

Definitions
-----------
``d_typ``  typical point separation: median over points of the rms distance to
           the other points.  Every eps below is quoted in units of ``d_typ``,
           which makes trajectories with different state norms comparable.

``C(eps)`` correlation integral.  With Theiler window W,
               C(eps) = #{i,j : |i-j| > W, ||x_i-x_j|| < eps} / #{i,j : |i-j| > W}
           The denominator is N(N-1) - 2*sum_{k<=W}(N-k), so C -> 1 at
           saturation.  For W = 0 this is exactly 2/(N(N-1)) #{i<j: d_ij<eps};
           the count is evaluated by ``searchsorted`` on the sorted upper
           triangle, so the whole curve costs one sort per Theiler setting.

local slope   d log C / d log eps (central differences on the eps grid).

valid window  three simultaneous conditions, all reported separately:
               (i)   flatness: the longest run of eps with
                     |slope - median(slope over run)| <= TOL and the run at
                     least MIN_DECADES wide (this replaces the reference
                     implementation's "max R^2 over every sub-interval",
                     which can always be driven to R^2 > 0.999 on two
                     adjacent grid points and is therefore not a test);
               (ii)  depletion floor, C(eps) >= 10/N  (docs/FINDINGS_AND_
                     PROCEDURE.md, "mark the valid window [10/N, ...]"),
                     i.e. at least ~5N pairs inside the ball;
               (iii) below saturation, C(eps) <= 0.1.
           If (i) is empty the verdict is "no scaling region" — which is the
           expected answer for these point clouds.

``twonn``  per point mu = r2/r1 (1st over 2nd nearest-neighbour distance).
           For a locally uniform d-dimensional cloud mu is Pareto with tail
           exponent d, so d = 1 / <ln mu> (MLE) and the reference code's
           fit of the empirical CDF, -ln(1-F) vs ln(mu), is the same
           estimator done with OLS.  Both are returned.  The frequently
           quoted "log(2)/<log(mu)>" is *not* this estimator: with base-2
           logs it returns ln(2)^2 * d = 0.4805 d (checked on uniform
           hypercubes, d = 2/5/10).

``mle``    Levina-Bickel: m_k(i) = (1/(k-1)) sum_{j<k} ln(d_k/d_j),
           d_k = <1/m_k> over points.

repetition screen
           per point, the distance to its nearest neighbour with |i-j| >= W
           (W = 50, same as the GPU screen), divided by d_typ.  Values below
           1e-3 flag near-exact loops; the reported number is the *fraction*
           of points below that threshold (the GPU job stores only median
           and p05).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import re
import sys
import time
import zipfile

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.8,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T4_REF = os.path.join(os.path.dirname(ROOT), "chaos_llm_2026", "results", "d2",
                      "d2_raw_vs_shuffled.json")

COL = ["#2166ac", "#d6604d", "#1b7837"]      # theiler 0 / 50 / 500
C_LOOP = "#b2182b"
C_OK = "#2166ac"
C_GREY = "#8a8a8a"

N_EPS = 90
EPS_LO, EPS_HI = 1e-2, 2.5           # in units of d_typ
TOL = 0.15                           # absolute slope-flatness floor
TOL_REL = 0.06                       # ... and 6% of the dimension
MIN_DECADES = 0.40
REP_THRESH = 1e-3


# ------------------------------------------------------------------ loading
class _PtUnpickler(pickle.Unpickler):
    """torch.save zip reader without torch (plain tensors only)."""

    def __init__(self, fh, keys):
        super().__init__(fh)
        self.keys = keys

    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return self._rebuild
        if module.startswith("torch"):
            return lambda *a, **k: None
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        return ("storage", pid[2], pid[4])

    @staticmethod
    def _rebuild(storage, storage_offset, size, stride, *rest):
        _, key, numel = storage
        return {"key": key, "offset": int(storage_offset),
                "size": tuple(int(s) for s in size),
                "stride": tuple(int(s) for s in stride), "numel": int(numel)}


def load_pt(path) -> np.ndarray:
    """Read a tensor saved by torch.save(..., weights_only-compatible) as numpy.

    Supports the fp32 CPU/CUDA tensors written by the trajectory collectors:
    a zip with ``*/data.pkl`` and ``*/data/<key>``.  Strides are honoured, so
    non-contiguous saves work too.
    """
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]
    obj = _PtUnpickler(io.BytesIO(z.read(pkl)), z.namelist()).load()
    if not isinstance(obj, dict) or "size" not in obj:
        raise ValueError(f"{path}: not a plain tensor save")
    raw = z.read(f"{prefix}data/{obj['key']}")
    flat = np.frombuffer(raw, dtype="<f4", count=obj["numel"])
    shape, stride = obj["size"], obj["stride"]
    # only the common contiguous / single-strided cases occur here
    if stride == tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape))):
        arr = flat.reshape(shape)
    else:
        base = np.lib.stride_tricks.as_strided(
            flat, shape=shape,
            strides=tuple(4 * s for s in stride), writeable=False)
        arr = np.array(base)
    return np.asarray(arr, dtype=np.float32)[obj["offset"]:] if obj["offset"] else arr


def load_states(path) -> np.ndarray:
    """[N,D] float32 states from .pt (torch), .npz (h_last) or .npy."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        h = load_pt(path)
    elif ext == ".npz":
        d = np.load(path)
        for k in ("h_last", "states", "h"):
            if k in d:
                h = d[k]
                break
        else:
            raise KeyError(f"{path}: no h_last/states/h field ({list(d.keys())})")
    else:
        h = np.load(path)
    h = np.asarray(h, dtype=np.float32)
    h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"{path}: expected [N,D] after squeeze, got {h.shape}")
    if h.shape[0] < h.shape[-1]:
        pass  # N<D is fine
    return h


def dist_matrix(h: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Stable Euclidean distance matrix, fp32 out.

    Gram expansion in float64 after removing the mean (cdist is translation
    invariant).  float32 Gram expansion would lose the 1e-3-relative
    distances the repetition screen looks at; float64 keeps ~10 decades of
    headroom there.
    """
    x = np.asarray(h, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    sq = np.einsum("ij,ij->i", x, x)
    n = x.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        g = x[i0:i1] @ x.T
        d2 = sq[i0:i1, None] + sq[None, :] - 2.0 * g
        np.maximum(d2, 0.0, out=d2)
        out[i0:i1] = np.sqrt(d2, out=d2)
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


# -------------------------------------------------------------- estimators
def typical_scale(D: np.ndarray, n_sample: int = 512, seed: int = 0) -> float:
    """Median over points of the rms distance to the cloud (a separation)."""
    n = D.shape[0]
    idx = np.arange(n)
    if n > n_sample:
        idx = np.random.default_rng(seed).choice(n, n_sample, replace=False)
    sub = D[np.ix_(idx, idx)]
    return float(np.median(np.sqrt((sub.astype(np.float64) ** 2).mean(axis=1))))


def upper_triangle_sorted(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    vals = np.concatenate([D[i, i + 1:] for i in range(n - 1)])
    return np.sort(vals)


def correlation_curve(D: np.ndarray, eps: np.ndarray, theiler: int = 0):
    """C(eps) from the distance matrix; returns (C, n_valid_pairs)."""
    n = D.shape[0]
    s = upper_triangle_sorted(D)
    counts = np.searchsorted(s, eps, side="right").astype(np.float64)
    n_valid = float(len(s))
    if theiler > 0:
        w = min(theiler, n - 1)
        for k in range(1, w + 1):
            dk = np.sort(np.diagonal(D, k))
            counts -= np.searchsorted(dk, eps, side="right")
            n_valid -= (n - k)
    return counts / max(n_valid, 1.0), int(n_valid)


def local_slope(eps: np.ndarray, C: np.ndarray):
    """d log C / d log eps, keeping only eps where C > 0."""
    ok = C > 0
    e, c = np.asarray(eps)[ok], np.asarray(C)[ok]
    if len(e) < 3:
        return e, np.full(len(e), np.nan)
    return e, np.gradient(np.log(c), np.log(e), edge_order=2)


def flat_window(eps, slope, tol=TOL, min_decades=MIN_DECADES):
    """Longest run of eps whose local slope stays within tol of its own median.

    Returns (i0, i1, d2) with the inclusive index range, or (None, None, nan).
    """
    e, s = np.asarray(eps, float), np.asarray(slope, float)
    ok = np.isfinite(s)
    e, s = e[ok], s[ok]
    n = len(s)
    if n < 5:
        return None, None, float("nan")
    le = np.log10(e)
    best = (0.0, None, None)
    i = 0
    while i < n - 4:
        j = i + 4                      # at least 5 points
        while j < n:
            seg = s[i:j + 1]
            med = np.median(seg)
            if max(seg.max() - med, med - seg.min()) > tol:
                break
            j += 1
        j -= 1
        if j < i + 4:
            i += 1
            continue
        dec = le[j] - le[i]
        if dec >= min_decades and dec > best[0]:
            best = (dec, i, j)
        i += 1
    if best[1] is None:
        return None, None, float("nan")
    _, i, j = best
    return i, j, float(np.median(s[i:j + 1]))


def _medfilt3(s):
    """3-point median filter (edge points kept)."""
    s = np.asarray(s, float)
    if len(s) < 3:
        return s
    out = s.copy()
    out[1:-1] = np.median(np.lib.stride_tricks.sliding_window_view(s, 3), axis=1)
    return out


def valid_window(n: int, eps, C, slope):
    """Apply the three conditions and return the window bookkeeping."""
    N = int(n)
    ok = C > 0
    e, c, s = np.asarray(eps)[ok], np.asarray(C)[ok], np.asarray(slope)[ok]

    depletion = 10.0 / N
    sat_frac = 0.10
    if np.any(c >= depletion):
        eps_dep = float(np.exp(np.interp(np.log(depletion), np.log(c), np.log(e))))
    else:
        eps_dep = float(e[-1])
    eps_sat = float(np.exp(np.interp(np.log(sat_frac), np.log(c), np.log(e))))

    # tolerance: flat means "within a few percent of its own value", with an
    # absolute floor so that low dimensions are not judged in relative units
    adm = (c >= depletion) & (c <= sat_frac)
    ref = float(np.median(s[adm])) if adm.sum() >= 3 else float(np.median(s))
    tol = float(max(TOL, TOL_REL * abs(ref)))
    i0, i1, d2_flat = flat_window(e, _medfilt3(s), tol=tol)

    if i0 is not None:
        w_lo, w_hi = float(e[i0]), float(e[i1])
        d2_flat = float(np.median(s[i0:i1 + 1]))
        drift = float((s[i1] - s[i0]) / max(np.log10(e[i1] / e[i0]), 1e-9))
    else:
        w_lo = w_hi = float("nan")
        drift = float("nan")
    lo = max(w_lo, eps_dep) if np.isfinite(w_lo) else float("nan")
    hi = min(w_hi, eps_sat) if np.isfinite(w_hi) else float("nan")
    decades = float(np.log10(hi / lo)) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 0.0

    # slope at the two edges of the admissible range: the honest "how much
    # does the slope move across the range that is actually measurable"
    s_dep = float(np.interp(np.log(eps_dep), np.log(e), s))
    s_sat = float(np.interp(np.log(eps_sat), np.log(e), s))
    s_adm_min = float(np.min(s[adm])) if adm.any() else float("nan")
    s_adm_max = float(np.max(s[adm])) if adm.any() else float("nan")

    # chord slope over the whole admissible range (the number you get if you
    # do not demand a plateau) — quoted so the window dependence is visible
    if adm.sum() >= 3:
        d2_adm = float(np.polyfit(np.log(e[adm]), np.log(c[adm]), 1)[0])
        dec_adm = float(np.log10(e[adm][-1] / e[adm][0]))
    else:
        d2_adm = dec_adm = float("nan")

    # chord slope over the surviving window points
    if decades > 0:
        sel = (e >= lo) & (e <= hi) & (c > 0)
        d2 = float(np.polyfit(np.log(e[sel]), np.log(c[sel]), 1)[0]) if sel.sum() >= 3 else float("nan")
    else:
        d2 = float("nan")

    return dict(
        d2=d2, d2_flat=d2_flat, decades=decades, valid=bool(decades >= MIN_DECADES),
        eps_lo=lo, eps_hi=hi, eps_flat_lo=w_lo, eps_flat_hi=w_hi,
        eps_depletion=eps_dep, eps_saturation=eps_sat,
        depletion_frac=depletion, slope_ceiling=float(2 * np.log10(N)),
        slope_drift_per_decade=drift, tol_used=tol,
        slope_at_depletion=s_dep, slope_at_saturation=s_sat,
        slope_admissible_min=s_adm_min, slope_admissible_max=s_adm_max,
        d2_admissible=d2_adm, decades_admissible=dec_adm,
    )


def twonn(D: np.ndarray, max_n: int = 2500):
    """TwoNN: MLE 1/<ln mu>, the reference-style CDF-OLS fit, and the wrong log2 one."""
    n = D.shape[0]
    sub = D[: min(n, max_n), : min(n, max_n)].astype(np.float64)
    nsub = len(sub)
    sub.flat[:: nsub + 1] = np.inf
    part = np.partition(sub, 2, axis=1)[:, :3]
    part.sort(axis=1)
    r1, r2 = part[:, 0], part[:, 1]
    mu = r2 / np.maximum(r1, 1e-30)
    good = np.isfinite(mu) & (mu > 1.0)
    nmax = min(n, max_n)
    mu = np.sort(mu[good])
    if len(mu) < 20:
        return dict(mle=float("nan"), cdf_fit=float("nan"), log2_bug=float("nan"),
                    n=int(len(mu)), n_dropped=int(nmax - len(mu)), mu=mu)
    N = len(mu)
    F = (np.arange(N) + 1) / N
    okc = (1 - F) > 1e-9
    cdf_fit = float(np.polyfit(np.log(mu[okc]), -np.log(1 - F[okc]), 1)[0])
    mle = float(1.0 / np.mean(np.log(mu)))
    log2_bug = float(np.log(2) / np.mean(np.log2(mu)))
    return dict(mle=mle, cdf_fit=cdf_fit, log2_bug=log2_bug, n=int(N),
                n_dropped=int(nmax - N), mu=mu, F=F)


def levina_bickel(D: np.ndarray, ks=(5, 10, 20), max_n: int = 2500):
    """MLE intrinsic dimension for a few neighbourhood sizes."""
    n = D.shape[0]
    m = min(n, max_n)
    sub = D[:m, :m].astype(np.float64)
    sub.flat[:: m + 1] = np.inf
    kmax = min(max(ks), m - 1)
    part = np.partition(sub, kmax, axis=1)[:, :kmax]
    part.sort(axis=1)                               # nearest neighbour at [:, 0]
    out = {}
    for k in ks:
        if k >= m:
            continue
        logs = np.log(part[:, k - 1][:, None]) - np.log(part[:, : k - 1])
        mk = logs.mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dk = 1.0 / mk
        dk = dk[np.isfinite(dk)]
        out[str(k)] = dict(d2=float(np.mean(dk)), n=int(len(dk)))
    return out


def repetition_screen(D: np.ndarray, theiler: int = 50, d_typ: float = 1.0):
    """Fraction of points with a near-exact repeat at |i-j| >= theiler."""
    n = D.shape[0]
    w = min(theiler, n - 1)
    m = np.array(D, dtype=np.float32, copy=True)
    np.fill_diagonal(m, np.inf)
    for k in range(1, w + 1):
        np.fill_diagonal(m[k:], np.inf)
        np.fill_diagonal(m[:, k:], np.inf)
    nearest = m.min(axis=1).astype(np.float64)
    rel = nearest / max(d_typ, 1e-30)
    return dict(frac_below_thresh=float(np.mean(rel < REP_THRESH)),
                median=float(np.median(rel)), p05=float(np.quantile(rel, 0.05)),
                thresh=REP_THRESH, theiler=int(w))


def period_screen(D: np.ndarray, d_typ: float, lags: np.ndarray,
                  lag_rel: np.ndarray):
    """Loop period from a sharp dip in the mean distance at lag k, plus its coverage.

    mean||x_t - x_{t+k}|| grows with k for any trajectory, so the raw minimum
    is meaningless; what matters is a *sharp* dip against its neighbours.
    ``period_prominence`` = md(k) / median(md(k±1,2,3)); a drifting loop gives
    ~0.1 at its period, a healthy trajectory ~1.  ``coverage`` is the fraction
    of positions whose state at lag p is within 0.1 d_typ of the state at t —
    the statistic that says how much of the trajectory participates.

    Empirically (20 local trajectories) consecutive states are 0.84-1.03 d_typ
    apart, so there is no smoothness confound: the lag curve is flat at ~1 for
    healthy runs and dips to 0.04-0.08 only at a loop period.  Prominence sorts
    them cleanly, 0.04-0.45 (loops) against 0.91-0.99 (healthy).
    """
    md = np.asarray(lag_rel, float)
    n = len(md)
    if n < 12:
        return dict(period_lag=None, period_prominence=float("nan"),
                    period_rel=float("nan"), coverage=float("nan"))
    prom = np.full(n, np.inf)
    for i in range(3, n - 3):
        nb = np.concatenate([md[i - 3:i], md[i + 1:i + 4]])
        prom[i] = md[i] / max(np.median(nb), 1e-30)
    cand = np.where(prom < 0.4)[0]
    i = int(cand[0]) if len(cand) else int(np.argmin(prom))
    lag = int(lags[i])
    dg = np.diagonal(D, lag) / max(d_typ, 1e-30)
    return dict(period_lag=lag, period_prominence=float(prom[i]),
                period_rel=float(md[i]), coverage=float(np.mean(dg < 0.1)))


# --------------------------------------------- reference (T4) estimator, for
# an apples-to-apples comparison with results/d2/*.json
def _t4_d2_curve(Dsub: np.ndarray, n_eps: int = 22, lo: int = 2, hi: int = 60,
                 theiler: int = 20):
    """The reference estimator (dimension.py:fit_plateau), reproduced verbatim.

    ``theiler=20`` matches scripts/d2_all_controls.py, the script that produced
    results/d2/d2_raw_vs_shuffled.json.  Also returned: how wide the winning
    window is in decades and how many pairs sit inside it — the two numbers
    that decide whether the reported D2 means anything.
    """
    n = Dsub.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 1000), replace=False)
    vals = Dsub[np.ix_(idx, idx)]
    vals = vals[~np.eye(len(idx), dtype=bool)]
    eps = np.percentile(vals.astype(np.float64), np.linspace(lo, hi, n_eps))
    s = upper_triangle_sorted(Dsub)
    C, _ = correlation_curve(Dsub, eps, theiler=theiler)
    ok = C > 0
    x, y = np.log(eps[ok]), np.log(C[ok])
    best = (-1.0, np.nan, None, 0, 0.0)
    mp = 6
    for i in range(len(x) - mp + 1):
        for j in range(i + mp, len(x) + 1):
            xx, yy = x[i:j], y[i:j]
            if np.ptp(xx) <= 0:
                continue
            sl, ic = np.polyfit(xx, yy, 1)
            pred = sl * xx + ic
            ss_res = np.sum((yy - pred) ** 2)
            ss_tot = np.sum((yy - yy.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 > best[0]:
                best = (r2, sl, (float(eps[ok][i]), float(eps[ok][j - 1])), j - i,
                        float(xx[-1] - xx[0]) / np.log(10))
    r2, sl, rng_, npts, dec = best
    pairs = None
    if rng_ is not None:
        pairs = int(np.searchsorted(s, rng_[1], side="right")
                    - np.searchsorted(s, rng_[0], side="right"))
    return dict(d2=float(sl), r2=float(r2), fit_range=rng_, n=int(n),
                n_eps_used=int(len(x)), n_fit_points=int(npts),
                fit_decades=float(dec), pairs_in_window=pairs,
                theiler=int(theiler))


# ------------------------------------------------------------------- analyse
def analyse(D: np.ndarray, name: str, theilers=(0, 50, 500), t4_style=True,
            states: np.ndarray | None = None):
    t0 = time.time()
    N = D.shape[0]
    d_typ = typical_scale(D)
    eps = d_typ * np.geomspace(EPS_LO, EPS_HI, N_EPS)

    curves = {}
    for W in theilers:
        C, nv = correlation_curve(D, eps, theiler=W)
        e, s = local_slope(eps, C)
        curves[int(W)] = dict(eps=(e / d_typ), C=C[C > 0], slope=s, n_valid=nv)

    vw = {str(W): valid_window(N, curves[W]["eps"], curves[W]["C"],
                               curves[W]["slope"])
          for W in theilers}
    # N-dependence of the local slope at fixed Theiler (the honest check)
    ndep = {}
    for frac in (1.0, 0.5, 0.25):
        m = int(round(N * frac))
        if m < 60:
            continue
        if frac == 1.0:
            Ds = D
        else:
            idx = np.sort(np.random.default_rng(7).choice(N, m, replace=False))
            Ds = D[np.ix_(idx, idx)]
        C, _ = correlation_curve(Ds, eps, theiler=max(0, min(theilers[0], m - 1)))
        e, s = local_slope(eps, C)
        v = valid_window(m, e / d_typ, C[C > 0], s)
        ndep[str(m)] = dict(d2=v["d2"], d2_flat=v["d2_flat"],
                            decades=v["decades"], valid=v["valid"],
                            slope=s, eps=(e / d_typ), C=C[C > 0])

    tw = twonn(D)
    mu = tw.pop("mu", np.array([]))
    F = tw.pop("F", np.array([]))
    mle = levina_bickel(D)
    rep = repetition_screen(D, theiler=50, d_typ=d_typ)
    lags = np.arange(1, min(600, N - 2))
    md = np.array([np.diagonal(D, int(k)).mean() for k in lags]) / d_typ
    per = period_screen(D, d_typ, lags, md)
    t4 = None
    if t4_style and N >= 100:
        # the reference script loads the first 2000 states and calls cdist on
        # them; rebuild that exact matrix when the states are available,
        # otherwise slice the full one (agrees to ~4%)
        if states is not None and len(states) >= 2000:
            t4 = _t4_d2_curve(dist_matrix(states[:2000]))
        else:
            t4 = _t4_d2_curve(D[: min(N, 2000)])

    verdict = "no stable plateau"
    if not vw[str(theilers[0])]["valid"]:
        verdict = "no scaling region (window empty)"
    cov = per["coverage"]
    prom = per["period_prominence"]
    if rep["frac_below_thresh"] > 0.02:
        verdict = "repetition loop: D2/TwoNN values are spurious"
    elif prom == prom and prom < 0.6:
        verdict = (f"period-{per['period_lag']} loop "
                   f"({100*cov:.0f}% of positions return): D2/TwoNN values are spurious")
    elif prom == prom and prom < 0.8:
        verdict = (f"partially repetitive ({100*cov:.0f}% at lag "
                   f"{per['period_lag']}): D2 unreliable")

    out = dict(
        name=name, n=int(N), dim=None, d_typ=float(d_typ),
        theiler_default=int(theilers[0]),
        theiler={str(W): dict(
            d2=vw[str(W)]["d2"], d2_flat=vw[str(W)]["d2_flat"],
            decades=vw[str(W)]["decades"], valid=vw[str(W)]["valid"],
            eps_lo=vw[str(W)]["eps_lo"], eps_hi=vw[str(W)]["eps_hi"],
            eps_depletion=vw[str(W)]["eps_depletion"],
            eps_saturation=vw[str(W)]["eps_saturation"],
            n_valid_pairs=curves[int(W)]["n_valid"],
            d2_admissible=vw[str(W)]["d2_admissible"],
            decades_admissible=vw[str(W)]["decades_admissible"],
            slope_at_depletion=vw[str(W)]["slope_at_depletion"],
            slope_at_saturation=vw[str(W)]["slope_at_saturation"],
            slope_admissible_min=vw[str(W)]["slope_admissible_min"],
            slope_admissible_max=vw[str(W)]["slope_admissible_max"],
            slope_drift_per_decade=vw[str(W)]["slope_drift_per_decade"],
            tol_used=vw[str(W)]["tol_used"],
            slope_min=float(np.nanmin(curves[int(W)]["slope"])),
            slope_max=float(np.nanmax(curves[int(W)]["slope"])),
        ) for W in theilers},
        slope_ceiling_2log10N=float(2 * np.log10(N)),
        ndep={k: dict(d2=v["d2"], decades=v["decades"]) for k, v in ndep.items()},
        twonn=tw, mle=mle, rep_screen=rep, period=per, t4_style=t4,
        verdict=verdict,
        runtime_s=float(time.time() - t0),
    )
    return out, dict(eps=eps / d_typ, curves=curves, vw=vw, ndep=ndep,
                     d_typ=d_typ, eps_abs=eps, mu=mu, F=F, lags=lags,
                     lag_rel=md)


# ---------------------------------------------------------------------- plots
def _shade(ax, lo, hi, color=C_OK, alpha=0.12, label=None):
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0, label=label)


def plot_one(res, fav, path, tag=""):
    fig, axs = plt.subplots(2, 4, figsize=(16.4, 6.9))
    ax = axs[0, 0]
    for i, W in enumerate(sorted(fav["curves"])):
        c = fav["curves"][W]
        ax.loglog(c["eps"], c["C"], color=COL[i % 3], lw=1.2, label=f"W={W}")
    ax.axhline(10.0 / res["n"], color=C_GREY, ls=":", lw=1)
    ax.text(fav["eps"][2], 10.0 / res["n"] * 1.3, "10/N", color=C_GREY, fontsize=7.5)
    ax.set_xlabel(r"$\epsilon/d_{typ}$")
    ax.set_ylabel(r"$C(\epsilon)$")
    ax.set_title("(a) correlation integral")
    ax.legend(loc="lower right")
    v = res["theiler"][str(res["theiler_default"])]
    _shade(ax, v["eps_lo"], v["eps_hi"], label="valid window (W=0)")
    if v["valid"]:
        ax.axvline(v["eps_lo"], color=C_OK, ls="--", lw=0.8)
        ax.axvline(v["eps_hi"], color=C_OK, ls="--", lw=0.8)

    ax = axs[0, 1]
    for i, W in enumerate(sorted(fav["curves"])):
        c = fav["curves"][W]
        ax.semilogx(c["eps"], c["slope"], color=COL[i % 3], lw=1.1, label=f"W={W}")
    _shade(ax, v["eps_lo"], v["eps_hi"])
    ax.axhline(res["slope_ceiling_2log10N"], color="#762a83", ls=":", lw=1.0,
               label=r"finite-$N$ ceiling $2\log_{10}N$")
    if np.isfinite(v["d2"]):
        ax.axhline(v["d2"], color=C_OK, ls="--", lw=1.0,
                   label=rf"$\hat D_2={v['d2']:.2f}$ over window")
    ax.set_xlabel(r"$\epsilon/d_{typ}$")
    ax.set_ylabel(r"$d\log C/d\log\epsilon$")
    ax.set_title("(b) local slope")
    ax.set_ylim(0, max(6.0, 1.5 * res["slope_ceiling_2log10N"]))
    ax.legend(loc="upper right")

    ax = axs[0, 2]
    for i, (k, c) in enumerate(sorted(fav["ndep"].items(), key=lambda kv: -int(kv[0]))):
        ax.semilogx(c["eps"], c["slope"], lw=1.0, color=plt.cm.viridis(i / 3.0),
                    label=f"N={k}  $D_2$={c['d2']:.2f}")
    ax.set_xlabel(r"$\epsilon/d_{typ}$")
    ax.set_ylabel("local slope")
    ax.set_title("(c) same curve at N, N/2, N/4")
    ax.legend(loc="upper right")

    ax = axs[1, 0]
    tw = res["twonn"]
    mu, F = fav["mu"], fav["F"]
    if len(mu) > 20:
        okc = (1 - F) > 1e-9
        ax.plot(np.log(mu[okc]), -np.log(1 - F[okc]), ".", ms=2.2, color=C_OK,
                label="empirical $\\mu$ CDF")
        xx = np.log(mu[okc])
        ax.plot(xx, tw["cdf_fit"] * xx, color=C_LOOP, lw=1.0,
                label=f"OLS slope {tw['cdf_fit']:.2f}")
    ax.set_xlabel(r"$\ln\mu=\ln(r_2/r_1)$")
    ax.set_ylabel(r"$-\ln(1-F)$")
    ax.set_title("(d) TwoNN: $\\mu$ CDF")
    ax.legend(loc="lower right")
    ax.text(0.03, 0.97,
            f"MLE $1/\\langle\\ln\\mu\\rangle$ = {tw['mle']:.2f}\n"
            f"log2 variant (wrong) = {tw['log2_bug']:.2f}\n"
            f"$n$={tw['n']}, {tw['n_dropped']} ties/dups dropped",
            va="top", fontsize=7.8, family="monospace", transform=ax.transAxes)

    ax = axs[1, 1]
    ks = sorted(int(k) for k in res["mle"])
    ax.semilogx(ks, [res["mle"][str(k)]["d2"] for k in ks], "o-", color=C_OK, lw=1.2)
    ax.set_xlabel("k (neighbours)")
    ax.set_ylabel("Levina-Bickel MLE $\\hat d$")
    ax.set_title("(e) MLE dimension vs k")

    ax = axs[1, 2]
    rep = res["rep_screen"]
    ax.hist(np.log10(np.maximum(fav["rel_nearest"], 1e-8)), bins=60,
            color=C_OK if rep["frac_below_thresh"] < 0.02 else C_LOOP, alpha=0.8)
    ax.axvline(np.log10(rep["thresh"]), color=C_LOOP, ls="--", lw=1.2)
    ax.axvline(np.log10(rep["median"]), color=C_GREY, ls=":", lw=1.2)
    ax.set_xlabel(r"$\log_{10}$ (nearest non-adjacent distance $/ d_{typ}$)")
    ax.set_ylabel("points")
    ax.set_title(f"(f) repetition screen: {100*rep['frac_below_thresh']:.1f}% below 1e-3")
    ax.text(0.03, 0.97, f"median {rep['median']:.3f}\np05 {rep['p05']:.3f}",
            va="top", fontsize=7.5, transform=ax.transAxes)

    ax = axs[1, 3]
    per = res["period"]
    ax.plot(fav["lags"], fav["lag_rel"], color=C_OK, lw=0.9)
    if per["period_lag"] and per["coverage"] > 0.1:
        ax.axvline(per["period_lag"], color=C_LOOP, ls="--", lw=1.0)
        ax.annotate(f"k={per['period_lag']}\nprom={per['period_prominence']:.2f}\n"
                    f"cov={100*per['coverage']:.0f}%",
                    xy=(per["period_lag"], per["period_rel"]),
                    xytext=(8, 18), textcoords="offset points",
                    fontsize=7.5, color=C_LOOP)
    ax.set_xlabel("lag k (tokens)")
    ax.set_ylabel(r"mean $\|x_t-x_{t+k}\|/d_{typ}$")
    ax.set_title("(g) lag-distance (loop test)")

    ax = axs[0, 3]
    v0 = res["theiler"][str(res["theiler_default"])]
    txt = (
        f"$D_2$ (valid window)  {v0['d2']:.2f}\n"
        f"  width              {v0['decades']:.2f} dec\n"
        f"  valid              {v0['valid']}\n"
        f"$D_2$ (flat run)       {v0['d2_flat']:.2f}\n"
        f"$D_2$ T4 estimator     "
        f"{res['t4_style']['d2'] if res['t4_style'] else float('nan'):.2f}\n"
        f"TwoNN (MLE)            {res['twonn']['mle']:.2f}\n"
        f"MLE k=10               "
        f"{res['mle'].get('10', {}).get('d2', float('nan')):.2f}\n"
        f"slope range            "
        f"{res['theiler'][str(res['theiler_default'])]['slope_min']:.1f}"
        f"-{res['theiler'][str(res['theiler_default'])]['slope_max']:.1f}\n"
        f"2 log10 N ceiling      {res['slope_ceiling_2log10N']:.1f}\n"
        f"period / coverage      {res['period']['period_lag']} / "
        f"{100*res['period']['coverage']:.0f}%\n"
        f"runtime                {res['total_s']:.1f} s"
    )
    ax.text(0.02, 0.98, txt, va="top", fontsize=8.2, family="monospace",
            transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title("(h) numbers")

    ttl = f"{res['name']}  N={res['n']}  W=0..{max(fav['curves'])}  —  {res['verdict']}"
    if tag:
        ttl = f"[{tag}] " + ttl
    fig.suptitle(ttl, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(results, favs, path, tag=""):
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.2))
    ax = axs[0]
    for r, f in zip(results, favs):
        c = f["curves"][0]
        col = C_LOOP if "loop" in r["verdict"] else C_OK
        ax.loglog(c["eps"], c["C"], lw=1.1, color=col, alpha=0.85,
                  label=f"{r['name']} (N={r['n']})")
    ax.set_xlabel(r"$\epsilon/d_{typ}$")
    ax.set_ylabel(r"$C(\epsilon)$")
    ax.set_title("(a) correlation integral, W=0  (red = loop, blue = healthy)")
    ax.legend(loc="lower right", fontsize=7)

    ax = axs[1]
    for r, f in zip(results, favs):
        c = f["curves"][0]
        col = C_LOOP if "loop" in r["verdict"] else C_OK
        ax.semilogx(c["eps"], c["slope"], lw=1.1, color=col, alpha=0.85,
                    label=r["name"])
    ax.set_xlabel(r"$\epsilon/d_{typ}$")
    ax.set_ylabel(r"$d\log C/d\log\epsilon$")
    ax.set_title("(b) local slope  (red = loop, blue = healthy)")
    ax.set_ylim(0, max(8.0, 1.6 * max(r["slope_ceiling_2log10N"] for r in results)))
    ax.legend(loc="upper right", fontsize=7)

    ax = axs[2]
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    d2 = [r["theiler"][str(r["theiler_default"])]["d2"] for r in results]
    t4 = [r["t4_style"]["d2"] if r["t4_style"] else np.nan for r in results]
    ax.bar(x - 0.2, np.nan_to_num(d2), 0.38, color=C_OK,
           label=r"this pipeline (valid window)")
    ax.bar(x + 0.2, np.nan_to_num(t4), 0.38, color=C_GREY,
           label="T4 estimator (max-$R^2$)")
    for xi, v in zip(x - 0.2, d2):
        ax.text(xi, 0.2, "no\nwindow" if not np.isfinite(v) else f"{v:.1f}",
                ha="center", va="bottom", fontsize=7, color="white",
                fontweight="bold")
    for xi, v in zip(x + 0.2, t4):
        if np.isfinite(v):
            ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel(r"$\hat D_2$")
    ax.set_title("(c) D2: pipeline vs reference estimator")
    ax.legend(loc="upper left", fontsize=7.5)

    ttl = "D2 local pipeline — validation" + (f" [{tag}]" if tag else "")
    fig.suptitle(ttl, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def resolve_inputs(args):
    paths = []
    for p in args.dist:
        paths.append(p)
    for d in args.dir:
        paths.extend(sorted(glob.glob(os.path.join(d, "dist_*.npy"))))
    for p in args.pt:
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.pt"))))
            paths.extend(sorted(glob.glob(os.path.join(p, "*.npz"))))
            paths.extend(sorted(glob.glob(os.path.join(p, "*", "*.pt"))))
        else:
            paths.append(p)
    out, seen = [], set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def name_of(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    m = re.match(r"dist_(.+)_(h_(?:last|first))$", stem)
    if m:
        return m.group(1), m.group(2)
    # pecs layout: <prompt>/last.pt
    if stem in ("last", "first"):
        return os.path.basename(os.path.dirname(path)), "h_" + stem
    return stem, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", nargs="*", default=[], help="distance-matrix .npy files")
    ap.add_argument("--dir", nargs="*", default=[], help="directories with dist_*.npy")
    ap.add_argument("--pt", nargs="*", default=[],
                    help=".pt/.npz trajectories (or directories); distances computed locally")
    ap.add_argument("--theiler", default="0,50,500")
    ap.add_argument("--out", default=os.path.join(ROOT, "notebook_big", "figures"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-n", type=int, default=4000)
    ap.add_argument("--no-t4-style", action="store_true")
    ap.add_argument("--ref-json", default=T4_REF)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    theilers = tuple(int(x) for x in args.theiler.split(",") if x.strip())
    paths = resolve_inputs(args)
    if not paths:
        ap.error("no inputs: pass --dist, --dir or --pt")

    ref = {}
    if args.ref_json and os.path.exists(args.ref_json):
        with open(args.ref_json) as fh:
            ref = json.load(fh)

    # the GPU job's own repetition screen, if this directory has one
    gpu = {}
    for d in args.dir:
        sp = os.path.join(d, "summary.json")
        if os.path.exists(sp):
            with open(sp) as fh:
                for rec in json.load(fh):
                    gpu[rec.get("name")] = rec

    os.makedirs(args.out, exist_ok=True)
    results, favs = [], []
    for p in paths:
        name, state = name_of(p)
        t0 = time.time()
        dim = None
        h = None
        if p.endswith(".npy") and os.path.basename(p).startswith("dist_"):
            D = np.load(p)
        else:
            h = load_states(p)[: args.max_n]
            dim = int(h.shape[1])
            D = dist_matrix(h)
        D = np.asarray(D, dtype=np.float32)
        if D.shape[0] != D.shape[1]:
            print(f"skip {p}: not square {D.shape}", flush=True)
            continue
        res, fav = analyse(D, name, theilers=theilers,
                           t4_style=not args.no_t4_style, states=h)
        res["source"] = os.path.abspath(p)
        res["state"] = state
        res["dim"] = dim
        res["load_s"] = float(time.time() - t0 - res["runtime_s"])
        res["total_s"] = float(time.time() - t0)
        # reference number from the T4 results, matched by name prefix
        res["t4_reference"] = None
        for k, v in ref.items():
            if k.startswith(name) or name.startswith(k):
                res["t4_reference"] = dict(key=k, d2=v.get("raw"), r2=v.get("r2"))
                break
        res["gpu_screen"] = gpu.get(name)
        print(f"{name}: N={res['n']} D2={res['theiler'][str(res['theiler_default'])]['d2']:.2f} "
              f"(valid={res['theiler'][str(res['theiler_default'])]['valid']}, "
              f"{res['theiler'][str(res['theiler_default'])]['decades']:.2f} dec) "
              f"D2_T4style={res['t4_style']['d2'] if res['t4_style'] else float('nan'):.2f} "
              f"TwoNN={res['twonn']['mle']:.2f} "
              f"rep={res['rep_screen']['frac_below_thresh']:.3f} "
              f"[{res['total_s']:.1f}s]  {res['verdict']}", flush=True)

        # histogram data for the figure
        w = min(50, res["n"] - 1)
        m = D.astype(np.float32).copy()
        for k in range(1, w + 1):
            np.fill_diagonal(m[w:], np.inf)
            np.fill_diagonal(m[:, w:], np.inf)
        np.fill_diagonal(m, np.inf)
        fav["rel_nearest"] = (m.min(axis=1) / res["d_typ"]).astype(np.float64)
        results.append(res)
        favs.append(fav)

    if not args.no_plots:
        for res, fav in zip(results, favs):
            suffix = f"_{res['state']}" if res["state"] else ""
            fn = f"d2pipe_{args.tag + '_' if args.tag else ''}{res['name']}{suffix}.png"
            plot_one(res, fav, os.path.join(args.out, fn), tag=args.tag)
        plot_summary(results, favs,
                     os.path.join(args.out,
                                  f"d2pipe_{args.tag + '_' if args.tag else ''}summary.png"),
                     tag=args.tag)

    jn = "d2pipe.json" if not args.tag else f"d2pipe_{args.tag}.json"
    with open(os.path.join(args.out, jn), "w") as fh:
        json.dump(dict(
            tag=args.tag, theiler=list(theilers), n_eps=N_EPS,
            eps_range=[EPS_LO, EPS_HI], tol=TOL, min_decades=MIN_DECADES,
            reference_json=args.ref_json if os.path.exists(args.ref_json) else None,
            trajectories=results), fh, indent=1, default=float)
    print("wrote", os.path.join(args.out, jn), flush=True)


if __name__ == "__main__":
    sys.exit(main())
