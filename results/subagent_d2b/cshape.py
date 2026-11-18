"""cshape: shape-focused C(eps) machinery for the D2 shape investigation.

Built on the copied pt_load.py / common.py from subagent_d2.
All pairwise work block-chunked; no NxN matrix stored. Two arithmetic paths:
  - 'f64'   : float64 Gram on (optionally centered) states  -> reference truth
  - 'f32'   : float32 Gram on RAW uncentered states         -> replicates the
              previous pipeline (common.py _block_dist_sq) exactly
Row transforms give metric variants without rewriting the pair engine:
  raw / center / zscore / unit (cosine) / unit_center (centered cosine)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pt_load import load_pt  # noqa: E402
from common import TRAJ, RNG, time_shuffle, iaaft  # noqa: E402

LOOPS = {'cyber': 53, 'future': 74, 'great': 69}


def load_traj(name):
    a = load_pt(TRAJ[name])
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    return a


# ---------------------------------------------------------------- transforms

def transform(X, mode):
    X64 = np.asarray(X, dtype=np.float64)
    if mode == 'raw':
        return X64
    if mode == 'center':
        return X64 - X64.mean(0, keepdims=True)
    if mode == 'zscore':
        return (X64 - X64.mean(0, keepdims=True)) / (X64.std(0, keepdims=True) + 1e-12)
    if mode == 'unit':
        n = np.linalg.norm(X64, axis=1, keepdims=True)
        return X64 / np.maximum(n, 1e-30)
    if mode == 'unit_center':
        Z = X64 - X64.mean(0, keepdims=True)
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(n, 1e-30)
    raise ValueError(mode)


# ---------------------------------------------------------------- pair engine

def pair_counts(Xsub, eps_grid, theiler=0, times=None, block=192,
                arith='f64', include_self=False):
    """Cumulative pair counts at eps_grid.

    Xsub rows are the points (already transformed). times = original time
    index of each row (used for the Theiler exclusion). include_self counts
    ordered pairs (i,j) including i==j over N^2 (the naive cdist notebook
    convention); otherwise unordered i<j pairs.
    Returns (cum_counts int64[len(eps)], npairs).
    """
    X = np.asarray(Xsub)
    N = len(X)
    e2 = np.asarray(eps_grid, dtype=np.float64) ** 2
    counts = np.zeros(len(e2), dtype=np.int64)
    if arith == 'f64':
        Xa = X.astype(np.float64)
        sq_b = (Xa ** 2).sum(1)[None, :]
    else:  # replicate common.py: float32 BLAS Gram on raw fp32 rows
        Xa = X.astype(np.float32)
        sq_b = (Xa.astype(np.float64) ** 2).sum(1)[None, :]
    if times is None:
        times = np.arange(N)
    for a in range(0, N, block):
        rows = np.arange(a, min(a + block, N))
        A = Xa[rows]
        if arith == 'f64':
            sq_a = (A ** 2).sum(1)[:, None]
            G = A @ Xa.T
        else:
            sq_a = (A.astype(np.float64) ** 2).sum(1)[:, None]
            G = (A @ Xa.T).astype(np.float64)
        D = sq_a + sq_b - 2.0 * G
        np.maximum(D, 0.0, out=D)
        if include_self:
            mask = np.ones(D.shape, dtype=bool)  # naive cdist convention: N^2 ordered pairs
        else:
            tj = times[None, :]
            ti = times[rows][:, None]
            mask = (tj > ti) & ((tj - ti) > theiler)
        vals = D[mask]
        pos = np.searchsorted(e2, vals)
        counts += np.bincount(pos, minlength=len(e2) + 1)[:len(e2)]
    if include_self:
        npairs = N * N
    else:
        tt = np.asarray(times)
        npairs = 0
        for i in range(N):
            npairs += int(np.sum((tt > tt[i]) & ((tt - tt[i]) > theiler)))
    return np.cumsum(counts), npairs


def c_curve(X, eps, theiler=0, times=None, arith='f64', mode='center',
            include_self=False, block=192, idx=None):
    """C(eps) with transform `mode`; idx = original row indices to subsample."""
    Xt = transform(X, mode)
    if idx is not None:
        Xt = Xt[idx]
        tm = (np.asarray(idx) if times is None else np.asarray(times)[idx])
    else:
        tm = times
    counts, npairs = pair_counts(Xt, eps, theiler, tm, block, arith, include_self)
    return counts / npairs, npairs, counts


def median_dist(X, mode='center', n=2000, rng=None):
    rng = rng or RNG
    Xt = transform(X, mode)
    idx = np.sort(rng.choice(len(Xt), size=min(n, len(Xt)), replace=False))
    A = Xt[idx].astype(np.float64)
    G = A @ A.T
    sq = (A ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * G
    np.maximum(D, 0, out=D)
    iu = np.triu_indices(len(idx), 1)
    d = np.sqrt(D[iu])
    d = d[d > 0]
    return float(np.median(d))


# ---------------------------------------------------------------- slopes etc.

def slope_curve(eps, C, win=7):
    """Rolling LSQ slope of log C vs log eps over `win` grid points.
    Returns (eps_mid, slope, count_per_point)."""
    loge = np.log(eps)
    ok = C > 0
    out_e, out_s = [], []
    half = win // 2
    for i in range(half, len(eps) - half):
        sl = slice(i - half, i + half + 1)
        if not ok[sl].all():
            out_e.append(np.nan)
            out_s.append(np.nan)
            continue
        x = loge[sl]
        y = np.log(C[sl])
        xm = x.mean()
        denom = ((x - xm) ** 2).sum()
        if denom <= 0:
            out_e.append(np.nan)
            out_s.append(np.nan)
            continue
        b = ((x - xm) * (y - y.mean())).sum() / denom
        out_e.append(np.exp(loge[sl].mean()))
        out_s.append(float(b))
    return np.array(out_e), np.array(out_s)


def fit_d2(eps, C, lo=0.1, hi=0.9):
    m = (C >= lo) & (C <= hi) & (C > 0)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    x = np.log(eps[m]); y = np.log(C[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = np.sum((y - A @ coef) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(coef[0]), float(1 - ss / sst if sst > 0 else np.nan), int(m.sum())


def plateau_scan(eps_mid, slope, C_mid, N, tol_rel=0.15, floor_tol=0.05,
                 min_dec=1.0, c_lo=None, c_hi=0.5, er_cap=True):
    """Longest contiguous run where slope stays within tol_rel (plus a small
    absolute floor) of its value at run start. Only counts runs whose C lies in
    [max(c_lo, ...), c_hi] and (if er_cap) slope <= 2*log10(N).
    Returns list of runs sorted by width (decades), best first."""
    er = 2.0 * np.log10(max(N, 10))
    if c_lo is None:
        c_lo = 10.0 / N
    good = (np.isfinite(slope)) & (C_mid >= c_lo) & (C_mid <= c_hi)
    if er_cap:
        good = good & (slope <= er)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = idx[0]
    prev = idx[0]
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
        out.append({'i0': int(a), 'i1': int(b), 'slope': float(np.median(slope[a:b + 1])),
                    'width_dec': float(width),
                    'eps0': float(eps_mid[a]), 'eps1': float(eps_mid[b]),
                    'C0': float(C_mid[a]), 'C1': float(C_mid[b])})
    out.sort(key=lambda r: -r['width_dec'])
    return out


# ---------------------------------------------------------------- scalars

def observables(X):
    """1-D series a notebook would delay-embed: state norm, PC1 score,
    distance to previous state. All returned at full length where defined."""
    Xc = X.astype(np.float64)
    mu = Xc.mean(0, keepdims=True)
    Z = Xc - mu
    C = (Z.T @ Z) / max(1, len(Z) - 1)
    w, V = np.linalg.eigh(C)
    V = V[:, ::-1]
    pc1 = Z @ V[:, 0]
    nrm = np.linalg.norm(Xc, axis=1)
    step = np.zeros(len(Xc))
    step[1:] = np.linalg.norm(np.diff(Xc, axis=0), axis=1)
    return {'norm': nrm, 'pc1': pc1, 'stepdist': step}


def embed(s, m, tau, t0=0):
    """Delay embedding [N-(m-1)tau, m]; returns (vecs, times)."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    w = (m - 1) * tau
    if n <= w:
        return np.empty((0, m)), np.empty(0)
    idx = np.arange(n - w)
    cols = [idx + k * tau for k in range(m)]
    vecs = np.stack([s[c] for c in cols], axis=1)
    return vecs, idx + t0


def acf(s, max_lag=20):
    s = np.asarray(s, dtype=np.float64)
    s = s - s.mean()
    v = (s ** 2).sum()
    return [float((s[:-k] * s[k:]).sum() / v) if k > 0 else 1.0 for k in range(max_lag + 1)]


def phase_rand_1d(x, rng=None):
    rng = rng or RNG
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    F = np.fft.rfft(x - x.mean())
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(F)))
    ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    y = np.fft.irfft(F * ph, n=n)
    return y * x.std() / (y.std() + 1e-12) + x.mean()


# ---------------------------------------------------------------- references

def henon(n=4096, a=1.4, b=0.3, burn=2000, rng=None):
    rng = rng or np.random.default_rng(7)
    x, y = 0.1 * rng.standard_normal(), 0.1 * rng.standard_normal()
    out = np.empty((n, 2))
    for i in range(-burn, n):
        x, y = 1.0 - a * x * x + b * y, x
        if i >= 0:
            out[i] = (x, y)
    return out


def lorenz(n_steps=60000, dt=0.01, s=10.0, r=28.0, bb=8.0 / 3.0, x0=(1.0, 1.0, 20.0)):
    out = np.empty((n_steps, 3))
    v = np.asarray(x0, dtype=np.float64)
    for i in range(n_steps):
        dv = np.array([s * (v[1] - v[0]), v[0] * (r - v[2]) - v[1],
                       v[0] * v[1] - bb * v[2]])
        v2 = v + 0.5 * dt * dv
        dv2 = np.array([s * (v2[1] - v2[0]), v2[0] * (r - v2[2]) - v2[1],
                        v2[0] * v2[1] - bb * v2[2]])
        v = v + dt * dv2
        out[i] = v
    return out


def ar1(n, rho, rng=None):
    rng = rng or RNG
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return y


# ---------------------------------------------------------------- misc

def dist_stats(X, mode='center', n=2000, rng=None):
    rng = rng or RNG
    Xt = transform(X, mode)
    idx = np.sort(rng.choice(len(Xt), size=min(n, len(Xt)), replace=False))
    A = Xt[idx].astype(np.float64)
    G = A @ A.T
    sq = (A ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    iu = np.triu_indices(len(idx), 1)
    d = np.sqrt(D[iu])
    nz = d[d > 0]
    return {
        'q10': float(np.quantile(d, 0.10)), 'q25': float(np.quantile(d, 0.25)),
        'med': float(np.median(d)), 'q75': float(np.quantile(d, 0.75)),
        'q90': float(np.quantile(d, 0.90)), 'max': float(d.max()),
        'min_nonzero': float(nz.min()) if nz.size else 0.0,
        'iqr_over_med': float((np.quantile(d, 0.75) - np.quantile(d, 0.25)) / np.median(d)),
        'frac_zero': float((d == 0).mean()),
    }


def jsonable(o):
    import math
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.floating):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    return o
