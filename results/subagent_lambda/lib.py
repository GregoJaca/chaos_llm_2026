"""Shared library for the lambda(t) (finite-time stretching rate) analysis.

Conventions
-----------
- Token index t is 1-based: array index i = t-1 (i=0 is the first generated token).
- a(t) = ||delta h_{t,l}|| from dh_norm[r, u, l, t-1].
- lambda_k(t) = (1/(2k)) * (log a(t+k) - log a(t-k)), valid for t in [k+1, T-k];
  NaN where either endpoint is 0 (float32 underflow) or NaN.
- Lambda(t) = (1/t) * log(a(t)/a(1)), i.e. array index i: log(a[i]/a[0])/i for i>=1.
- "tail" region: t > 150. "early" region: t in [2, 30].
- Bands: 25-75 percentiles unless stated; bootstrap CIs are 95%.
"""
import os
import numpy as np
from scipy import optimize, stats

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(OUT)                 # .../chaos_llm_2026/results
DATA = os.path.join(RES, "local_data")      # teacher-forced npz data
BIG = RES                                  # free_run files live directly under here

T = 300            # generated tokens per run
K_LOCAL = 5        # default half-width for local rate
TAIL_T0 = 150      # tail region: t > 150
EARLY = (2, 30)    # early-rate window (t range, inclusive)


# ---------------------------------------------------------------- data loading
def load(rel):
    """Load an npz under local_data. rel is like 'w100_layers/a1.npz'."""
    return dict(np.load(os.path.join(DATA, rel.replace("/", os.sep))))


def layers_of(z):
    return list(map(int, z["layers"]))


# ---------------------------------------------------------------- core metrics
def lam_k(a, k=K_LOCAL):
    """Local log-stretching rate for one series a[T] (1-based t = i+1).

    lam[i] corresponds to t=i+1 and uses a[i-k] .. a[i+k].
    """
    a = np.asarray(a, dtype=float)
    T = a.shape[0]
    out = np.full(T, np.nan)
    if T <= 2 * k:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.log(a[2 * k:]) - np.log(a[:-2 * k])
    seg = num / (2.0 * k)
    ok = np.isfinite(seg)
    out[k:T - k][ok] = seg[ok]
    return out


def lam_k_2d(A, k=K_LOCAL):
    """A[n_dir, T] -> [n_dir, T] local rates."""
    return np.vstack([lam_k(A[u], k) for u in range(A.shape[0])])


def cum_rate(a):
    """Lambda(t) = log(a(t)/a(1))/t for one series; NaN after first zero."""
    a = np.asarray(a, dtype=float)
    T = a.shape[0]
    out = np.full(T, np.nan)
    if not (a[0] > 0):
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.log(a / a[0]) / np.arange(1, T + 1)
    out[np.isfinite(vals)] = vals[np.isfinite(vals)]
    # kill everything after the first exact zero (underflow)
    zeros = np.where(a <= 0)[0]
    if len(zeros):
        out[zeros[0]:] = np.nan
    return out


def cum_rate_2d(A):
    return np.vstack([cum_rate(A[u]) for u in range(A.shape[0])])


def tvec(arr):
    """1-based token axis for an array of length T."""
    return np.arange(1, arr.shape[-1] + 1)


def smooth(x, w=9):
    """Centered rolling nan-mean."""
    x = np.asarray(x, float)
    n = len(x)
    h = w // 2
    out = np.copy(x)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        out[i] = np.nanmean(seg) if len(seg) else np.nan
    return out


def med_band(X, lo=25, hi=75):
    """Median + percentile band along axis 0, NaN-aware. Returns (med, lo, hi)."""
    m = np.nanmedian(X, axis=0)
    return m, np.nanpercentile(X, lo, axis=0), np.nanpercentile(X, hi, axis=0)


# ---------------------------------------------------------------- bootstraps
def boot_dirs_curve(A, k, n_boot=2000, seed=0, stat="median"):
    """Bootstrap over directions (axis 0) of the per-t median/mean curve.

    Returns (lo, hi) 95% percentile bands, arrays of length T.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    T = A.shape[1]
    reps = np.empty((n_boot, T))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = A[idx]
        reps[b] = np.nanmedian(sub, axis=0) if stat == "median" else np.nanmean(sub, axis=0)
    return np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)


def boot_dirs_stat(x, n_boot=2000, seed=0):
    """Bootstrap over directions of the scalar mean of per-direction values.

    x: per-direction scalars (NaNs dropped). Returns (mean, lo, hi).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def block_boot_mean(x, block=10, n_boot=2000, seed=0):
    """Moving-block bootstrap CI for the mean of a 1-D series (token axis).

    Circular blocks. Returns (mean, lo, hi).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < block:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(x)
    nblocks = int(np.ceil(n / block))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        reps[b] = x[idx.ravel()].mean()
    m = float(x.mean())
    return m, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def two_way_boot(LA, tmask, block=10, n_boot=1000, seed=0):
    """Joint bootstrap: resample directions AND token blocks within each.

    LA: [n_dir, T] local-rate array; tmask: boolean over T selecting the region.
    Returns (mean, lo, hi) of the region-mean rate.
    """
    rng = np.random.default_rng(seed)
    n = LA.shape[0]
    cols = np.where(tmask)[0]
    L = len(cols)
    if n == 0 or L < block:
        return (np.nan, np.nan, np.nan)
    per = LA[:, cols]
    nblocks = int(np.ceil(L / block))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        di = rng.integers(0, n, n)
        acc = 0.0
        for j in di:
            x = per[j]
            if np.all(~np.isfinite(x)):
                continue
            starts = rng.integers(0, L, nblocks)
            idx = (starts[:, None] + np.arange(block)[None, :]) % L
            vals = x[idx.ravel()]
            vals = vals[np.isfinite(vals)]
            acc += vals.mean() if len(vals) else 0.0
        reps[b] = acc / n
    obs = np.nanmean(per)
    return float(obs), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# ---------------------------------------------------------------- fits
def r2(y, yhat):
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    ss = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / sst) if sst > 0 else np.nan


def aic_bic(y, yhat, npar):
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    if n == 0 or rss <= 0:
        return (np.nan, np.nan)
    return (float(n * np.log(rss / n) + 2 * (npar + 1)),
            float(n * np.log(rss / n) + (npar + 1) * np.log(n)))


def fit_exp_decay(t, y, p0=None):
    """y = c + A*exp(-(t-t0)/tau), t0 = t[0]. Profile grid over tau + polish.

    Returns dict(model='exp', params, yhat, t, r2, aic, bic) or None.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 8:
        return None
    t0 = t[0]
    x = np.clip(t - t0, 0.0, None)
    best = None
    for tau in np.concatenate([np.linspace(0.5, 50, 60), np.linspace(50, 500, 40)]):
        e = np.exp(-x / tau)
        M = np.column_stack([np.ones_like(x), e])
        try:
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        except Exception:
            continue
        rss = float(np.sum((y - M @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, tau, coef)
    if best is None:
        return None
    _, tau, coef = best
    f = lambda tt, c, A, ta: c + A * np.exp(-np.clip(tt - t0, 0, None) / ta)

    def err(p):
        return f(t, *p) - y
    try:
        p, _ = optimize.leastsq(err, [coef[0], coef[1], tau])
        p[2] = abs(p[2])
        if not np.all(np.isfinite(p)):
            p = [coef[0], coef[1], tau]
    except Exception:
        p = [coef[0], coef[1], tau]
    yhat = f(t, *p)
    return dict(model="exp", params=dict(c=float(p[0]), A=float(p[1]), tau=float(p[2])),
                yhat=yhat, t=t, r2=r2(y, yhat),
                aic=aic_bic(y, yhat, 3)[0], bic=aic_bic(y, yhat, 3)[1])


def fit_pow_decay(t, y, p0=None):
    """y = c + A*(t-t0+1)^(-p), bounded p in [0.05, 8]. Profile grid + polish."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 8:
        return None
    t0 = t[0]
    x = np.clip(t - t0 + 1.0, 1.0, None)
    best = None
    for p in np.concatenate([np.linspace(0.05, 2.0, 60), np.linspace(2.0, 8.0, 30)]):
        g = x ** (-p)
        M = np.column_stack([np.ones_like(x), g])
        try:
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        except Exception:
            continue
        rss = float(np.sum((y - M @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p, coef)
    if best is None:
        return None
    _, pg, coef = best
    f = lambda tt, c, A, p: c + A * np.clip(tt - t0 + 1, 1.0, None) ** (-np.clip(p, 0.05, 8.0))

    def err(q):
        return f(t, *q) - y
    try:
        q, _ = optimize.leastsq(err, [coef[0], coef[1], pg])
        q[2] = min(max(q[2], 0.05), 8.0)
        if not np.all(np.isfinite(q)):
            q = [coef[0], coef[1], pg]
    except Exception:
        q = [coef[0], coef[1], pg]
    yhat = f(t, *q)
    return dict(model="pow", params=dict(c=float(q[0]), A=float(q[1]), p=float(q[2])),
                yhat=yhat, t=t, r2=r2(y, yhat),
                aic=aic_bic(y, yhat, 3)[0], bic=aic_bic(y, yhat, 3)[1])


def fit_const(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(y)
    t, y = t[m], y[m]
    c = y.mean()
    yhat = np.full_like(y, c)
    return dict(model="const", params=dict(c=float(c)), yhat=yhat, t=t, r2=r2(y, yhat),
                aic=aic_bic(y, yhat, 1)[0], bic=aic_bic(y, yhat, 1)[1])


def fit_const_ci(y):
    """t-based 95% CI ( Student, NaN-aware) for a constant fit to y."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return (np.nan, np.nan, np.nan)
    m, se = y.mean(), y.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - tcrit * se), float(m + tcrit * se)


def ols_slope(t, y):
    """OLS slope of y on t with 95% CI. Returns (slope, lo, hi, r2)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(t)
    t, y = t[m], y[m]
    n = len(t)
    if n < 3:
        return (np.nan,) * 4
    res = stats.linregress(t, y)
    tcrit = stats.t.ppf(0.975, n - 2)
    hw = tcrit * res.stderr
    return float(res.slope), float(res.slope - hw), float(res.slope + hw), float(res.rvalue ** 2)


def ecdf(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


# ---------------------------------------------------------------- synthetic sanity models
def synth_chaos(T=T, n_dir=20, lam_true=0.035, sigma=0.18, seed=1):
    """Randomly forced *expanding* map: a(t+1) = a(t)*exp(mu_t), mu_t ~ N(lam, sigma).

    Bounded by renormalising a whenever it exceeds 1e3 (as in a real attractor).
    This is what a genuine positive finite-time rate looks like in our estimators.
    """
    rng = np.random.default_rng(seed)
    A = np.empty((n_dir, T))
    for u in range(n_dir):
        mu = rng.normal(lam_true, sigma, T)
        a = np.empty(T)
        x = 1e-3
        loga = np.log(x)
        for i in range(T):
            if i > 0:
                loga += mu[i]
            if loga > np.log(1e3):        # renormalise (attractivity at large a)
                loga = np.log(1e-3) + (loga - np.log(1e3))
            a[i] = np.exp(loga)
        A[u] = a
    return A


def synth_stable_reset(T=T, n_dir=20, lam_true=-0.012, sigma=0.18, w=65, reset=0.55, seed=2):
    """Randomly forced *contracting* linear system with periodic reset (window flush).

    a(t+1) = a(t)*exp(mu_t), and every w tokens the 'window' loses the oldest
    memory: a *= reset (<1). Sustained rate is negative.
    """
    rng = np.random.default_rng(seed)
    A = np.empty((n_dir, T))
    for u in range(n_dir):
        mu = rng.normal(lam_true, sigma, T)
        a = np.empty(T)
        loga = np.log(1e-3)
        for i in range(T):
            if i > 0:
                loga += mu[i]
            if i > 0 and i % w == 0:
                loga += np.log(reset)
            a[i] = np.exp(loga)
        A[u] = a
    return A


def synth_transient(T=T, n_dir=20, lam_asym=-0.012, amp=0.25, tau=18.0, sigma=0.18, seed=3):
    """Transient-growth model: mu_t = lam_asym + amp*exp(-t/tau), multiplicative noise.

    Mimics 'short growth phase then slow decay' -- positive early rate that is
    NOT a sustained exponent.
    """
    rng = np.random.default_rng(seed)
    A = np.empty((n_dir, T))
    tt = np.arange(1, T + 1)
    for u in range(n_dir):
        mu = lam_asym + amp * np.exp(-tt / tau) + rng.normal(0, sigma, T)
        a = np.empty(T)
        loga = np.log(1e-3)
        for i in range(T):
            if i > 0:
                loga += mu[i]
            a[i] = np.exp(loga)
        A[u] = a
    return A


# ---------------------------------------------------------------- misc
def save_json(obj, name):
    import json
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    print("wrote", p)


def pf(x, nd=4):
    """pretty float for titles/json"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if isinstance(x, (int, float, np.floating)) else str(x)
