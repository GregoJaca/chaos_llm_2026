"""Fitting library: log-space least squares for response curves a(t).

All full-curve models are fitted on log a(t) (natural log) over t in [1, T-1].
Model list (task 1):
  M1 exp        : log y = a + b*t                       (2 params)
  M2 power      : log y = a + b*log t                   (2)
  M3 exp+floor  : y = A*exp(-t/tau)+C  (fit in log y)   (3)
  M4 lognormal  : log y = logA - (log t - mu)^2/(2 s^2) (3)
  M5 pw-logt    : log y = a + b*min(x,x*) + c*max(0,x-x*), x=log t (4, cont.)
  M6 two-exp    : log y = a + b*min(t,t*)+c*max(0,t-t*) (hump; bonus model) (4)
Segment fits:
  rise   : log y = a + lam*t on t in [1, t_peak]
  decay  : exp  log y = a + s*t          on t in [t_peak+5, T-1]
           exp+floor y = A e^{-t/tau}+C
           power log y = a + b log t
"""
import numpy as np
from scipy.optimize import least_squares

TINY = 1e-300

def _ols(X, y):
    """OLS with covariance. X: [n,p]. Returns beta, cov, rss, r2, n."""
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - p
    s2 = rss / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(X.T @ X)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else np.nan
    return beta, cov, rss, r2, n

def _ic(rss, n, k):
    """AIC/BIC with sigma counted: k_tot = k+1."""
    kt = k + 1
    aic = n * np.log(max(rss, TINY) / n) + 2 * kt
    bic = n * np.log(max(rss, TINY) / n) + kt * np.log(n)
    return aic, bic

def _se_diag(cov):
    return np.sqrt(np.clip(np.diag(cov), 0, None))

def fit_exp(t, ly):
    X = np.column_stack([np.ones_like(t), t])
    beta, cov, rss, r2, n = _ols(X, ly)
    aic, bic = _ic(rss, n, 2)
    return dict(model="exp", params=dict(A=float(np.exp(beta[0])), a0=float(beta[0]), lam=float(beta[1])),
                se=dict(a0=float(_se_diag(cov)[0]), lam=float(_se_diag(cov)[1])),
                k=2, rss=rss, r2=r2, n=n, aic=aic, bic=bic)

def fit_power(t, ly):
    X = np.column_stack([np.ones_like(t), np.log(t)])
    beta, cov, rss, r2, n = _ols(X, ly)
    aic, bic = _ic(rss, n, 2)
    return dict(model="power", params=dict(a0=float(beta[0]), b=float(beta[1])),
                se=dict(a0=float(_se_diag(cov)[0]), b=float(_se_diag(cov)[1])),
                k=2, rss=rss, r2=r2, n=n, aic=aic, bic=bic)

def fit_lognormal(t, ly):
    # log y = c - (x - mu)^2/(2 s^2), x = log t  -> linear in (c, mu/s^2, 1/s^2)... solve directly
    x = np.log(t)
    # y = c - (x^2 - 2 mu x + mu^2)/(2 s^2): quadratic in x. Fit parabola by OLS on [1, x, x^2].
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    beta, cov, rss, r2, n = _ols(X, ly)
    c, b1, b2 = beta
    # b2 = -1/(2 s^2) -> s2 = -1/(2 b2) requires b2<0
    if b2 >= 0:
        return dict(model="lognormal", params=None, se=None, k=3, rss=rss, r2=r2, n=n,
                    aic=np.inf, bic=np.inf, failed="degenerate (b2>=0, no hump)")
    s2 = -1.0 / (2 * b2)
    mu = -b1 / (2 * b2)
    A = float(np.exp(c + mu ** 2 / (2 * s2)))
    aic, bic = _ic(rss, n, 3)
    return dict(model="lognormal",
                params=dict(A=A, mu=float(mu), sigma=float(np.sqrt(s2))),
                se=None,  # propagate via bootstrap at aggregate level
                k=3, rss=rss, r2=r2, n=n, aic=aic, bic=bic)

def fit_expfloor(t, y, t0=None):
    """y = A*exp(-(t-t0)/tau)+C, fitted in log y. t0 fixed at min(t) for conditioning."""
    t0 = float(t.min()) if t0 is None else t0
    tt = t - t0
    ly = np.log(np.maximum(y, TINY))
    ymax = float(np.max(y))
    def resid(p):
        A, tau, C = p
        v = A * np.exp(-tt / tau) + C
        return np.log(np.maximum(v, TINY)) - ly
    best = None
    for A0 in (ymax, ymax / 2):
        for tau0 in (10.0, 50.0):
            for C0 in (ymax / 10, ymax / 100):
                try:
                    r = least_squares(resid, [A0, tau0, C0], method="trf",
                                       bounds=([1e-12, 0.2, 0.0], [np.inf, 1e4, np.inf]),
                                       max_nfev=200)
                    if best is None or r.cost < best.cost:
                        best = r
                except Exception:
                    pass
    if best is None:
        return dict(model="exp+floor", params=None, se=None, k=3, rss=np.nan, r2=np.nan,
                    n=len(t), aic=np.inf, bic=np.inf, failed="optimizer")
    rss = float(2 * best.cost)
    tss = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    n = len(t)
    aic, bic = _ic(rss, n, 3)
    # SEs from J^T J
    try:
        J = best.jac
        cov = np.linalg.inv(J.T @ J) * rss / (n - 3)
        se = [float(np.sqrt(max(cov[i, i], 0))) for i in range(3)]
    except Exception:
        se = None
    A, tau, C = best.x
    return dict(model="exp+floor",
                params=dict(A=float(A), tau=float(tau), C=float(C)),
                se=dict(A=se[0] if se else None, tau=se[1] if se else None, C=se[2] if se else None),
                k=3, rss=rss, r2=r2, n=n, aic=aic, bic=bic)

def _piecewise_grid(t, ly, xmode, k=4, ngrid=25):
    x = np.log(t) if xmode == "logt" else t.astype(float)
    lo, hi = float(x.min()), float(x.max())
    cands = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), ngrid)
    best = None
    for xs in cands:
        X = np.column_stack([np.ones_like(x), np.minimum(x, xs), np.maximum(0, x - xs)])
        try:
            beta, cov, rss, r2, n = _ols(X, ly)
        except Exception:
            continue
        if best is None or rss < best["rss"]:
            best = dict(xs=float(xs), beta=beta, cov=cov, rss=rss, r2=r2, n=n)
    if best is None:
        return None
    aic, bic = _ic(best["rss"], best["n"], k)
    b1, b2 = best["beta"][1], best["beta"][2]
    se = _se_diag(best["cov"])
    out = dict(k=k, rss=best["rss"], r2=best["r2"], n=best["n"], aic=aic, bic=bic,
               se=dict(a0=float(se[0]), b1=float(se[1]), b2=float(se[2]))
               if np.all(np.isfinite(se)) else None)
    if xmode == "logt":
        out.update(model="pw-logt", params=dict(a0=float(best["beta"][0]), b1=float(b1), b2=float(b2),
                                                tstar=float(np.exp(best["xs"]))))
    else:
        out.update(model="two-exp", params=dict(a0=float(best["beta"][0]), lam1=float(b1), lam2=float(b2),
                                                tstar=float(best["xs"])))
    return out

def fit_pw_logt(t, ly):
    r = _piecewise_grid(t, ly, "logt")
    if r is None:
        return dict(model="pw-logt", params=None, se=None, k=4, rss=np.nan, r2=np.nan,
                    n=len(t), aic=np.inf, bic=np.inf, failed="no grid")
    return r

def fit_two_exp(t, ly):
    r = _piecewise_grid(t, ly, "t")
    if r is None:
        return dict(model="two-exp", params=None, se=None, k=4, rss=np.nan, r2=np.nan,
                    n=len(t), aic=np.inf, bic=np.inf, failed="no grid")
    return r

FULL_MODELS = [("exp", fit_exp), ("power", fit_power), ("exp+floor", fit_expfloor),
               ("lognormal", fit_lognormal), ("pw-logt", fit_pw_logt), ("two-exp", fit_two_exp)]

def fit_all_full(t, y):
    """Fit all six full-curve models. y>0 required; zeros masked out by caller."""
    ly = np.log(y)
    out = {}
    out["exp"] = fit_exp(t, ly)
    out["power"] = fit_power(t, ly)
    out["lognormal"] = fit_lognormal(t, ly)
    out["pw-logt"] = fit_pw_logt(t, ly)
    out["two-exp"] = fit_two_exp(t, ly)
    out["exp+floor"] = fit_expfloor(t, y)
    return out

# ---------------- segment fits ----------------

def smooth_log(y, w=7):
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")[:len(y)]

def find_peak(t, y, w=7, tmin=3):
    lys = smooth_log(y, w)
    lys[:tmin] = np.nan
    return int(np.nanargmax(lys))

def fit_segments(t, y, t_peak, skip_after=5):
    """Rise on [1, t_peak] exp; decay on [t_peak+skip, end]: exp, exp+floor, power."""
    out = {}
    m_r = (t >= 1) & (t <= t_peak)
    if m_r.sum() >= 6:
        out["rise"] = fit_exp(t[m_r], np.log(y[m_r]))
        out["rise"]["model"] = "rise-exp"
    m_d = (t >= t_peak + skip_after) & (t <= t.max())
    if m_d.sum() >= 10:
        td, yd = t[m_d], y[m_d]
        out["decay-exp"] = fit_exp(td, np.log(yd))
        out["decay-exp"]["model"] = "decay-exp"
        out["decay-expfloor"] = fit_expfloor(td, yd)
        out["decay-power"] = fit_power(td, np.log(yd))
        out["decay-power"]["model"] = "decay-power"
    return out

def boot_ci(samples, stat=np.median, nboot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of a statistic over axis 0 (directions)."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, float)
    n = samples.shape[0]
    idx = rng.integers(0, n, size=(nboot, n))
    if samples.ndim == 1:
        vals = np.array([stat(samples[i]) for i in idx])
    else:
        # stat over axis 0 for each column
        vals = np.empty((nboot,) + samples.shape[1:])
        for b in range(nboot):
            vals[b] = stat(samples[idx[b]], axis=0)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi, vals

def boot_stat_ci(samples, stat=np.median, nboot=2000, seed=0, alpha=0.05):
    lo, hi, _ = boot_ci(samples, stat=stat, nboot=nboot, seed=seed, alpha=alpha)
    pt = stat(samples, axis=0) if np.asarray(samples).ndim > 1 else stat(samples)
    return pt, lo, hi
