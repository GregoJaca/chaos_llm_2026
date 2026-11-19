"""Fast, principled functional-form identification for 1-D curves.

Given x, y, try a library of forms (power laws, exponentials, stretched exponentials,
saturating growth, logs, polynomials, rationals, Hill) *with and without offsets/shifts*,
rank them by BIC, and return parameters with standard errors.

Design rules
------------
* Log-domain fits.  For pure exponential / power law and strictly positive y, the model is
  fitted by ordinary least squares in log space, i.e. it minimises squared *relative* error.
  This is what you want when the curve spans decades; fitting y = a e^{bx} directly would let
  the largest values dominate.  The fit is scored in the same domain that was used to fit it,
  so the comparison is fair (documented in `domain`).
* Offset models cannot be log-transformed; they are fitted by nonlinear least squares in the
  native domain.  When a curve needs an offset, the log-domain pure forms lose on BIC -- that
  is the intended behaviour, not a bug.
* BIC = n ln(RSS/n) + k ln n, with k the free-parameter count (including sigma).
* Every returned parameter has a standard error from the Gauss-Newton covariance.

Usage
-----
    from chaos_llm.curvefit import fit_curve
    r = fit_curve(x, y)          # r.best, r.table
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares


# ----------------------------------------------------------------------------- helpers
def _rss(y, f):
    return float(np.sum((y - f) ** 2))


def _bic(rss, n, k):
    if rss <= 0:
        rss = 1e-300
    return n * np.log(rss / n) + k * np.log(n)


def _r2(y, f):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - f) ** 2) / ss) if ss > 0 else float("nan")


def _cov_se(J, resid, k):
    """Gauss-Newton parameter standard errors."""
    n = len(resid)
    dof = max(n - k, 1)
    s2 = float(np.sum(resid ** 2) / dof)
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        return np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        return np.full(k, np.nan)


@dataclass
class Fit:
    name: str
    formula: str
    params: np.ndarray
    se: np.ndarray
    r2: float
    r2_fit: float
    bic: float
    aic: float
    k: int
    n: int
    domain: str
    pnames: List[str] = field(default_factory=list)
    ok: bool = True
    msg: str = ""
    predict: Optional[Callable] = None

    def line(self, prec: int = 4) -> str:
        parts = []
        for nm, v, s in zip(self.pnames, self.params, self.se):
            parts.append(f"{nm}={v:.{prec}g}±{s:.2g}")
        return ", ".join(parts)


# ----------------------------------------------------------------------------- library
def _nls(x, y, fn, p0s, bounds, names, formula, domain="native", weights=None):
    """Multi-start bounded least squares.  fn(p, x) -> yhat."""
    best = None
    for p0 in p0s:
        try:
            sol = least_squares(lambda p: (fn(p, x) - y) / (weights if weights is not None else 1.0),
                                p0, bounds=bounds, max_nfev=4000, method="trf")
        except Exception:
            continue
        if not np.all(np.isfinite(sol.fun)):
            continue
        cost = float(np.sum(sol.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, sol)
    if best is None:
        return None
    _, sol = best
    p = sol.x
    f = fn(p, x)
    res = f - y
    se = _cov_se(sol.jac, res, len(p))
    return p, se, f, res


def _poly(x, y, deg, names, formula):
    k = deg + 1
    V = np.vander(x, k, increasing=True)
    p, *_ = np.linalg.lstsq(V, y, rcond=None)
    f = V @ p
    res = f - y
    se = _cov_se(V, res, k)
    return p, se, f, res


def _families():
    """name -> (builder, n_params, needs_positive_y, domain)."""
    fams = {}

    # ---- algebraic / polynomial
    fams["linear"] = ("a + b*x", 2, False, "native")
    fams["quadratic"] = ("a + b*x + c*x^2", 3, False, "native")
    fams["cubic"] = ("a + b*x + c*x^2 + d*x^3", 4, False, "native")

    # ---- logarithmic
    fams["log"] = ("a + b*log(x)", 2, False, "native")
    fams["log+linear"] = ("a + b*log(x) + c*x", 3, False, "native")

    # ---- inverse
    fams["inverse"] = ("a + b/x", 2, False, "native")
    fams["rational"] = ("a/(1 + b*x)", 2, False, "native")

    # ---- power laws
    fams["power"] = ("a*x^b", 2, True, "loglog")          # OLS in log-log
    fams["power+offset"] = ("c + a*x^b", 3, False, "native")
    fams["power+offset+shift"] = ("c + a*(x-x0)^b", 4, False, "native")

    # ---- exponentials
    fams["exponential"] = ("a*exp(b*x)", 2, True, "semilog")   # OLS in semilog
    fams["exp+offset"] = ("c + a*exp(b*x)", 3, False, "native")
    fams["exp+offset+shift"] = ("c + a*exp(b*(x-x0))", 4, False, "native")

    # ---- saturating / stretched
    fams["saturating"] = ("c + a*(1 - exp(-x/tau))", 3, False, "native")
    fams["stretched_decay"] = ("c + a*exp(-(x/tau)^beta)", 4, False, "native")
    fams["stretched_rise"] = ("c + a*(1 - exp(-(x/tau)^beta))", 4, False, "native")
    fams["gaussian"] = ("c + a*exp(-(x/tau)^2)", 3, False, "native")
    fams["hill"] = ("c + a*x^b/(k^b + x^b)", 4, False, "native")
    fams["loglogistic"] = ("c + a/(1+(x/k)^b)", 4, False, "native")
    fams["loglogistic0"] = ("a/(1+(x/k)^b)", 3, False, "native")
    fams["rise_decay"] = ("c + a*(1-exp(-x/t1))*exp(-x/t2)", 4, False, "native")
    fams["two_exponential"] = ("c + a1*exp(-x/t1) + a2*exp(x/t2)", 5, False, "native")
    return fams


# ----------------------------------------------------------------------------- fit
def _build(name, x, y):
    """Return (param_names, objective, p0 list, bounds, domain, formula_string)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, xM = float(np.min(x)), float(np.max(x))
    ym, yM = float(np.min(y)), float(np.max(y))
    span = max(xM - xm, 1e-12)
    yspan = max(yM - ym, 1e-12)

    pos_x = x[x > 0]
    if name == "linear":
        return (["a", "b"], lambda p, xx: p[0] + p[1] * xx,
                [[ym, 0], [ym, yspan / span]],
                ([-np.inf, -np.inf], [np.inf, np.inf]), "native", "a + b*x")
    if name == "quadratic":
        return (["a", "b", "c"], lambda p, xx: p[0] + p[1] * xx + p[2] * xx ** 2,
                [[ym, 0, 0], [ym, yspan / span, 0], [yM, -yspan / span, yspan / span ** 2]],
                ([-np.inf] * 3, [np.inf] * 3), "native", "a + b*x + c*x^2")
    if name == "cubic":
        return (["a", "b", "c", "d"],
                lambda p, xx: p[0] + p[1] * xx + p[2] * xx ** 2 + p[3] * xx ** 3,
                [[ym, 0, 0, 0], [yM, yspan / span, 0, 0], [ym, 0, 0, yspan / span ** 3]],
                ([-np.inf] * 4, [np.inf] * 4), "native", "a + b*x + c*x^2 + d*x^3")
    if name == "log":
        lx = np.log(np.maximum(pos_x, 1e-12))
        sl = (y[-1] - y[0]) / max(lx[-1] - lx[0], 1e-12) if len(lx) > 1 else 0.0
        return (["a", "b"], lambda p, xx: p[0] + p[1] * np.log(np.maximum(xx, 1e-12)),
                [[y[0], sl], [np.median(y), 0], [y[-1], -sl]],
                ([-np.inf, -np.inf], [np.inf, np.inf]), "native", "a + b*log(x)")
    if name == "log+linear":
        return (["a", "b", "c"],
                lambda p, xx: p[0] + p[1] * np.log(np.maximum(xx, 1e-12)) + p[2] * xx,
                [[y[0], 0, 0], [y[0], 1, 0], [y[0], 0, 0.01]],
                ([-np.inf] * 3, [np.inf] * 3), "native", "a + b*log(x) + c*x")
    if name == "inverse":
        return (["a", "b"], lambda p, xx: p[0] + p[1] / np.maximum(xx, 1e-12),
                [[y[-1], (y[0] - y[-1]) * xm], [ym, 0], [yM, -yspan * xm]],
                ([-np.inf, -np.inf], [np.inf, np.inf]), "native", "a + b/x")
    if name == "rational":
        return (["a", "b"], lambda p, xx: p[0] / (1 + p[1] * xx),
                [[y[0], 1 / span], [y[0] * 2, 0.1 / span], [y[0], 10 / span]],
                ([-np.inf, 0], [np.inf, np.inf]), "native", "a/(1 + b*x)")
    if name == "power":
        return (["a", "b"], None, None, ([-np.inf, -np.inf], [np.inf, np.inf]), "loglog", "a*x^b")
    if name == "power+offset":
        return (["c", "a", "b"], lambda p, xx: p[0] + p[1] * np.maximum(xx, 1e-12) ** p[2],
                [[0, y[0], -1], [0, y[0], -0.5], [ym, yspan, 1], [ym, y[0] - ym, -1],
                 [0.5 * ym, yspan, 0.5]],
                ([-np.inf, -np.inf, -5], [np.inf, np.inf, 5]), "native", "c + a*x^b")
    if name == "power+offset+shift":
        return (["c", "a", "b", "x0"],
                lambda p, xx: p[0] + p[1] * np.maximum(xx - p[3], 1e-12) ** p[2],
                [[0, y[0], -1, 0], [ym, yspan, 1, 0], [0, y[0] - ym, -1, 0.1 * span],
                 [0, y[0], 0.5, 0]],
                ([-np.inf, -np.inf, -5, -0.9 * span], [np.inf, np.inf, 5, 0.9 * span]),
                "native", "c + a*(x-x0)^b")
    if name == "exponential":
        return (["a", "b"], None, None, ([-np.inf, -np.inf], [np.inf, np.inf]), "semilog", "a*exp(b*x)")
    if name == "exp+offset":
        return (["c", "a", "b"], lambda p, xx: p[0] + p[1] * np.exp(np.clip(p[2] * xx, -50, 50)),
                [[0, y[0], -1 / span], [ym, y[0] - ym, -1 / span], [y[-1], y[0] - y[-1], -3 / span],
                 [0.5 * ym, yspan, 0.5 / span]],
                ([-np.inf, -np.inf, -50], [np.inf, np.inf, 50]), "native", "c + a*exp(b*x)")
    if name == "exp+offset+shift":
        return (["c", "a", "b", "x0"],
                lambda p, xx: p[0] + p[1] * np.exp(np.clip(p[2] * (xx - p[3]), -50, 50)),
                [[0, y[0], -1 / span, 0], [ym, y[0] - ym, -1 / span, 0.1 * span],
                 [0, y[0], 1 / span, 0]],
                ([-np.inf, -np.inf, -50, -0.9 * span], [np.inf, np.inf, 50, 0.9 * span]),
                "native", "c + a*exp(b*(x-x0))")
    if name == "saturating":
        return (["c", "a", "tau"],
                lambda p, xx: p[0] + p[1] * (1 - np.exp(-np.maximum(xx, 0) / max(p[2], 1e-9))),
                [[ym, yspan, span], [ym, yspan, 0.3 * span], [y[0], y[-1] - y[0], 0.1 * span]],
                ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1e6]), "native",
                "c + a*(1 - exp(-x/tau))")
    if name == "stretched_decay":
        return (["c", "a", "tau", "beta"],
                lambda p, xx: p[0] + p[1] * np.exp(-(np.maximum(xx, 0) / max(p[2], 1e-9)) ** max(p[3], 1e-3)),
                [[0, y[0], 0.5 * span, 1], [ym, yspan, 0.2 * span, 0.5], [0, y[0], 2 * span, 2],
                 [y[-1], y[0] - y[-1], 0.1 * span, 1]],
                ([-np.inf, -np.inf, 1e-6, 1e-3], [np.inf, np.inf, 1e6, 20]), "native",
                "c + a*exp(-(x/tau)^beta)")
    if name == "stretched_rise":
        return (["c", "a", "tau", "beta"],
                lambda p, xx: p[0] + p[1] * (1 - np.exp(-(np.maximum(xx, 0) / max(p[2], 1e-9)) ** max(p[3], 1e-3))),
                [[ym, 0, 0.5 * span, 1], [y[0], yspan, 0.3 * span, 0.7], [ym, yspan, 0.1 * span, 2]],
                ([-np.inf, -np.inf, 1e-6, 1e-3], [np.inf, np.inf, 1e6, 20]), "native",
                "c + a*(1 - exp(-(x/tau)^beta))")
    if name == "gaussian":
        return (["c", "a", "tau"],
                lambda p, xx: p[0] + p[1] * np.exp(-(np.maximum(xx, 0) / max(p[2], 1e-9)) ** 2),
                [[0, y[0], 0.5 * span], [ym, yspan, 0.2 * span], [0, y[0], 2 * span]],
                ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1e6]), "native",
                "c + a*exp(-(x/tau)^2)")
    if name == "hill":
        return (["c", "a", "b", "k"],
                lambda p, xx: p[0] + p[1] * xx ** max(p[2], 1e-3) /
                (max(p[3], 1e-9) ** max(p[2], 1e-3) + xx ** max(p[2], 1e-3)),
                [[0, y[0], 1, 0.5 * span], [ym, yspan, 1, span], [0, y[0], 2, 0.2 * span]],
                ([-np.inf, -np.inf, 1e-3, 1e-9], [np.inf, np.inf, 20, 1e6]), "native",
                "c + a*x^b/(k^b + x^b)")
    if name == "loglogistic":
        return (["c", "a", "k", "b"],
                lambda p, xx: p[0] + p[1] / (1 + (np.maximum(xx, 1e-12) / max(p[2], 1e-9)) ** max(p[3], 1e-3)),
                [[0, 1, np.median(x), 1], [0, 1, 0.5 * span, 2], [0, y[0], 2 * span, 0.5]],
                ([-np.inf, -np.inf, 1e-9, 1e-3], [np.inf, np.inf, 1e6, 20]), "native",
                "c + a/(1+(x/k)^b)")
    if name == "loglogistic0":
        return (["a", "k", "b"],
                lambda p, xx: p[0] / (1 + (np.maximum(xx, 1e-12) / max(p[1], 1e-9)) ** max(p[2], 1e-3)),
                [[1, np.median(x), 1], [1, np.median(x), 2], [1, 0.5 * span, 0.5], [y[0], 2 * span, 1]],
                ([0, 1e-9, 1e-3], [np.inf, 1e6, 20]), "native",
                "a/(1+(x/k)^b)")
    if name == "rise_decay":
        return (["c", "a", "t1", "t2"],
                lambda p, xx: p[0] + p[1] * (1 - np.exp(-np.maximum(xx, 0) / max(p[2], 1e-9)))
                * np.exp(-np.maximum(xx, 0) / max(p[3], 1e-9)),
                [[0, yM - ym, 0.1 * span, span], [ym, yM - ym, 0.3 * span, 2 * span],
                 [0, 2 * (yM - ym), span, 0.3 * span]],
                ([-np.inf, -np.inf, 1e-6, 1e-6], [np.inf, np.inf, 1e6, 1e6]), "native",
                "c + a*(1-exp(-x/t1))*exp(-x/t2)")
    if name == "two_exponential":
        return (["c", "a1", "t1", "a2", "t2"],
                lambda p, xx: p[0] + p[1] * np.exp(-np.maximum(xx, 0) / max(p[2], 1e-9))
                + p[3] * np.exp(np.clip(np.maximum(xx, 0) / max(p[4], 1e-9), -50, 50)),
                [[0, y[0], 0.3 * span, 0, span], [ym, y[0] - ym, span, 0, 0.3 * span],
                 [0, 0, span, y[0] - ym, 0.2 * span]],
                ([-np.inf, -np.inf, 1e-6, -np.inf, 1e-6], [np.inf, np.inf, 1e6, np.inf, 1e6]),
                "native", "c + a1*exp(-x/t1) + a2*exp(x/t2)")
    raise KeyError(name)


def fit_curve(x, y, families: Optional[List[str]] = None, log_mode: str = "auto",
              min_points: int = 5) -> "FitSet":
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < min_points:
        raise ValueError("too few finite points")
    order = np.argsort(x)
    x, y = x[order], y[order]
    fams = _families()
    names = families or list(fams)

    positive = bool(np.all(y > 0))
    if log_mode == "auto":
        ratio = (np.max(np.abs(y)) / max(np.min(np.abs(y[np.abs(y) > 0])) if np.any(np.abs(y) > 0) else 1, 1e-30))
        log_mode_eff = "log" if (positive and ratio > 100) else "native"
    else:
        log_mode_eff = log_mode

    fits: List[Fit] = []
    for name in names:
        formula = fams[name][0]
        try:
            if name == "power" and positive and np.all(x > 0):
                lx, ly = np.log(x), np.log(y)
                A = np.vstack([np.ones_like(lx), lx]).T
                p, *_ = np.linalg.lstsq(A, ly, rcond=None)
                res = A @ p - ly
                se = _cov_se(A, res, 2)
                rss = float(np.sum(res ** 2))
                n = len(x)
                f = np.exp(p[0] + p[1] * x * 0 + p[1] * np.log(x))
                fits.append(Fit(name="power", formula="a*x^b", params=np.array([np.exp(p[0]), p[1]]),
                                se=np.array([np.exp(p[0]) * se[0], se[1]]),
                                r2=_r2(y, f), r2_fit=_r2(ly, A @ p),
                                bic=_bic(rss, n, 3), aic=2 * 3 + n * np.log(rss / n), k=3, n=n,
                                domain="loglog (OLS on log y)", pnames=["a", "b"],
                                predict=lambda xx, p=np.array([np.exp(p[0]), p[1]]): p[0] * np.maximum(xx, 1e-30) ** p[1]))
                continue
            if name == "exponential" and positive:
                A = np.vstack([np.ones_like(x), x]).T
                ly = np.log(y)
                p, *_ = np.linalg.lstsq(A, ly, rcond=None)
                res = A @ p - ly
                se = _cov_se(A, res, 2)
                rss = float(np.sum(res ** 2))
                n = len(x)
                f = np.exp(p[0] + p[1] * x)
                fits.append(Fit(name="exponential", formula="a*exp(b*x)",
                                params=np.array([np.exp(p[0]), p[1]]),
                                se=np.array([np.exp(p[0]) * se[0], se[1]]),
                                r2=_r2(y, f), r2_fit=_r2(ly, A @ p),
                                bic=_bic(rss, n, 3), aic=2 * 3 + n * np.log(rss / n), k=3, n=n,
                                domain="semilog (OLS on log y)", pnames=["a", "b"],
                                predict=lambda xx, a=np.exp(p[0]), b=p[1]: a * np.exp(b * np.asarray(xx, float))))
                continue

            pnames, obj, p0s, bounds, dom, fs = _build(name, x, y)
            if obj is None:
                continue
            if p0s is None:
                p0s = [[np.mean(y)] + [0.0] * (len(pnames) - 1)]
            out = _nls(x, y, obj, p0s, bounds, pnames, fs, domain=dom)
            if out is None:
                continue
            p, se, f, res = out
            n = len(x)
            rss = float(np.sum(res ** 2))
            k = len(p) + 1  # + sigma
            # score in the domain the fit minimises
            if log_mode_eff == "log" and positive and np.all(f > 0):
                lr = np.log(f) - np.log(y)
                rss_s, yy = float(np.sum(lr ** 2)), np.log(y)
                r2f = _r2(yy, np.log(f))
            else:
                rss_s, yy, r2f = rss, y, _r2(y, f)
            fits.append(Fit(name=name, formula=fs, params=p, se=se,
                            r2=_r2(y, f), r2_fit=r2f, bic=_bic(rss_s, n, k),
                            aic=2 * k + n * np.log(max(rss_s, 1e-300) / n), k=k, n=n,
                            domain=log_mode_eff if log_mode_eff == "log" else dom,
                            pnames=pnames,
                            predict=lambda xx, obj=obj, p=p: obj(p, np.asarray(xx, float))))
        except Exception as e:  # keep going; a family may be singular for this curve
            fits.append(Fit(name=name, formula=formula, params=np.zeros(0), se=np.zeros(0),
                            r2=np.nan, r2_fit=np.nan, bic=np.inf, aic=np.inf, k=0, n=len(x),
                            domain="", ok=False, msg=str(e)[:120]))

    ok = [f for f in fits if np.isfinite(f.bic)]
    ok.sort(key=lambda f: f.bic)

    # drop sign-degenerate duplicates: keep the better of two fits that predict the same curve
    dedup = []
    for f in ok:
        try:
            yf = f.predict(x)
        except Exception:
            dedup.append(f)
            continue
        dup = False
        for g in dedup:
            try:
                yg = g.predict(x)
            except Exception:
                continue
            if np.allclose(yf, yg, rtol=1e-6, atol=1e-12 * max(1.0, np.nanmax(np.abs(y)))):
                dup = True
                break
        if not dup:
            dedup.append(f)

    return FitSet(x=x, y=y, fits=dedup, log_mode=log_mode_eff)


@dataclass
class FitSet:
    x: np.ndarray
    y: np.ndarray
    fits: List[Fit]
    log_mode: str
    extra: Dict = field(default_factory=dict)

    @property
    def best(self) -> Fit:
        return self.fits[0]

    def delta_bic(self, f: Fit) -> float:
        return float(f.bic - self.fits[0].bic)

    def table(self, top: int = 8) -> str:
        rows = ["| model | formula | params | R2 | R2(fit domain) | dBIC |",
                "|---|---|---|---|---|---|"]
        for f in self.fits[:top]:
            rows.append(f"| {f.name} | `{f.formula}` | {f.line()} | {f.r2:.4f} | "
                        f"{f.r2_fit:.4f} | {self.delta_bic(f):.1f} |")
        return "\n".join(rows)

    def summary(self) -> str:
        b = self.best
        sig = [f for f in self.fits if self.delta_bic(f) < 2]
        alt = ", ".join(f.name for f in sig[1:]) or "none"
        return (f"best={b.name} ({b.formula}); R2={b.r2:.4f}; dBIC(next)<2: {alt}; "
                f"n={b.n}; domain={self.log_mode}")
