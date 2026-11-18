"""Shared loaders + analysis helpers for the stable-chaos study.

Data layout (verified by inspection):
  fr_*.npz (free runs):  obs[nr,20]  first flip step (-1 = no flip within 300)
                         pred[nr,20] teacher-forced threshold-model prediction (== a1 pred_T at matching radii)
                         radii[nr], ref_ids[nb,300], gen_ids[nb] object; batch = [ref row, then 6 (or 2) dir rows]
                         runs ordered RADIUS-major; first-mismatch of run vs ref == obs exactly.
  a1.npz (teacher-forced): margins[300] (top1-top2 logit gap along reference),
                         dm[i,d,t] margin change for radius radii[i] in dir d,
                         r_c[i,d,t] = margins[t]*radii[i]/|dm[i,d,t]|  (critical radius; verified exact),
                         pred_T[i,d] = first t with r_c[i,d,t] < radii[i] (verified exact),
                         subspace[20] direction-token ids (identical across windows/prompts files).
"""
import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos"


def load_free(name):
    d = np.load(BASE + "\\" + name, allow_pickle=True)
    ref = np.asarray(d["ref_ids"])[0].astype(np.int64)
    runs = []
    for el in d["gen_ids"]:
        el = np.asarray(el)
        for row in el[1:]:
            runs.append(np.asarray(row).astype(np.int64))
    runs = np.array(runs)
    nr = d["obs"].shape[0]
    assert runs.shape[0] == nr * 20, (name, runs.shape)
    return dict(obs=np.asarray(d["obs"]), pred=np.asarray(d["pred"]),
                radii=np.asarray(d["radii"], float), ref=ref, runs=runs,
                mismatch=(runs != ref[None, :]))


def load_a1(rel):
    d = np.load(BASE + "\\" + rel, allow_pickle=True)
    return dict(margins=np.asarray(d["margins"], float),
                r_c=np.asarray(d["r_c"], float), dm=np.asarray(d["dm"], float),
                pred_T=np.asarray(d["pred_T"]), radii=np.asarray(d["radii"], float),
                gen=np.asarray(d["gen_ids"]), top1=np.asarray(d["top1"]),
                top2=np.asarray(d["top2"]), subspace=np.asarray(d["subspace"]))


# ---------------- survival analysis ----------------

def km_curve(T, tmax=300):
    """T: int array of flip times; -1 = censored (no flip by 300).
    Returns t_grid (0..tmax), S(t) = P(no flip yet at step t) = P(T > t-1)... we define
    S(t) = fraction of runs whose flip time (if any) is >= t, i.e. survived steps < t.
    S(0)=1; S(t) drops at each observed flip. Censored runs counted as surviving up to tmax.
    Vectorized: S(t) = mean( (T == -1) | (T >= t) )."""
    t_grid = np.arange(0, tmax + 1)
    Tc = np.where(T == -1, tmax + 1, T)  # censored -> survives beyond tmax
    S = (Tc[None, :] >= t_grid[:, None]).mean(axis=1)
    return t_grid, S


def mean_restricted(T, tmax=300):
    """Mean flip time censored-aware: E[min(T, tmax)] = sum_{t=0}^{tmax-1} S(t)."""
    _, S = km_curve(T, tmax)
    return float(np.sum(S[:-1]))  # area under survival up to tmax


def bootstrap_mean_T(T, tmax=300, nboot=2000, seed=0):
    rng = np.random.default_rng(seed)
    Tc = np.where(T == -1, tmax + 1, T)
    vals = np.empty(nboot)
    n = len(T)
    for b in range(nboot):
        samp = Tc[rng.integers(0, n, n)]
        vals[b] = mean_restricted(np.where(samp == tmax + 1, -1, samp), tmax)
    return vals


def bootstrap_S(T, tmax=300, nboot=2000, seed=0, thin=5):
    """Pointwise 95% CI band for the survival curve."""
    rng = np.random.default_rng(seed)
    t_grid = np.arange(0, tmax + 1, thin)
    Tc = np.where(T == -1, tmax + 1, T)
    n = len(T)
    curves = np.empty((nboot, len(t_grid)))
    for b in range(nboot):
        samp = Tc[rng.integers(0, n, n)]
        curves[b] = (samp[None, :] >= t_grid[:, None]).mean(axis=1)
    lo, hi = np.quantile(curves, [0.025, 0.975], axis=0)
    return t_grid, lo, hi


def threshold_survival(r_c_dir_t, r, tmax=300):
    """Threshold-field prediction. r_c_dir_t: [ndir, nsteps] critical-radius field.
    S_r(t) = fraction of dirs with min_{t'<=t} r_c(t') > r  (no flip through step t).
    Flip at step t iff r_c(t) < r (pred_T convention)."""
    below = r_c_dir_t < r                    # [ndir, nsteps]
    Tpred = np.argmax(below, axis=1)
    none = ~below.any(axis=1)
    Tpred[none] = -1
    return km_curve(Tpred, tmax), Tpred


# ---------------- survival fits ----------------

def fit_survival_forms(T, tmax=300):
    """Fit S(t) with exponential, stretched exponential, power law (censored LS on log S).
    Uses only t where S>0; censored observations are handled by S itself."""
    t, S = km_curve(T, tmax)
    m = S > 0
    tt, ss = t[m].astype(float), S[m]
    y = np.log(ss)
    res = {}

    # exponential: S = exp(-lam t)
    A = np.vstack([np.ones_like(tt), tt]).T
    c, lam = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = c + lam * tt
    res["exp"] = dict(lam=float(-lam), R2=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)))

    # stretched exponential: S = exp(-(t/tau)^beta) ; linearize via log(-log S) = beta log t - beta log tau
    ok = ss < 1.0
    if ok.sum() > 5:
        lx = np.log(tt[ok][tt[ok] > 0] if (tt[ok] > 0).any() else tt[ok])
        ly = np.log(-np.log(ss[ok]))
        A = np.vstack([np.ones_like(lx), lx]).T
        cc, beta = np.linalg.lstsq(A, ly, rcond=None)[0]
        tau = float(np.exp(-cc / beta)) if beta != 0 else np.nan
        yy = np.log(-np.log(np.clip(ss, 1e-12, 1)))
        pr = cc + beta * np.log(np.clip(tt, 1e-12, None))
        okk = np.isfinite(yy) & np.isfinite(pr)
        R2 = float(1 - np.sum((yy[okk] - pr[okk]) ** 2) / np.sum((yy[okk] - yy[okk].mean()) ** 2)) if okk.sum() > 2 else np.nan
        res["stretched"] = dict(beta=float(beta), tau=tau, R2=R2)
    else:
        res["stretched"] = dict(beta=np.nan, tau=np.nan, R2=np.nan)

    # power law: S = (1 + t/t0)^(-g)
    ok2 = ss < 1.0
    if ok2.sum() > 5:
        # y = log S ; grid search t0 then linear in log(1+t/t0)
        best = None
        for t0 in np.geomspace(1e-2, 1e3, 200):
            x = np.log1p(tt / t0)
            A = np.vstack([-x, np.ones_like(x)]).T
            coef, g = np.linalg.lstsq(A, y, rcond=None)[0]
            pr = coef - g * x
            ssr = np.sum((y - pr) ** 2)
            if best is None or ssr < best[0]:
                best = (ssr, float(g), float(t0))
        yy = y
        R2 = float(1 - best[0] / np.sum((yy - yy.mean()) ** 2))
        res["power"] = dict(gamma=best[1], t0=best[2], R2=R2)
    else:
        res["power"] = dict(gamma=np.nan, t0=np.nan, R2=np.nan)
    return res


def r2_of_prediction(S_obs, S_pred):
    """R^2 of a predicted survival curve against observed (both on same t grid)."""
    y = S_obs
    if np.sum((y - y.mean()) ** 2) == 0:
        return np.nan
    return float(1 - np.sum((y - S_pred) ** 2) / np.sum((y - y.mean()) ** 2))


def sup_dist(S_obs, S_pred):
    return float(np.max(np.abs(np.asarray(S_obs) - np.asarray(S_pred))))


def mad(S_obs, S_pred):
    """Mean absolute deviation between survival curves (robust metric for step functions)."""
    return float(np.mean(np.abs(np.asarray(S_obs) - np.asarray(S_pred))))


# ---------------- T(r) scaling fits ----------------

def fit_T_vs_r(r, Tmean):
    """Fit mean flip time vs radius: log-law T = a - b ln r (ordinary chaos, b=1/lambda);
    power law log T = log c - alpha log r (fit in log-log).
    Returns dict with params and R^2 for both."""
    r = np.asarray(r, float)
    y = np.asarray(Tmean, float)
    out = {}
    # log law: T = a - b ln r
    A = np.vstack([np.ones_like(r), np.log(r)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ coef
    sst = np.sum((y - y.mean()) ** 2)
    out["log"] = dict(a=float(coef[0]), b=float(-coef[1]),
                      R2=float(1 - np.sum((y - pred) ** 2) / sst))
    # power law: log T = log c - alpha log r
    ly = np.log(y)
    A = np.vstack([np.ones_like(r), np.log(r)]).T
    coef = np.linalg.lstsq(A, ly, rcond=None)[0]
    predly = A @ coef
    sstl = np.sum((ly - ly.mean()) ** 2)
    out["pow"] = dict(c=float(np.exp(coef[0])), alpha=float(-coef[1]),
                      R2=float(1 - np.sum((ly - predly) ** 2) / sstl))
    return out
