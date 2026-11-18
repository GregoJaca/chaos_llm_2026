"""Shared utilities for subagent d2c: D2 / RQA / predictability on 16x1000-token
last-layer trajectories (DeepSeek-R1-Distill-Qwen-1.5B, greedy FP32, window 4096).
CPU-only, numpy/scipy. All distances float64. N=1000 -> full distance matrices OK.
"""
import numpy as np
import json
from pathlib import Path

DATA = Path(r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\long_last")
OUT = Path(r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2c")

PROMPTS = [
    "interstellar_travel", "childhood_experiences", "cybersecurity",
    "healthy_habits", "transformer_architecture", "astronomy_observations",
    "evolution_of_language", "history_of_computing", "history_of_mathematics",
    "renewable_energy_economics", "history_of_music", "ocean_currents",
    "sleep_and_memory", "software_engineering_practices",
    "urban_transportation", "volcanoes_and_plate_tectonics",
]
REP = "renewable_energy_economics"  # flagged as most repetitive


def load(name):
    d = np.load(DATA / (name + ".npz"))
    return d["h_last"].astype(np.float64), d["gen_ids"].astype(np.int64)


def distm(X):
    X = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", X, X)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2)


def pair_values(D, w=0):
    """Upper-triangle distances with time separation > w."""
    N = D.shape[0]
    i, j = np.triu_indices(N, 1)
    m = (j - i) > w
    return D[i[m], j[m]]


def c_eps_curve(D, eps_grid, w=0):
    d = np.sort(pair_values(D, w))
    cnt = np.searchsorted(d, eps_grid, side="right")
    return cnt / d.size


def eps_grid_from_med(med, lo=1e-3, hi=3.0, n=96):
    return np.logspace(np.log10(lo * med), np.log10(hi * med), n)


def log_slope(eps, C):
    le = np.log10(eps)
    lc = np.log10(np.maximum(C, 1e-300))
    s = np.full(le.size, np.nan)
    ok = C > 0
    for k in range(1, le.size - 1):
        if ok[k - 1] and ok[k + 1]:
            s[k] = (lc[k + 1] - lc[k - 1]) / (le[k + 1] - le[k - 1])
    return s


def smooth3(s):
    out = np.full_like(s, np.nan)
    for k in range(len(s)):
        a, b = max(0, k - 1), min(len(s), k + 2)
        seg = s[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[k] = seg.mean()
    return out


def widest_run(eps, C, s, rel_tol=0.15, abs_tol=0.10, min_pts=4):
    """Widest contiguous grid run where |slope - median(run slope)| <=
    max(abs_tol, rel_tol*|median|).  Returns None if no run of >= min_pts points."""
    idx = [k for k in range(len(eps)) if np.isfinite(s[k]) and C[k] > 0]
    best = None
    for a in range(len(idx)):
        run = [idx[a]]
        for b in range(a + 1, len(idx)):
            if idx[b] != run[-1] + 1:
                break
            cand = run + [idx[b]]
            med = float(np.median(s[cand]))
            tol = max(abs_tol, rel_tol * abs(med))
            if not np.all(np.abs(s[cand] - med) <= tol):
                break
            run = cand
        if len(run) >= min_pts:
            w = float(np.log10(eps[run[-1]] / eps[run[0]]))
            if best is None or w > best["width_dec"]:
                best = dict(
                    width_dec=round(w, 4), i0=run[0], i1=run[-1], npts=len(run),
                    slope=round(float(np.median(s[run])), 3),
                    eps_lo=float(eps[run[0]]), eps_hi=float(eps[run[-1]]),
                    C_lo=float(C[run[0]]), C_hi=float(C[run[-1]]),
                )
    if best is not None:
        N = None  # caller fills N for ER-legality
    return best


def run_is_er_legal(run, N, c_lo_mult=10.0, c_hi=0.5):
    if run is None:
        return False
    ceil = 2.0 * np.log10(N)
    return (run["C_lo"] >= c_lo_mult / N) and (run["C_hi"] <= c_hi) and (run["slope"] <= ceil)


def erlegal_width(eps, C, s, N, c_lo_mult=10.0, c_hi=0.5):
    """Total/widest contiguous log-width of grid region with C in [10/N, 0.5]
    and smoothed slope <= 2*log10(N) (Eckmann-Ruelle legal band, d2b convention)."""
    ok = (C >= c_lo_mult / N) & (C <= c_hi) & np.isfinite(s) & (s <= 2 * np.log10(N))
    tot, bestw = 0.0, 0.0
    k = 0
    while k < len(eps):
        if ok[k]:
            k0 = k
            while k < len(eps) and ok[k]:
                k += 1
            w = float(np.log10(eps[k - 1] / eps[k0]))
            tot += w
            bestw = max(bestw, w)
        else:
            k += 1
    return round(tot, 4), round(bestw, 4)


# ---------------- surrogates ----------------

def _common_phases(T, rng):
    F = T // 2 + 1
    ph = rng.uniform(0, 2 * np.pi, size=F)
    ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    return ph


def phase_surrogate(Y, rng):
    """Multivariate common-phase randomization: EXACT cross-periodogram
    preservation (same random phase for every component at each frequency),
    Gaussianized marginals."""
    T = Y.shape[0]
    Yf = np.fft.rfft(Y, axis=0)
    rot = np.exp(1j * _common_phases(T, rng))
    Zf = Yf * rot[:, None]
    if T % 2 == 0:
        Zf[-1, :] = Yf[-1, :] * rng.choice([-1.0, 1.0])
    return np.fft.irfft(Zf, n=T, axis=0)


def _rank_match(Z, Ysorted):
    """Per column: replace Z values by Ysorted values at the same ranks."""
    T, d = Z.shape
    order = np.argsort(Z, axis=0, kind="stable")
    out = np.empty_like(Ysorted)
    cols = np.broadcast_to(np.arange(d)[None, :], (T, d))
    out[order, cols] = Ysorted
    return out


def miiaft(Y, rng, n_iter=40):
    """Multivariate IAAFT: per-column marginal preserved exactly (final rank
    match), per-column spectrum preserved to ~0.5%, cross-spectrum approximately
    (common-phase init; typical lag-1 cross-cov error ~0.1). Transposed layout
    for speed; rank matching via gather, not scatter."""
    YT = np.ascontiguousarray(np.asarray(Y, dtype=np.float64).T)   # (d, T)
    d, T = YT.shape
    Ysorted = np.sort(YT, axis=1)
    Yf = np.fft.rfft(YT, axis=1)
    amp = np.abs(Yf)
    rot = np.exp(1j * _common_phases(T, rng))                      # (F,)
    Zf = Yf * rot[None, :]
    if T % 2 == 0:
        Zf[:, -1] = Yf[:, -1] * rng.choice([-1.0, 1.0])
    ZT = np.fft.irfft(Zf, n=T, axis=1)
    for _ in range(n_iter):
        order = np.argsort(ZT, axis=1, kind="stable")
        # rank match: position m gets the value whose rank equals rank(Z[m]) =
        # position of m in `order`, i.e. gather with argsort(order)
        Z2T = np.take_along_axis(Ysorted, np.argsort(order, axis=1), axis=1)
        Zf2 = np.fft.rfft(Z2T, axis=1)
        Zf2 = amp * np.exp(1j * np.angle(Zf2))
        Zf2[:, 0] = Yf[:, 0]
        if T % 2 == 0:
            Zf2[:, -1] = np.real(Zf2[:, -1])
        ZT = np.fft.irfft(Zf2, n=T, axis=1)
    order = np.argsort(ZT, axis=1, kind="stable")
    outT = np.take_along_axis(Ysorted, np.argsort(order, axis=1), axis=1)
    return np.ascontiguousarray(outT.T)


def cloud_miiaft(Xc, rng, n_top=50, n_iter=25):
    """Statics-preserving cloud surrogate: MIAAFT on the top-n_top PCs (which
    carry the shape of the distance distribution) + multivariate phase
    randomization of the low-variance tail; full-rank reconstruction."""
    U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
    Ytop = U[:, :n_top] * sv[:n_top]
    Ztop = miiaft(Ytop, rng, n_iter=n_iter)
    Ytail = U[:, n_top:] * sv[n_top:]
    Ztail = phase_surrogate(Ytail, rng) if Ytail.shape[1] > 0 else np.zeros((Xc.shape[0], 0))
    return np.hstack([Ztop, Ztail]) @ Vh


def naive_best_fit(eps, C, min_pts=5, r2_min=0.99):
    """Widest window with linear log-log fit r^2 >= r2_min (the 'innocent
    notebook' fit that produced the earlier artifacts). Vectorized prefix-sum
    implementation over all windows."""
    le = np.log10(eps)
    lc = np.log10(np.maximum(C, 1e-300))
    ok = np.where(C > 0)[0]
    if ok.size == 0:
        return None
    k0 = int(ok[0])
    x, y = le[k0:], lc[k0:]
    n = x.size
    if n < min_pts:
        return None
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cy = np.concatenate(([0.0], np.cumsum(y)))
    cxx = np.concatenate(([0.0], np.cumsum(x * x)))
    cyy = np.concatenate(([0.0], np.cumsum(y * y)))
    cxy = np.concatenate(([0.0], np.cumsum(x * y)))
    A = np.arange(n)[:, None]
    B = np.arange(min_pts - 1, n)[None, :]
    aa, bb = np.where(B >= A + min_pts - 1)
    m = bb - aa + 1
    Sx = cx[bb + 1] - cx[aa]
    Sy = cy[bb + 1] - cy[aa]
    Sxx = cxx[bb + 1] - cxx[aa]
    Syy = cyy[bb + 1] - cyy[aa]
    Sxy = cxy[bb + 1] - cxy[aa]
    num = m * Sxy - Sx * Sy
    den1 = m * Sxx - Sx ** 2
    den2 = m * Syy - Sy ** 2
    good = (den1 > 0) & (den2 > 0)
    slope = np.zeros_like(num)
    r2 = np.full_like(num, -1.0)
    slope[good] = num[good] / den1[good]
    r2[good] = (num[good] ** 2) / (den1[good] * den2[good])
    width = x[bb] - x[aa]
    sel = good & (r2 >= r2_min)
    if not sel.any():
        return None
    j = int(np.argmax(np.where(sel, width, -1.0)))
    return dict(
        width_dec=round(float(width[j]), 4), slope=round(float(slope[j]), 3),
        r2=round(float(r2[j]), 5), i0=int(aa[j] + k0), i1=int(bb[j] + k0),
        C_lo=float(C[aa[j] + k0]), C_hi=float(C[bb[j] + k0]),
    )


def iaaft_scalar(y, rng, n_iter=100):
    """Standard univariate IAAFT for a scalar series."""
    y = np.asarray(y, dtype=np.float64)
    ysorted = np.sort(y)
    T = y.size
    yf = np.fft.rfft(y)
    amp = np.abs(yf)
    Z = phase_surrogate(y[:, None], rng)[:, 0]
    for _ in range(n_iter):
        order = np.argsort(Z, kind="stable")
        Z2 = np.empty_like(Z)
        Z2[order] = ysorted
        Zf = np.fft.rfft(Z2)
        Zf = amp * np.exp(1j * np.angle(Zf))
        Zf[0] = yf[0]
        if T % 2 == 0:
            Zf[-1] = np.real(Zf[-1])
        Z = np.fft.irfft(Zf, n=T)
    order = np.argsort(Z, kind="stable")
    out = np.empty_like(Z)
    out[order] = ysorted
    return out


# ---------------- RQA ----------------

def _run_length_hist(R):
    """Maximal diagonal run lengths of a boolean matrix (down-right)."""
    N = R.shape[0]
    L = np.zeros(R.shape, dtype=np.int32)
    L[N - 1] = R[N - 1]
    for a in range(N - 2, -1, -1):
        L[a, :-1] = np.where(R[a, :-1], 1 + L[a + 1, 1:], 0)
        L[a, -1] = R[a, -1]
    starts = R.copy()
    starts[1:, 1:] &= ~R[:-1, :-1]
    lens = L[starts]
    return np.bincount(lens, minlength=N + 1), int(R.sum())


def _run_length_hist_vert(R):
    N = R.shape[0]
    L = np.zeros(R.shape, dtype=np.int32)
    L[N - 1] = R[N - 1]
    for a in range(N - 2, -1, -1):
        L[a] = np.where(R[a], 1 + L[a + 1], 0)
    starts = R.copy()
    starts[1:] &= ~R[:-1]
    lens = L[starts]
    return np.bincount(lens, minlength=N + 1), int(R.sum())


def rqa_from_D(D, rrs=(0.05, 0.10, 0.20), w=10):
    N = D.shape[0]
    dvals = pair_values(D, w)
    band = np.abs(np.subtract.outer(np.arange(N), np.arange(N))) <= w
    out = []
    for rr in rrs:
        thr = float(np.quantile(dvals, rr))  # smallest-rr fraction of pairs are recurrent
        R = (D < thr) & (~band)
        cnt, tot = _run_length_hist(R)
        assert (cnt[1:] * np.arange(1, len(cnt))).sum() == tot, "run decomposition failed"
        det = (tot - cnt[1]) / tot if tot else np.nan
        idx = np.arange(2, len(cnt))
        lmean = float((idx * cnt[2:]).sum() / max(cnt[2:].sum(), 1))
        cntv, _ = _run_length_hist_vert(R)
        lam = (tot - cntv[1]) / tot if tot else np.nan
        tt = float((idx * cntv[2:]).sum() / max(cntv[2:].sum(), 1))
        out.append(dict(rr=rr, thr=thr, det=round(det, 4), lmean=round(lmean, 2),
                        lam=round(lam, 4), tt=round(tt, 2)))
    return out


# ---------------- json ----------------

def save_json(obj, name):
    def enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=1, default=enc)


def rng_for(name, k=0):
    return np.random.default_rng(12345 + 1000 * PROMPTS.index(name) + k)
