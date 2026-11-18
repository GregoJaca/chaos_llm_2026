"""Shared machinery: chunked-pairwise D2, TwoNN, MLE, surrogates, prediction, PE.

All pairwise work is block-wise; no N x N matrix is ever materialized fully.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pt_load import load_pt  # noqa: E402

BASE = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data\pecs'
TRAJ = {
    'cyber': BASE + r'\cyber\last.pt',
    'future': BASE + r'\future\last.pt',
    'great': BASE + r'\great\last.pt',
    'interstellar': BASE + r'\interstellar\last.pt',
    'first': BASE + r'\hidden_states_first.pt',
    'launch': BASE + r'\..\launch\hidden_states_cond_0_layer_-1.pt',
}
RNG = np.random.default_rng(20261118)


def load_traj(name, center=False):
    a = load_pt(TRAJ[name])
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if center:
        a = a - a.mean(0, keepdims=True)
    return a


# ---------------------------------------------------------------- distances

def _block_dist_sq(X, rows, Xref=None):
    """Squared Euclidean distances X[rows] vs (Xref or X). float32, clipped>=0."""
    Xref = X if Xref is None else Xref
    A = X[rows]
    sq_a = (A.astype(np.float64) ** 2).sum(1)[:, None]
    sq_b = (Xref.astype(np.float64) ** 2).sum(1)[None, :]
    G = A @ Xref.T  # BLAS float32
    D = sq_a + sq_b - 2.0 * G.astype(np.float64)
    np.maximum(D, 0.0, out=D)
    return D


def median_pairwise_dist(X, n=1500, rng=None):
    rng = rng or RNG
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    D = _block_dist_sq(X, idx, X[idx])
    iu = np.triu_indices(len(idx), 1)
    return float(np.sqrt(np.median(D[iu])))


def d2_counts(X, eps_grid, theiler=0, block=256, idx=None):
    """counts[i] = #pairs (i<j, j-i>theiler) with dist < eps_grid[i].

    Returns (counts, npairs). Never stores the full matrix.
    """
    X = np.asarray(X, dtype=np.float32)
    N = len(X)
    if idx is not None:
        X = X[idx]
        N = len(X)
    e2 = np.asarray(eps_grid, dtype=np.float64) ** 2
    counts = np.zeros(len(e2), dtype=np.int64)
    npairs = 0
    for a in range(0, N, block):
        rows = np.arange(a, min(a + block, N))
        D = _block_dist_sq(X, rows)
        j = np.arange(N)[None, :]
        mask = (j > rows[:, None]) & (j - rows[:, None] > theiler)
        vals = D[mask]
        npairs += vals.size
        pos = np.searchsorted(e2, vals)
        counts += np.bincount(pos, minlength=len(e2) + 1)[:len(e2)]
    return counts, npairs


def c_curve(X, eps_grid, theiler=0, block=256, idx=None):
    counts, npairs = d2_counts(X, eps_grid, theiler, block, idx)
    if npairs == 0:
        return np.full(len(eps_grid), np.nan), npairs
    return np.cumsum(counts) / npairs, npairs


def local_slopes(eps, C):
    """d log C / d log eps over adjacent grid points."""
    out = np.full(len(eps) - 1, np.nan)
    ok = (C[:-1] > 0) & np.isfinite(C[:-1]) & np.isfinite(C[1:])
    out[ok] = (np.log(C[1:][ok]) - np.log(C[:-1][ok])) / (np.log(eps[1:]) - np.log(eps[:-1]))[ok]
    return out


def slope_at_C(eps, C, targets=(0.05, 0.1, 0.3, 0.5, 0.7, 0.9)):
    """Interpolated local slope where cumulative C crosses each target."""
    res = {}
    lc = local_slopes(eps, C)
    for tg in targets:
        k = np.argmax(C >= tg) if np.any(C >= tg) else -1
        if k <= 0:
            res[tg] = np.nan
        else:
            res[tg] = float(lc[max(k - 1, 0)])
    return res


def d2_fit(eps, C, lo=0.1, hi=0.9):
    """Least-squares D2 over the eps window where C in [lo, hi]."""
    m = (C >= lo) & (C <= hi) & (C > 0)
    if m.sum() < 4:
        return np.nan, np.nan
    x = np.log(eps[m]); y = np.log(C[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = np.sum((y - A @ coef) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(coef[0]), float(1 - ss / sst if sst > 0 else np.nan)


def eps_grid_for(X, n=48, rng=None):
    med = median_pairwise_dist(X, rng=rng)
    return np.geomspace(0.004 * med, 1.6 * med, n)


# ---------------------------------------------------------------- intrinsic dim

def knn_dists(X, k, theiler=0, block=256, idx=None):
    """For each point i (in idx or all): distances to its k nearest neighbours
    with time separation > theiler. Returns [N, k] array (sorted per row)."""
    X = np.asarray(X, dtype=np.float32)
    N0 = len(X)
    if idx is None:
        idx = np.arange(N0)
    N = len(idx)
    out = np.empty((N, k), dtype=np.float64)
    BIG = np.inf
    for a in range(0, N, block):
        rows = idx[a:a + block]
        D = np.sqrt(_block_dist_sq(X, rows))
        pos_in_rows = np.searchsorted(idx, rows)  # rows are members of idx
        j = np.arange(N0)[None, :]
        mask = np.abs(j - rows[:, None]) <= theiler
        mask[np.arange(len(rows)), pos_in_rows] = True
        D[mask] = BIG
        part = np.partition(D, k - 1, axis=1)[:, :k]
        part.sort(axis=1)
        out[a:a + len(rows)] = part
    return out


def twonn(X, theiler=0, idx=None, exclude_zero=True):
    kd = knn_dists(X, 2, theiler=theiler, idx=idx)
    r1, r2 = kd[:, 0], kd[:, 1]
    if exclude_zero:
        good = r1 > 0
    else:
        good = np.isfinite(r1)
    mu = np.log(r2[good] / r1[good])
    mu = mu[np.isfinite(mu)]
    if mu.size == 0:
        return np.nan, 0
    return float(1.0 / mu.mean()), int(mu.size)


def mle_dim(X, ks=(2, 4, 8, 16, 32), theiler=0, idx=None):
    kd = knn_dists(X, max(ks), theiler=theiler, idx=idx)
    out = {}
    for k in ks:
        if k < 2:
            continue
        Tk = kd[:, k - 1]
        good = (Tk > 0) & np.isfinite(Tk)
        s = 0.0
        n = 0
        for j in range(k - 1):
            dij = kd[good, j]
            s += np.sum(np.log(Tk[good] / dij))
            n += dij.size
        out[k] = float(1.0 / (s / n)) if n > 0 else np.nan
    return out


def pca_stats(X, ncomp=100):
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, len(X) - 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.maximum(ev, 0)
    tr = ev.sum()
    fr = ev / tr
    pr = (fr.sum() ** 2) / (fr ** 2).sum()  # participation ratio
    c90 = int(np.searchsorted(np.cumsum(fr), 0.9) + 1)
    return {'eig': ev[:ncomp], 'pr': float(pr), 'eff_rank_shannon': float(np.exp(-(fr * np.log(fr + 1e-12)).sum())),
            'n90': c90, 'top10_var': float(fr[:10].sum()), 'top3_var': float(fr[:3].sum())}


# ---------------------------------------------------------------- surrogates

def time_shuffle(X, rng=None):
    rng = rng or RNG
    return X[rng.permutation(len(X))]


def coord_shuffle(X, rng=None):
    rng = rng or RNG
    out = np.empty_like(X)
    for c in range(X.shape[1]):
        out[:, c] = X[rng.permutation(len(X)), c]
    return out


def phase_rand(X, shared=True, rng=None):
    """Multivariate phase randomization over time (no padding).

    shared=True: same phase per frequency for all coordinates -> preserves the
    full cross-spectrum (all second-order structure), destroys nonlinear flow.
    """
    rng = rng or RNG
    n, d = X.shape
    F = np.fft.rfft(X, axis=0)
    m = F.shape[0]
    if shared:
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=m))
        Fnew = F * ph[:, None]
    else:
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(m, d)))
        Fnew = F * ph
    Fnew[0] = F[0]  # keep mean
    if n % 2 == 0:
        Fnew[-1] = F[-1]  # keep Nyquist real
    y = np.fft.irfft(Fnew, n=n, axis=0)
    for c in range(d):
        y[:, c] *= X[:, c].std() / (y[:, c].std() + 1e-12)
    return y.astype(np.float32)


def miaaft(X, niter=12, rng=None, shared=True):
    """Multivariate IAAFT (Chavez et al. style): iterate phase adjustment to the
    data's cross-spectrum and rank adjustment to the data's marginals.

    Null preserving second-order structure AND marginals, but with no
    deterministic flow. Returns (surrogate, quality) with quality = corr of
    per-coordinate spectra and cross-spectra before/after.
    """
    rng = rng or RNG
    X0 = np.asarray(X, dtype=np.float64)
    n, d = X0.shape
    F = np.fft.rfft(X0, axis=0)
    amp = np.abs(F)
    sorted_cols = np.sort(X0, axis=0)
    y = phase_rand(X, shared=shared, rng=rng).astype(np.float64)
    for _ in range(niter):
        Fy = np.fft.rfft(y, axis=0)
        ph = Fy / (np.abs(Fy) + 1e-12)
        Fnew = amp * ph
        Fnew[0] = F[0]
        if n % 2 == 0:
            Fnew[-1] = F[-1]
        r = np.fft.irfft(Fnew, n=n, axis=0)
        ranks = np.argsort(np.argsort(r, axis=0, kind='stable'), axis=0, kind='stable')
        y = np.take_along_axis(np.broadcast_to(sorted_cols, r.shape), ranks, axis=0)
    # quality: spectrum + marginal + cross-spectrum match
    Fy = np.fft.rfft(y, axis=0)
    spec_corr = float(np.corrcoef(np.abs(F[:, 1:]).ravel(), np.abs(Fy[:, 1:]).ravel())[0, 1])
    # cross-spectrum on a coordinate subsample for cost
    cs = min(d, 40)
    idx = np.linspace(0, d - 1, cs).astype(int)
    Cx = (F[idx] * np.conj(F[idx]).T) if False else None
    A = np.abs(F[:, idx]); B = np.abs(Fy[:, idx])
    Mx = A.T @ A; My = B.T @ B
    cross_corr = float(np.corrcoef(Mx.ravel(), My.ravel())[0, 1])
    return y.astype(np.float32), {'spec_corr': spec_corr, 'cross_corr': cross_corr}


def iaaft(x, niter=50, rng=None):
    """Univariate IAAFT: preserves marginal distribution and power spectrum."""
    rng = rng or RNG
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    xf = np.fft.rfft(x)
    am = np.abs(xf)
    sorted_x = np.sort(x)
    y = x[rng.permutation(n)]
    for _ in range(niter):
        yf = np.fft.rfft(y)
        ph = yf / (np.abs(yf) + 1e-12)
        newf = am * ph
        if n % 2 == 0:
            newf[0] = xf[0]
        else:
            newf[0] = xf[0]
        r = np.fft.irfft(newf, n=n)
        ranks = np.argsort(np.argsort(r, kind='stable'), kind='stable')
        y = sorted_x[ranks]
    return y.astype(np.float32)


def ou_null(X, rng=None, match='lag1'):
    """Per-coordinate AR(1)/OU null with marginal variance matched; lag-1
    autocorrelation matched to data. Purely linear stochastic process."""
    rng = rng or RNG
    Xc = X.astype(np.float64)
    n, d = Xc.shape
    mu = Xc.mean(0)
    X0 = Xc - mu
    sd = X0.std(0) + 1e-12
    Xs = X0 / sd
    if match == 'lag1':
        rho = np.array([np.corrcoef(Xs[:-1, c], Xs[1:, c])[0, 1] for c in range(d)])
        rho = np.clip(rho, -0.99, 0.99)
    else:
        rho = np.full(d, 0.9)
    out = np.empty((n, d))
    out[0] = Xs[0]
    eps = rng.standard_normal((n - 1, d))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + np.sqrt(1 - rho ** 2) * eps[t - 1]
    return (out * sd + mu).astype(np.float32)


def cov_gauss(X, rng=None):
    """Gaussian with exactly the empirical covariance (eigh on the cov matrix)."""
    rng = rng or RNG
    Xc = (X - X.mean(0, keepdims=True)).astype(np.float64)
    C = (Xc.T @ Xc) / max(1, len(X) - 1)
    w, V = np.linalg.eigh(C)  # ascending
    w = np.maximum(w[::-1], 0)
    V = V[:, ::-1]
    scores = rng.standard_normal((len(X), len(w))) * np.sqrt(w)[None, :]
    return (scores @ V.T + X.mean(0, keepdims=True)).astype(np.float32)


def iid_gauss(X, rng=None):
    rng = rng or RNG
    sd = X.std(0)
    return (X.mean(0) + rng.standard_normal(X.shape) * sd).astype(np.float32)


# ---------------------------------------------------------------- dynamics

def knn_neighbors(X, k, W, test_slice, block=256):
    """For each test t: k nearest library neighbours (indices + distances).

    Library = all i with |i - t| > W. Returns (idx [T,k], dist [T,k], tgap [T,k]).
    """
    Xd = np.asarray(X, dtype=np.float32)
    n = len(Xd)
    T = np.arange(*test_slice)
    out_i = np.empty((len(T), k), dtype=np.int64)
    out_d = np.empty((len(T), k), dtype=np.float64)
    B = 512
    for a in range(0, len(T), B):
        tt = T[a:a + B]
        D = _block_dist_sq(Xd, tt, Xd)
        j = np.arange(n)[None, :]
        mask = np.abs(j - tt[:, None]) <= W
        Dm = np.where(mask, np.inf, D)
        sel = np.argpartition(Dm, k - 1, axis=1)[:, :k]
        rows = np.arange(len(tt))[:, None]
        dsq = Dm[rows, sel]
        o = np.argsort(dsq, axis=1)
        out_i[a:a + len(tt)] = np.take_along_axis(sel, o, axis=1)
        out_d[a:a + len(tt)] = np.sqrt(np.maximum(np.take_along_axis(dsq, o, axis=1), 0))
    return out_i, out_d


def predict_from_neighbors(X, nb_idx, nb_d, horizons, k=12, weight=True, test_offset=0):
    """Simplex-style weighted k-NN forecast of x(t+h) from cached neighbours.

    test_offset = time index of test row 0.
    """
    Xd = np.asarray(X, dtype=np.float32)
    n, d = Xd.shape
    out = {}
    for h in horizons:
        kk = min(k, nb_idx.shape[1])
        sel = nb_idx[:, :kk]
        dsel = nb_d[:, :kk]
        if weight:
            w = 1.0 / (dsel + 1e-3)
            w = w / w.sum(1, keepdims=True)
        else:
            w = np.full_like(dsel, 1.0 / kk)
        # guard: target index in range
        valid = sel + h < n
        P = np.einsum('ij,ijd->id', w * valid, Xd[np.clip(sel + h, 0, n - 1)])
        P /= np.maximum((w * valid).sum(1, keepdims=True), 1e-9)
        tt = test_offset + np.arange(len(nb_idx))
        Y = Xd[np.clip(tt + h, 0, n - 1)]
        mlib = Xd[:test_offset].mean(0)
        nse = float(1 - ((P - Y) ** 2).sum() / (((mlib - Y) ** 2).sum() + 1e-12))
        Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
        Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
        pear = float((Pz * Yz).mean())
        pers = Xd[tt]
        nse_p = float(1 - ((pers - Y) ** 2).sum() / (((mlib - Y) ** 2).sum() + 1e-12))
        out[h] = {'nse_knn': nse, 'nse_persist': nse_p, 'pearson': pear}
    return out


def ridge_pcskill(X, horizons, npc=50, lam=1.0, test_frac=0.5):
    """Reduced-rank linear baseline: ridge map from top-npc PCs of x(t) to
    top-npc PCs of x(t+h), evaluated by NSE in PC space (train=first half)."""
    Xd = np.asarray(X, dtype=np.float64)
    n = len(Xd)
    mu = Xd.mean(0); sd = Xd.std(0) + 1e-9
    Z = (Xd - mu) / sd
    C = (Z.T @ Z) / max(1, n - 1)
    w, V = np.linalg.eigh(C)
    P = V[:, ::-1][:, :npc].T  # [npc, d]
    Zp = Z @ P.T  # [n, npc]
    ntr = int(n * test_frac)
    out = {}
    for h in horizons:
        i_tr = np.arange(0, ntr - h)
        i_te = np.arange(ntr, n - h)
        A = Zp[i_tr]
        B = Zp[i_tr + h]
        G = A.T @ A + lam * np.eye(npc)
        W = np.linalg.solve(G, A.T @ B)  # [npc, npc]
        pred = Zp[i_te] @ W
        Y = Zp[i_te + h]
        mu_tr = Zp[:ntr].mean(0)
        nse = float(1 - ((pred - Y) ** 2).sum() / (((mu_tr - Y) ** 2).sum() + 1e-12))
        # persistence in PC space
        pers = Zp[i_te]
        nse_p = float(1 - ((pers - Y) ** 2).sum() / (((mu_tr - Y) ** 2).sum() + 1e-12))
        out[h] = {'nse_ridge': nse, 'nse_persist': nse_p}
    return out


def continuation_curves(X, nb_idx, nb_d, horizons, nbins=12, ref_scale=None, test_offset=0):
    """Kaplan-Glass-style determinism test from cached neighbours.

    For test points t with neighbour i at distance r = ||x_t - x_i||, compare
    r with the continuation distance rc = ||x_{t+h} - x_{i+h}||. Binning r
    against the cloud scale: deterministic dynamics => rc -> 0 as r -> 0;
    stochastic transitions => rc plateaus at a noise floor.
    """
    Xd = np.asarray(X, dtype=np.float32)
    n = len(Xd)
    if ref_scale is None:
        ref_scale = median_pairwise_dist(Xd)
    tt = test_offset + np.arange(len(nb_idx))
    bins = np.geomspace(0.003, 1.0, nbins + 1) * ref_scale
    out = {}
    for h in horizons:
        sel = nb_idx  # all cached neighbours
        r = nb_d.ravel()
        rc = np.linalg.norm(Xd[tt[:, None] + h] - Xd[np.clip(sel + h, 0, n - 1)], axis=2).ravel()
        prof = []
        for b in range(nbins):
            m = (r >= bins[b]) & (r < bins[b + 1])
            if b == nbins - 1:
                m = (r >= bins[b]) & (r <= bins[b + 1])
            prof.append((float(np.median(r[m])) if m.any() else np.nan,
                         float(np.median(rc[m])) if m.any() else np.nan,
                         int(m.sum())))
        out[h] = {'bin_r': [p[0] for p in prof], 'bin_rc': [p[1] for p in prof],
                  'counts': [p[2] for p in prof], 'ref': float(ref_scale)}
    return out


def determinism_score(X, nb_idx, nb_d, h, qlo=0.05, ref_scale=None, test_offset=0):
    """Median rc/r for the closest 5% of neighbour pairs -> sharp map test.

    det = median(rc / r): 0 for deterministic Lipschitz flow at small r;
    ~O(1) or >1 for stochastic transitions.
    """
    Xd = np.asarray(X, dtype=np.float32)
    n = len(Xd)
    tt = test_offset + np.arange(len(nb_idx))
    r = nb_d.ravel()
    thr = np.quantile(r, qlo)
    m = r <= max(thr, 1e-9)
    rc = np.linalg.norm(Xd[tt[:, None] + h] - Xd[np.clip(nb_idx + h, 0, n - 1)], axis=2).ravel()
    rr = r[m]; rcv = rc[m]
    good = rr > 0
    if good.sum() < 50:
        return np.nan, float(thr), int(good.sum())
    return float(np.median(rcv[good] / rr[good])), float(thr), int(good.sum())


def perm_entropy_fast(x, m=5, tau=1):
    """Standardized Bandt-Pompe permutation entropy (1 = fully random order)."""
    import math
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    w = (m - 1) * tau + 1
    if n <= w:
        return np.nan
    Xw = np.lib.stride_tricks.sliding_window_view(x, m * tau)
    if tau > 1:
        Xw = Xw[:, ::tau]
    order = np.argsort(Xw, axis=1, kind='stable')
    codes = np.zeros(len(Xw), dtype=np.int64)
    for j in range(m):
        codes = codes * m + order[:, j]
    _, cnt = np.unique(codes, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


# ---------------------------------------------------------------- misc

def subsample(X, n, rng=None):
    rng = rng or RNG
    return X[np.sort(rng.choice(len(X), size=n, replace=False))]


def jsonable(o):
    import math
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    return o
