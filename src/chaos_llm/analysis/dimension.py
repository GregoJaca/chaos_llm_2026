"""Chunked correlation dimension and RQA. No N x N matrices are stored."""
import numpy as np
import torch


def _pair_counts(X, eps, theiler=0, chunk=512):
    N = len(X)
    counts = np.zeros(len(eps), dtype=np.int64)
    valid_total = 0
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        D = torch.cdist(Xt[i0:i1], Xt)  # [b,N]
        ii = torch.arange(i0, i1)[:, None]
        jj = torch.arange(N)[None, :]
        valid = (torch.abs(ii - jj) > theiler)
        valid_total += int(valid.sum())
        D = D.masked_fill(~valid, float("inf"))
        for k, e in enumerate(eps):
            counts[k] += int((D <= e).sum())
    return counts, valid_total


def correlation_curve(X, eps, theiler=0, chunk=512):
    counts, total = _pair_counts(X, eps, theiler=theiler, chunk=chunk)
    C = counts.astype(np.float64) / max(total, 1)
    return C


def fit_plateau(eps, C, min_points=6):
    ok = C > 0
    eps = np.asarray(eps)[ok]; C = np.asarray(C)[ok]
    if len(C) < min_points:
        return np.nan, np.nan, None
    x = np.log(eps); y = np.log(C)
    best = (-1, np.nan, np.nan, None)
    for i in range(len(x) - min_points + 1):
        for j in range(i + min_points, len(x) + 1):
            xx = x[i:j]; yy = y[i:j]
            if np.ptp(xx) <= 0: continue
            slope, intercept = np.polyfit(xx, yy, 1)
            pred = slope * xx + intercept
            ss_res = np.sum((yy - pred) ** 2); ss_tot = np.sum((yy - yy.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best[0]:
                best = (r2, slope, intercept, (float(eps[i]), float(eps[j-1])))
    return best[1], best[0], best[3]


def eps_grid(X, n=25, lo=2, hi=60):
    # percentiles of off-diagonal distances, sampled
    N = len(X); rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, 1000), replace=False)
    D = torch.cdist(torch.as_tensor(X[idx], dtype=torch.float32),
                    torch.as_tensor(X[idx], dtype=torch.float32)).numpy()
    vals = D[~np.eye(len(idx), dtype=bool)]
    return np.percentile(vals, np.linspace(lo, hi, n))


def d2_curve(X, quintiles=(2, 60), n_eps=25, theiler=0, chunk=512):
    eps = eps_grid(X, n=n_eps, lo=quintiles[0], hi=quintiles[1])
    C = correlation_curve(X, eps, theiler=theiler, chunk=chunk)
    slope, r2, rng = fit_plateau(eps, C)
    return dict(eps=eps.tolist(), C=C.tolist(), d2=float(slope), r2=float(r2), fit_range=rng)


def rqa(X, target_rr=0.02, theiler=0, lmin=2, max_n=2500):
    if len(X) > max_n:
        X = X[:max_n]
    N = len(X)
    eps = eps_grid(X, n=200, lo=1, hi=50)
    C = correlation_curve(X, eps, theiler=0, chunk=512)
    k = int(np.argmin(np.abs(C - target_rr)))
    e = eps[k]
    D = torch.cdist(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(X, dtype=torch.float32)).numpy()
    R = D <= e
    mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > theiler
    R = R & mask
    RR = R.mean()
    # diagonal lines
    dlines = []
    for off in range(theiler + 1, N):
        d = np.diag(R, k=off)
        run = 0
        for v in d:
            if v: run += 1
            elif run: dlines.append(run); run = 0
        if run: dlines.append(run)
    vlines = []
    for off in range(theiler + 1, N):
        d = np.diag(R, k=-off)
        run = 0
        for v in d:
            if v: run += 1
            elif run: vlines.append(run); run = 0
        if run: vlines.append(run)
    def ratio(lines):
        arr = np.asarray(lines)
        if len(arr) == 0: return 0.0
        return float(arr[arr >= lmin].sum() / arr.sum())
    return dict(eps=float(e), RR=float(RR), DET=ratio(dlines), LAM=ratio(vlines),
                Lmax=int(max(dlines) if dlines else 0), n_diag=int(len(dlines)), n_vert=int(len(vlines)))
