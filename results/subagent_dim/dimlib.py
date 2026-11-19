"""dimlib.py -- shared library for the dimensionality analysis of greedy
LLM generation dynamics.

Data conventions (verified in s0_check.py + forensics in the report):
  dim/<tag>/vocab.npz : GE, GW = Gram matrices E^T E, W_U^T W_U (float32),
      sE/sW singular values (desc), normE/normW row norms [V],
      sample_E/sample_W [1024, D] rows = every (V//1024)-th row,
      tied, D, V, L.
  dim/<tag>/acts.npz  : acts [3, L+1, T=400, D] float16 hidden states at the
      GENERATED positions of 3 prompts; index 0 = embedding output (verified:
      norms equal normE[gen_ids] exactly); indices 1..L-1 = outputs of blocks
      0..L-2; index L = POST-FINAL-RMSNORM state (verified: logits == W_U h_L
      with corr 1.0000, ratio 1.0000).  The raw final block is NOT saved.
  dim/<tag>/logits.npz: logits [128, V] at the first 128 generated positions
      of prompt 1; ids == gen_ids[0,:128]; prompt_ids identifies the prompt.
  a1 npz (dynamics)   : 'layers' l <-> acts index l (verified via norm_h0 at
      both ends); a1 does NOT include the post-norm index.

Layer terminology used everywhere below:
  idx  = acts array index 0..L (0 = embedding, 1..L-1 = raw blocks, L = postnorm)
  rel  = idx / L  ("relative depth"; postnorm sits at 1.0)
  a1 layer l (0..L-1) = acts index l.
"""
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- paths
ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results"
DIMROOT = os.path.join(ROOT, "local_data", "dim")
OUT = os.path.join(ROOT, "subagent_dim")
FIGS = os.path.join(OUT, "figs")
CACHE = os.path.join(OUT, "cache")
for _d in (OUT, FIGS, CACHE):
    os.makedirs(_d, exist_ok=True)

DISPLAY = {
    "deepseek": "DeepSeek-R1-Distill-1.5B",
    "qwen3_4b": "Qwen3-4B-Instruct",
    "phi4mini": "Phi-4-mini",
    "qwen3_5": "Qwen3.5-2B",
}
COLOR = {"deepseek": "#4477AA", "qwen3_4b": "#CC3311", "phi4mini": "#228833",
         "qwen3_5": "#EE7733"}
CE, CW = "#66CCEE", "#AA3377"          # embedding / unembedding colors
CMARK = {"deepseek": "o", "qwen3_4b": "s", "phi4mini": "^", "qwen3_5": "D"}

# dynamics a1 npz keyed by dim tag (main sweep; deepseek also has w100)
A1_BY_TAG = {
    "deepseek": os.path.join(ROOT, "local_data", "big", "all_layers", "a1.npz"),
    "deepseek_w100": os.path.join(ROOT, "local_data", "w100_layers", "a1.npz"),
    "qwen3_4b": os.path.join(ROOT, "local_data", "models", "qwen3_4b", "a1.npz"),
    "phi4mini": os.path.join(ROOT, "local_data", "models", "phi", "a1.npz"),
    "qwen3_5": os.path.join(ROOT, "local_data", "models", "qwen35", "a1.npz"),
}

# 4D-mask window sweep (non-DeepSeek models): window4d/<tag>/w*/a1.npz
W4D = os.path.join(ROOT, "local_data", "window4d")

# dynamics numbers supplied by the main project (4 models)
DYN = {
    "late_rate_lambda": {"deepseek": -0.011, "qwen3_4b": 0.0004,
                         "phi4mini": 0.00003, "qwen3_5": 0.0006},
    "kick_gain_s16": {"deepseek": 0.66, "qwen3_4b": 5.93, "phi4mini": 22.99,
                      "qwen3_5": 0.47},
    "dir_spread": {"deepseek": 5.85, "qwen3_4b": 1.15, "phi4mini": 1.23,
                   "qwen3_5": 1.93},
}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 170, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.3,
})


# ---------------------------------------------------------------- io
def discover_tags():
    tags = []
    for t in sorted(os.listdir(DIMROOT)):
        if os.path.isfile(os.path.join(DIMROOT, t, "meta.json")):
            tags.append(t)
    return tags


def discover_windows(tag):
    """{(window:int): path} for window4d/<tag>/w*/a1.npz if present."""
    out = {}
    d = os.path.join(W4D, tag)
    if not os.path.isdir(d):
        return out
    for w in sorted(os.listdir(d)):
        f = os.path.join(d, w, "a1.npz")
        if os.path.isfile(f) and w.startswith("w"):
            out[int(w[1:])] = f
    return out


def load_tag(tag):
    root = os.path.join(DIMROOT, tag)
    meta = json.load(open(os.path.join(root, "meta.json")))
    v = np.load(os.path.join(root, "vocab.npz"))
    a = np.load(os.path.join(root, "acts.npz"))
    lg = np.load(os.path.join(root, "logits.npz"))
    return meta, v, a, lg


def _sanitize(x):
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    if isinstance(x, np.ndarray):
        return _sanitize(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def section_numbers(section, data):
    """Merge one top-level section into numbers.json."""
    path = os.path.join(OUT, "numbers.json")
    js = {}
    if os.path.exists(path):
        with open(path) as f:
            js = json.load(f)
    js[section] = _sanitize(data)
    with open(path, "w") as f:
        json.dump(js, f, indent=1)
    print(f"[numbers.json] wrote section '{section}'")


def savefig(fig, name, close=True):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight", dpi=170)
    if close:
        plt.close(fig)
    print("  fig ->", name)


def band(ax, x, Y, color, label, qs=(0.0, 1.0)):
    """Plot median curve with a percentile band across rows of Y [m, len(x)]."""
    lo, hi = np.quantile(Y, qs[0], axis=0), np.quantile(Y, qs[1], axis=0)
    med = np.median(Y, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, color=color, label=label)


# ------------------------------------------------------- spectral measures
def pr_lambda(lam):
    """Participation ratio (sum l)^2 / sum l^2 ; lam = eigenvalues >= 0."""
    lam = np.asarray(lam, float)
    s1, s2 = lam.sum(), (lam**2).sum()
    return float(s1**2 / s2) if s2 > 0 else float("nan")


def stable_rank(s):
    """Standard stable rank (sum s^2)/(s_1^2)."""
    s = np.asarray(s, float)
    return float((s**2).sum() / s[0] ** 2)


def top_frac(s):
    """Fraction of variance in the top singular direction s1^2/sum s^2."""
    s = np.asarray(s, float)
    return float(s[0] ** 2 / (s**2).sum())


def eff_rank(lam, q=0.9):
    """Smallest k with cumulative lambda share >= q."""
    lam = np.sort(np.asarray(lam, float))[::-1]
    cs = np.cumsum(lam) / lam.sum()
    return int(np.searchsorted(cs, q) + 1)


def powerlaw_slope(vals, i0=10, i1=None):
    """Fit log vals vs log index i (1-based) on [i0, i1]; vals ~ i^-alpha."""
    vals = np.asarray(vals, float)
    n = len(vals)
    i1 = min(i1 or n // 2, n)
    i = np.arange(i0, i1 + 1, dtype=float)
    y = vals[i0 - 1:i1]
    ok = y > 0
    if ok.sum() < 5:
        return float("nan"), float("nan")
    lx, ly = np.log10(i[ok]), np.log10(y[ok])
    A = np.vstack([lx, np.ones_like(lx)]).T
    (beta, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = A @ np.array([beta, c])
    ss = 1 - ((ly - yhat) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(-beta), float(ss)


def eig_desc(M, clip=True):
    lam = np.linalg.eigvalsh(M)[::-1].astype(np.float64)
    if clip:
        lam = np.clip(lam, 0, None)
    return lam


# ------------------------------------------------------- anisotropy / cone
def cos_pairs_from_gram(G, n_pairs=20000, seed=0):
    """Exact pairwise cosines cos(i,j)=G_ij/sqrt(G_ii G_jj) for random i<j."""
    rng = np.random.default_rng(seed)
    V = G.shape[0]
    i = rng.integers(0, V, n_pairs)
    j = rng.integers(0, V, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    c = G[i, j] / np.sqrt(G[i, i] * G[j, j])
    return c.astype(np.float64)


def mean_cos_from_gram(G, n_pairs=20000, seed=0):
    c = cos_pairs_from_gram(G, n_pairs, seed)
    return float(c.mean()), float(c.std() / np.sqrt(len(c)))


def allpairs_cos(X):
    """All pairwise cosines between rows of X (n <= ~2000)."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    return G[iu]


def mean_cos_rows(X):
    c = allpairs_cos(X)
    return float(c.mean()), float(c.std() / np.sqrt(len(c)))


def cone_ratio_from_gram(G, rownorms):
    """||mean row|| / mean ||row||, exact from the Gram:
    ||(1/V) sum_i e_i||^2 = 1^T G 1 / V^2."""
    V = G.shape[0]
    m2 = float(np.einsum("i,ij,j->", np.ones(V, G.dtype), G,
                          np.ones(V, G.dtype))) / V**2
    return float(np.sqrt(m2) / np.mean(rownorms))


def cone_ratio_rows(X):
    mu = X.mean(axis=0)
    return float(np.linalg.norm(mu) / np.linalg.norm(X, axis=1).mean())


# ------------------------------------------------------- distances & kNN
def pairwise_dists(X, chunk=256):
    """Full euclidean distance matrix, float32, chunked."""
    n = len(X)
    sq = np.einsum("ij,ij->i", X, X)
    D = np.empty((n, n), np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        g = X[a:b] @ X.T
        g *= -2.0
        g += sq[a:b, None]
        g += sq[None, :]
        np.maximum(g, 0, out=g)
        np.sqrt(g, out=g)
        D[a:b] = g
    return D


def knn_dists(D, kmax):
    """Sorted kNN distances (excluding self) from a distance matrix."""
    part = np.partition(D, np.arange(1, kmax + 1), axis=1)[:, 1:kmax + 1]
    return np.sort(part, axis=1)


# ------------------------------------------------------- intrinsic dim
def lb_mle_orig(nn, k):
    """Levina-Bickel MLE, original (harmonic) form:
    d = [ mean_i (1/(k-1)) sum_{j<k} log(T_k / T_j) ]^{-1}.
    nn[i, j] = distance to (j+1)-th nearest neighbor."""
    T = np.maximum(nn[:, :k].astype(np.float64), 1e-12)
    with np.errstate(divide="ignore"):
        logs = np.log(T[:, k - 1][:, None] / T[:, : k - 1])
    return float(1.0 / logs.mean())


def lb_mle_perpoint(nn, k):
    """Per-point LB dimension d_i (same estimator, before averaging)."""
    T = np.maximum(nn[:, :k].astype(np.float64), 1e-12)
    with np.errstate(divide="ignore"):
        logs = np.log(T[:, k - 1][:, None] / T[:, : k - 1])
    d_i = 1.0 / logs.mean(axis=1)
    return d_i


def two_nn(nn):
    """TWO-NN (Facco et al. 2017): d = -slope of log(1-F) vs log(mu),
    mu = r2/r1. Returns (d, R^2, n_used)."""
    mu = nn[:, 1].astype(np.float64) / np.maximum(nn[:, 0], 1e-12)
    mu = mu[mu > 1.0 + 1e-12]
    mu.sort()
    n = len(mu)
    if n < 10:
        return float("nan"), float("nan"), n
    F = np.arange(1, n + 1) / n
    lx, ly = np.log(mu), np.log(1.0 - F)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (beta, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = A @ np.array([beta, c])
    ss = 1 - ((ly - yhat) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(-beta), float(ss), n


def corr_dim_from_dists(dpairs, n_min_pairs=80):
    """Correlation dimension: slope of log C(r) vs log r over the scaling
    region 2e-3 <= C(r) <= 0.2 (and >= n_min_pairs pairs below r)."""
    dp = np.sort(np.asarray(dpairs, np.float64))
    npair = len(dp)
    rlo, rhi = dp[0], dp[npair // 2]
    if not (rhi > rlo > 0):
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    rs = np.logspace(np.log10(rlo * (1 + 1e-6)), np.log10(rhi), 48)
    C = np.searchsorted(dp, rs, side="right") / npair
    ok = (C >= 2e-3) & (C <= 0.2)
    ok &= np.searchsorted(dp, rs, side="right") >= n_min_pairs
    if ok.sum() < 6:
        return float("nan"), float("nan"), (float(rlo), float(rhi))
    lx, ly = np.log10(rs[ok]), np.log10(C[ok])
    A = np.vstack([lx, np.ones_like(lx)]).T
    (beta, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = A @ np.array([beta, c])
    ss = 1 - ((ly - yhat) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(beta), float(ss), (float(rs[ok][0]), float(rs[ok][-1]))


# ------------------------------------------------------- misc
def subspace_overlap(U, V):
    """Mean squared principal cosines between column-orthonormal U, V."""
    M = U.T @ V
    s = np.linalg.svd(M, compute_uv=False)
    return float((s**2).mean())


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def note(text):
    print("  *", text)
