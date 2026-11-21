"""Part 5: trajectories, D2, recurrence — Mamba-130m control vs DeepSeek-1.5B.

numpy-only mirror of src/chaos_llm/analysis/dimension.py (same eps grid,
same plateau fit), with the reference systems built exactly as in the
project's validated D2 work (subagent_d2b): Henon direct 2-D states,
Lorenz x[::4] delay-embedded m=8, tau=4, both at N=3031, wide eps grid.

Outputs: notebook/figures_controls/c5_*.png + results/ctrl_local/d2_summary.json
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAM_TRAJ = os.path.join(ROOT, "results", "ctrl_local", "mamba130", "traj")
DS_TRAJ = os.path.join(ROOT, "results", "local_data_ds", "long_last")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)

C_MAM = "#0072B2"
C_DS = "#D55E00"
NPTS = 1000
NREF = 3031


# ---------------- estimator (numpy mirror of analysis/dimension.py) ----------------
def _pair_counts(X, eps, theiler=0, chunk=512):
    N = len(X)
    counts = np.zeros(len(eps), dtype=np.int64)
    valid_total = 0
    sq = (X * X).sum(1)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        D2 = sq[i0:i1, None] + sq[None, :] - 2.0 * X[i0:i1] @ X.T
        np.maximum(D2, 0.0, out=D2)
        valid = np.abs(np.arange(i0, i1)[:, None] - np.arange(N)[None, :]) > theiler
        valid_total += int(valid.sum())
        D2[~valid] = np.inf
        D = np.sqrt(D2)
        for k, e in enumerate(eps):
            counts[k] += int((D <= e).sum())
    return counts, valid_total


def correlation_curve(X, eps, theiler=0, chunk=512):
    counts, total = _pair_counts(X, eps, theiler=theiler, chunk=chunk)
    return counts.astype(np.float64) / max(total, 1)


def fit_plateau(eps, C, min_points=6):
    ok = C > 0
    eps = np.asarray(eps)[ok]
    C = np.asarray(C)[ok]
    if len(C) < min_points:
        return np.nan, np.nan, None
    x = np.log(eps)
    y = np.log(C)
    best = (-1, np.nan, np.nan, None)
    for i in range(len(x) - min_points + 1):
        for j in range(i + min_points, len(x) + 1):
            xx = x[i:j]
            yy = y[i:j]
            if np.ptp(xx) <= 0:
                continue
            slope, intercept = np.polyfit(xx, yy, 1)
            pred = slope * xx + intercept
            ss_res = np.sum((yy - pred) ** 2)
            ss_tot = np.sum((yy - yy.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if r2 > best[0]:
                best = (r2, slope, intercept, (float(eps[i]), float(eps[j - 1])))
    return best[1], best[0], best[3]


def eps_grid(X, n=25, lo=2, hi=60):
    N = len(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, 1000), replace=False)
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[idx][:, None] + sq[idx][None, :] - 2.0 * X[idx] @ X[idx].T, 0))
    vals = D[~np.eye(len(idx), dtype=bool)]
    return np.percentile(vals, np.linspace(lo, hi, n))


def eps_grid_wide(X, n=48):
    """Log-spaced eps over the 0.5..95th percentile of distances (references)."""
    N = len(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, 1000), replace=False)
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[idx][:, None] + sq[idx][None, :] - 2.0 * X[idx] @ X[idx].T, 0))
    vals = D[~np.eye(len(idx), dtype=bool)]
    lo, hi = np.percentile(vals, [0.5, 95])
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def d2_curve(X, n_eps=25, wide=False, theiler=0):
    eps = eps_grid_wide(X, n=n_eps) if wide else eps_grid(X, n=n_eps)
    C = correlation_curve(X, eps, theiler=theiler)
    slope, r2, rng = fit_plateau(eps, C)
    width = float(np.log10(rng[1] / rng[0])) if rng else 0.0
    return dict(eps=np.asarray(eps), C=C, d2=float(slope), r2=float(r2),
                fit_range=rng, width_dec=width)


def prep(X, n=None):
    if n is not None and len(X) > n:
        X = X[:n]
    return X - X.mean(axis=0, keepdims=True)


# ---------------- plateau scan (port of the project's cshape.py detector) ------
def plateau_scan(eps_mid, slope, C_mid, N, tol_rel=0.15, floor_tol=0.05,
                 c_lo=None, c_hi=0.5, er_cap=True):
    """Longest contiguous run where the local slope stays within tol_rel of its
    value at run start, restricted to unsaturated/non-depleted C and (ER cap)
    slope <= 2 log10 N. Returns runs sorted by width in decades."""
    er = 2.0 * np.log10(max(N, 10))
    if c_lo is None:
        c_lo = 10.0 / N
    good = np.isfinite(slope) & (C_mid >= c_lo) & (C_mid <= c_hi)
    if er_cap:
        good = good & (slope <= er)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = prev = idx[0]
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
        out.append(dict(slope=float(np.median(slope[a:b + 1])), width_dec=float(width),
                        eps0=float(eps_mid[a]), eps1=float(eps_mid[b])))
    out.sort(key=lambda r: -r["width_dec"])
    return out


def plateau_stats(X, n_eps=48):
    """Best plateau (flat local-slope run) of the correlation curve."""
    eps = eps_grid_wide(X, n=n_eps)
    C = correlation_curve(X, eps, theiler=0)
    lm = np.log(eps)
    lC = np.log(np.maximum(C, 1e-300))
    slope = np.diff(lC) / np.diff(lm)
    mid = np.sqrt(eps[:-1] * eps[1:])
    C_mid = np.sqrt(np.maximum(C[:-1], 1e-300) * np.maximum(C[1:], 1e-300))
    runs = plateau_scan(mid, slope, C_mid, N=len(X))
    return (runs[0] if runs else None), eps, C


# ---------------- references (as in the project's validated D2 work) ----------
def henon(n=NREF, a=1.4, b=0.3, burn=2000):
    rng = np.random.default_rng(7)
    x, y = 0.1 * rng.standard_normal(), 0.1 * rng.standard_normal()
    out = np.empty((n, 2))
    for i in range(-burn, n):
        x, y = 1.0 - a * x * x + b * y, x
        if i >= 0:
            out[i] = (x, y)
    return out


def lorenz(n_steps=60000, dt=0.01, s=10.0, r=28.0, bb=8.0 / 3.0):
    out = np.empty((n_steps, 3))
    v = np.asarray([1.0, 1.0, 20.0])
    for i in range(n_steps):
        dv = np.array([s * (v[1] - v[0]), v[0] * (r - v[2]) - v[1], v[0] * v[1] - bb * v[2]])
        v2 = v + 0.5 * dt * dv
        dv2 = np.array([s * (v2[1] - v2[0]), v2[0] * (r - v2[2]) - v2[1], v2[0] * v2[1] - bb * v2[2]])
        v = v + dt * dv2
        out[i] = v
    return out


def delay_embed(x, m, tau):
    T = len(x) - (m - 1) * tau
    return np.stack([x[i * tau: i * tau + T] for i in range(m)], axis=1)


# ---------------- loop screening ----------------
def loop_screen(x):
    T = len(x)
    x = x - x.mean(0, keepdims=True)
    scale = float(np.median(np.linalg.norm(x, axis=1)))
    n = 1 << (2 * T - 1).bit_length()
    F = np.fft.rfft(x, n=n, axis=0)
    ac = np.fft.irfft((F * np.conj(F)).sum(1), n=n)[:T]
    sq = (x * x).sum(1)
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    best = (9e9, -1)
    for P in range(5, min(260, T - 10)):
        s = cs[T - P] + (cs[T] - cs[P]) - 2 * ac[P]
        d2 = np.sqrt(max(s / (T - P), 0.0)) / scale
        if d2 < best[0]:
            best = (d2, P)
    return best[1], best[0]


# ---------------- run ----------------
print("== reference validation (same construction as the project D2 work) ==")
refs = {}
H = prep(henon())
c = d2_curve(H, wide=True)
refs["henon"] = dict(d2=c["d2"], r2=c["r2"], width=c["width_dec"])
print(f"henon N={len(H)}: D2={c['d2']:.3f} (project value 1.157, true ~1.26) "
      f"plateau {c['width_dec']:.2f} dec r2={c['r2']:.4f}")
L = lorenz()
Lx = delay_embed(L[:, 0][::4], m=8, tau=4)[:NREF]
Lx = prep(Lx)
c = d2_curve(Lx, wide=True)
refs["lorenz"] = dict(d2=c["d2"], r2=c["r2"], width=c["width_dec"])
print(f"lorenz x[::4] m=8 tau=4 N={len(Lx)}: D2={c['d2']:.3f} (project value 1.984) "
      f"plateau {c['width_dec']:.2f} dec r2={c['r2']:.4f}")

print("\n== loop screening ==")
rows = []
for tag, root in [("mamba", MAM_TRAJ), ("deepseek", DS_TRAJ)]:
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
        X = np.load(f)["h_last"].astype(np.float64)
        P, v = loop_screen(X)
        verdict = "LOOP" if v < 0.10 else ("weak" if v < 0.25 else "ok")
        rows.append(dict(model=tag, name=os.path.basename(f)[:-4], period=P, rel=float(v), verdict=verdict))
        print(f"{tag:9s} {os.path.basename(f)[:-4]:30s} P={P:4d} rel={v:.3f} {verdict}")

fig, ax = plt.subplots(figsize=(9.5, 4.8))
for tag, c in [("mamba", C_MAM), ("deepseek", C_DS)]:
    rs = sorted([r for r in rows if r["model"] == tag], key=lambda r: r["rel"])
    ax.barh([f"{tag[0]}: {r['name'][:20]}" for r in rs], [r["rel"] for r in rs], color=c, alpha=0.75)
ax.axvline(0.10, color="k", ls="--", lw=1)
ax.set_xlabel(r"best repetition relative distance $\min_P \langle\|x_t-x_{t+P}\|\rangle / \mathrm{med}\|x\|$")
ax.set_title("Loop screening at the project's 0.10 cutoff (dashed):\nthe control's greedy trajectories are non-repetitive, like the LLM's")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5_loops.png"), dpi=190); plt.close(fig)

print("\n== D2 curves (last-layer states, N=%d, same detector as the LLM work) ==")
results = {"refs": refs, "rows": rows, "traj": {}}
d2s = {"mamba": [], "deepseek": []}
widths = {"mamba": [], "deepseek": []}
examples = {}
for tag, root in [("mamba", MAM_TRAJ), ("deepseek", DS_TRAJ)]:
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
        name = os.path.basename(f)[:-4]
        X = prep(np.load(f)["h_last"].astype(np.float64), n=NPTS)
        c = d2_curve(X)
        r = next(r for r in rows if r["model"] == tag and r["name"] == name)
        if r["verdict"] == "LOOP":
            continue
        d2s[tag].append(c["d2"])
        widths[tag].append(c["width_dec"])
        results["traj"][f"{tag}/{name}"] = dict(d2=c["d2"], r2=c["r2"],
                                                fit_width_dec=c["width_dec"],
                                                fit_range=c["fit_range"])
        examples.setdefault(tag, []).append((name, c))
        print(f"{tag:9s} {name:30s} D2={c['d2']:.2f} r2={c['r2']:.3f} fit width={c['width_dec']:.2f} dec")

for tag in ["mamba", "deepseek"]:
    v = np.array(d2s[tag])
    w = np.array(widths[tag])
    results[f"{tag}_d2"] = v.tolist()
    results[f"{tag}_width"] = w.tolist()
    print(f"{tag}: D2 median {np.median(v):.2f} [{v.min():.2f},{v.max():.2f}]; "
          f"fit width median {np.median(w):.2f} dec (max {w.max():.2f})")

# surrogate test: does the control's D2 survive destroying temporal order?
print("\n== surrogates (2 trajectories per model) ==")
def phase_surrogate(X, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(X, axis=0)
    ph = np.exp(2j * np.pi * rng.random(F.shape))
    return np.fft.irfft(F * ph, n=X.shape[0], axis=0)

surr = {}
for tag, root in [("mamba", MAM_TRAJ), ("deepseek", DS_TRAJ)]:
    for f in sorted(glob.glob(os.path.join(root, "*.npz")))[:2]:
        name = os.path.basename(f)[:-4]
        X = prep(np.load(f)["h_last"].astype(np.float64), n=NPTS)
        out = {}
        for kind, Z in [("raw", X),
                        ("shuffle", X[np.random.default_rng(1).permutation(len(X))]),
                        ("phase", phase_surrogate(X, 2))]:
            c = d2_curve(Z)
            out[kind] = dict(d2=c["d2"], r2=c["r2"], width=c["width_dec"])
        surr[f"{tag}/{name}"] = out
        print(f"{tag:9s} {name:30s} raw D2={out['raw']['d2']:.2f}  "
              f"shuffled D2={out['shuffle']['d2']:.2f}  phase D2={out['phase']['d2']:.2f}")
results["surrogates"] = surr

# plateau scan (the project's flatness criterion) on refs + trajectories
print("\n== plateau scan (flat local-slope run, the project's criterion) ==")
plw = {"mamba": [], "deepseek": []}
pls = {"mamba": [], "deepseek": []}
ref_pl = {}
for name, X in [("henon", H), ("lorenz", Lx)]:
    run, eps, C = plateau_stats(X)
    ref_pl[name] = dict(slope=run["slope"], width_dec=run["width_dec"]) if run else None
    print(f"{name:9s} plateau slope={run['slope']:.3f} width={run['width_dec']:.2f} dec")
for tag, root in [("mamba", MAM_TRAJ), ("deepseek", DS_TRAJ)]:
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
        nm = os.path.basename(f)[:-4]
        r = next(r for r in rows if r["model"] == tag and r["name"] == nm)
        if r["verdict"] == "LOOP":
            continue
        X = prep(np.load(f)["h_last"].astype(np.float64), n=NPTS)
        run, eps, C = plateau_stats(X)
        if run:
            plw[tag].append(run["width_dec"])
            pls[tag].append(run["slope"])
            results["traj"][f"{tag}/{nm}"]["plateau"] = dict(slope=run["slope"], width=run["width_dec"])
        print(f"{tag:9s} {nm:30s} plateau slope={run['slope']:.2f} width={run['width_dec']:.2f} dec"
              if run else f"{tag:9s} {nm:30s} no plateau")
results["plateau"] = dict(ref=ref_pl, mamba=plw, deepseek=plw)
for tag in ["mamba", "deepseek"]:
    v = np.array(plw[tag])
    print(f"{tag}: plateau width median {np.median(v):.2f} dec, max {v.max():.2f} dec "
          f"(henon {ref_pl['henon']['width_dec']:.2f}, lorenz {ref_pl['lorenz']['width_dec']:.2f})")
# ---- figures ----
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (name, X, pub) in [(axs[0], ("Henon map (2-D states)", H, "project 1.157")),
                           (axs[1], ("Lorenz x[::4], m=8, tau=4", Lx, "project 1.984"))]:
    c = d2_curve(X, wide=True)
    ax.loglog(c["eps"], np.maximum(c["C"], 1e-12), "o-", ms=3, color="#333")
    if c["fit_range"]:
        ax.axvspan(c["fit_range"][0], c["fit_range"][1], color="C2", alpha=0.2)
    ax.set_title(f"{name}: D2={c['d2']:.2f} ({pub}),\nplateau {c['width_dec']:.1f} decades, $r^2$={c['r2']:.3f}")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$C(\varepsilon)$")
    ax.grid(alpha=0.3, which="both")
ax = axs[2]
cols = {"mamba": C_MAM, "deepseek": C_DS}
for tag in ["mamba", "deepseek"]:
    for name, c in examples.get(tag, [])[:2]:
        ax.loglog(c["eps"], np.maximum(c["C"], 1e-12), "o-", ms=3, color=cols[tag],
                  alpha=0.9, label=f"{tag}: {name[:14]}")
        if c["fit_range"]:
            ax.axvspan(c["fit_range"][0], c["fit_range"][1], color=cols[tag], alpha=0.10)
ax.set_title("Model trajectories through the same detector:\nthe control gives *nicer-looking* plateaus (D2 1.7-3.6) than the LLM")
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$C(\varepsilon)$")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5_d2.png"), dpi=190); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, key, ylab, ttl in [
        (axs[0], "d2", "fitted $D_2$ (best-plateau slope)",
         "(a) Fitted D2: the contracting control gives LOWER, prettier values\n(D2 1.7-3.6) than the transformer (2.0-10.9); neither is chaos evidence"),
        (axs[1], "plateau", "flat-slope plateau width (decades of $\\varepsilon$)",
         "(b) Plateau width (project criterion): Henon/Lorenz 2.6/1.0 decades;\nboth models' trajectories stay below ~0.4 decades")]:
    data = []
    labels = []
    for tag in ["mamba", "deepseek"]:
        data.append(d2s[tag] if key == "d2" else plw[tag])
        labels.append(f"{tag} (n={len(data[-1])})")
    for ref_name in ["henon", "lorenz"]:
        if key == "d2":
            data.append([ref_pl[ref_name]["slope"]])
        else:
            data.append([ref_pl[ref_name]["width_dec"]])
        labels.append(ref_name)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], [C_MAM, C_DS, "#666", "#666"]):
        p.set_facecolor(c)
        p.set_alpha(0.6)
    ax.set_ylabel(ylab)
    ax.set_title(ttl)
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5_d2_summary.png"), dpi=190); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2))
shown = 0
for tag, root in [("mamba", MAM_TRAJ), ("deepseek", DS_TRAJ)]:
    f = sorted(glob.glob(os.path.join(root, "*.npz")))[0]
    X = prep(np.load(f)["h_last"].astype(np.float64), n=800)
    eps = eps_grid(X, n=200, lo=1, hi=50)
    C = correlation_curve(X, eps, theiler=0)
    e = eps[int(np.argmin(np.abs(C - 0.02)))]
    sq = (X * X).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0))
    R = (D <= e)
    ax = axs[shown]
    ax.imshow(R, cmap="Greys", origin="lower", extent=[0, len(X), 0, len(X)])
    ax.set_title(f"{tag}: {os.path.basename(f)[:-4][:22]}, RR={R.mean():.3f}")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$t$")
    shown += 1
fig.suptitle("Recurrence plots at matched RR = 2%: the control shows band/patch structure\nof the same kind as the transformer -- not a chaos-specific signature", y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c5_rp.png"), dpi=190, bbox_inches="tight"); plt.close(fig)

json.dump(results, open(os.path.join(ROOT, "results", "ctrl_local", "d2_summary.json"), "w"), indent=1, default=float)
print("\nwrote", OUT)
