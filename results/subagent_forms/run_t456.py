"""Tasks 4-6: window dependence, radius dependence, depth dependence.

A) lambda_early and rise/decay rates vs attention window W (window_fixed, same
   tokens, W in {100,150,200,300,1000}; cross-check w100_layers/w200/w500).
B) decay tau(r) and rise rate vs r (w100_layers, 12 radii): scaling tau ~ r^beta?
C) per-layer gain A(l) ~ l^p with bootstrap CI; consistency across r and W.
"""
import os, json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitlib import fit_exp, fit_segments, find_peak, _ols

warnings.filterwarnings("ignore")
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"

def boot_med_ci(v, nboot=2000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan, np.nan, np.nan, len(v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (nboot, len(v)))
    vals = np.median(v[idx], axis=1)
    lo, hi = np.quantile(vals, [.025, .975])
    return float(np.median(v)), float(lo), float(hi), len(v)

def curve_rates(y, T):
    """y: [T] single curve. Returns dict: lam_early (slope ln y on t in [1,50]),
    lam_rise (segment), lam_decay, tau_exp, t_peak."""
    t = np.arange(T)
    m = (t >= 1) & (y > 0)
    t_, y_ = t[m], y[m]
    out = {}
    m50 = t_ <= 50
    if m50.sum() >= 5:
        out["lam_early"] = fit_exp(t_[m50], np.log(y_[m50]))["params"]["lam"]
    tpk = find_peak(t_, y_)
    out["t_peak"] = tpk
    seg = fit_segments(t_, y_, tpk)
    out["lam_rise"] = seg.get("rise", {}).get("params", {}).get("lam", np.nan)
    out["lam_decay"] = seg.get("decay-exp", {}).get("params", {}).get("lam", np.nan)
    out["tau_exp"] = (-1 / out["lam_decay"]) if np.isfinite(out.get("lam_decay", np.nan)) else np.nan
    return out

# ---------------- A) window_fixed ----------------
WIN = [100, 150, 200, 300, 1000]
wf = {}
for W in WIN:
    z = np.load(os.path.join(BASE, "window_fixed", f"w{W}", "a1.npz"))
    wf[W] = dict(dh=z["dh_norm"].astype(np.float64), radii=z["radii"], layers=z["layers"])
wfL = list(wf[100]["layers"])           # [7,14,21,27]
wfR = list(wf[100]["radii"])            # [0.01,0.1,0.3]
Tw = wf[100]["dh"].shape[-1]            # 400

win_stats = {}
for W in WIN:
    dh = wf[W]["dh"]
    for ri, r in enumerate(wfR):
        for L in wfL:
            li = wfL.index(L)
            rates = [curve_rates(dh[ri, di, li, :], Tw) for di in range(dh.shape[1])]
            for key in ["lam_early", "lam_rise", "lam_decay", "tau_exp", "t_peak"]:
                med, lo, hi, n = boot_med_ci([q[key] for q in rates])
                win_stats[(W, float(r), int(L), key)] = dict(med=med, lo=lo, hi=hi, n=n)
print("=== A) window_fixed (same tokens) ===")
for key in ["lam_early", "lam_rise", "lam_decay"]:
    for L in [27, 14]:
        print(f"-- {key}, L={L}")
        for r in [0.01, 0.1, 0.3]:
            row = [f"W={W}: {win_stats[(W, r, L, key)]['med']:+.4f}[{win_stats[(W, r, L, key)]['lo']:+.4f},{win_stats[(W, r, L, key)]['hi']:+.4f}]"
                   for W in WIN]
            print(f"   r={r:g}  " + "  ".join(row))

# cross-dataset windows (different token streams!)
cd = {}
for W, run, rsub in [(100, "w100_layers", None), (200, "w200", None), (500, "w500", None)]:
    z = np.load(os.path.join(BASE, run, "a1.npz"))
    cd[W] = dict(dh=z["dh_norm"].astype(np.float64), radii=list(z["radii"]), layers=list(z["layers"]))
cd_stats = {}
for W in cd:
    dh, Rs, Ls = cd[W]["dh"], cd[W]["radii"], cd[W]["layers"]
    for r in [0.01, 0.1, 0.3]:
        if r not in Rs: continue
        ri = Rs.index(r)
        for L in [7, 14, 21, 27]:
            if L not in Ls: continue
            li = Ls.index(L)
            rates = [curve_rates(dh[ri, di, li, :], dh.shape[-1]) for di in range(dh.shape[1])]
            for key in ["lam_early", "lam_rise", "lam_decay"]:
                cd_stats[(W, r, L, key)] = boot_med_ci([q[key] for q in rates])
print("\ncross-dataset (different token streams), lam_rise L=27:")
for r in [0.01, 0.1, 0.3]:
    print(f"  r={r}: " + "  ".join(f"W={W}: {cd_stats[(W, r, 27, 'lam_rise')][0]:+.4f}" for W in [100, 200, 500] if (W, r, 27, 'lam_rise') in cd_stats))

# ---------------- B) radius dependence (w100_layers, all 12 radii) ----------------
z = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = z["radii"]; layers = z["layers"]; dh0 = z["dh_norm"].astype(np.float64)
rad_stats = {}
for ri, r in enumerate(radii):
    for L in [7, 27]:
        li = int(np.where(layers == L)[0][0])
        rates = [curve_rates(dh0[ri, di, li, :], dh0.shape[-1]) for di in range(dh0.shape[1])]
        for key in ["lam_rise", "lam_decay", "tau_exp", "t_peak"]:
            rad_stats[(float(r), L, key)] = boot_med_ci([q[key] for q in rates])

def loglog_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([np.ones(m.sum()), np.log(x[m])])
    beta, cov, rss, r2, n = _ols(X, np.log(y[m]))
    return beta[1], np.sqrt(cov[1, 1]), r2, n

# tau ~ r^beta over r<=0.3 at L=27
rs = np.array([r for r in radii if r <= 0.31])
tau_med = np.array([rad_stats[(float(r), 27, "tau_exp")][0] for r in rs])
tau_lo = np.array([rad_stats[(float(r), 27, "tau_exp")][1] for r in rs])
tau_hi = np.array([rad_stats[(float(r), 27, "tau_exp")][2] for r in rs])
b_tau, se_tau, r2_tau, n_tau = loglog_slope(rs, tau_med)
# bootstrap beta over dirs
rng = np.random.default_rng(3)
n_dir = dh0.shape[1]
bs = []
for b in range(500):
    idx = rng.integers(0, n_dir, n_dir)
    tvals = []
    for r in rs:
        ri = int(np.where(radii == r)[0][0])
        vals = [curve_rates(dh0[ri, i, -1, :], 300)["tau_exp"] for i in idx]
        tvals.append(np.median(vals))
    bs.append(loglog_slope(rs, np.array(tvals))[0])
b_lo, b_hi = np.quantile(bs, [.025, .975])
print(f"\n=== B) tau ~ r^beta (L=27, r<=0.3): beta={b_tau:+.3f} (OLS se {se_tau:.3f}), boot 95% [{b_lo:+.3f},{b_hi:+.3f}], R2={r2_tau:.3f}")
# all radii including large
rs_all = radii.astype(float)
tau_all = np.array([rad_stats[(float(r), 27, "tau_exp")][0] for r in rs_all])
print("tau_exp vs r (all):", np.round(tau_all, 1))
print("lam_rise vs r (L=27):", np.round([rad_stats[(float(r), 27, 'lam_rise')][0] for r in rs_all], 4))
print("lam_decay vs r (L=27):", np.round([rad_stats[(float(r), 27, 'lam_decay')][0] for r in rs_all], 4))
print("t_peak vs r (L=27):", [rad_stats[(float(r), 27, 't_peak')][0] for r in rs_all])

# ---------------- C) depth ----------------
def depth_gain(dh, Rs, Ls, ri, r, tpoint=1):
    """median_d a(tpoint, l) per layer."""
    return np.array([np.median(dh[ri, :, Ls.index(L), tpoint]) for L in Ls])

Lfit = [3, 7, 14, 21, 27]
depth_res = {}
for r in [0.01, 0.1, 0.3, 1.0, 2.0]:
    ri = int(np.argmin(np.abs(radii - r)))
    A1 = depth_gain(dh0, radii, Lfit, ri, r, 1)
    p, se, r2, n = loglog_slope(np.array(Lfit, float), A1)
    # bootstrap over dirs
    bs = []
    for b in range(500):
        idx = rng.integers(0, n_dir, n_dir)
        Ab = np.array([np.median(dh0[ri, idx, li, 1]) for li in [int(np.where(layers == L)[0][0]) for L in Lfit]])
        bs.append(loglog_slope(np.array(Lfit, float), Ab)[0])
    lo, hi = np.quantile(bs, [.025, .975])
    depth_res[("w100_layers", float(r))] = dict(A1=A1.tolist(), p=float(p), se=float(se), r2=float(r2),
                                                boot_lo=float(lo), boot_hi=float(hi))
    print(f"C) w100_layers r={r:g}: A(1)={[f'{a:.3g}' for a in A1]}  p={p:+.3f} [{lo:+.3f},{hi:+.3f}] R2={r2:.3f}")
# window_fixed depth (W=100..1000, layers 7,14,21,27)
for W in WIN:
    for r in wfR:
        ri = wfR.index(r)
        A1 = np.array([np.median(wf[W]["dh"][ri, :, wfL.index(L), 1]) for L in wfL])
        p, se, r2, n = loglog_slope(np.array(wfL, float), A1)
        depth_res[(f"window_fixed_w{W}", float(r))] = dict(A1=A1.tolist(), p=float(p), se=float(se), r2=float(r2))
    print(f"C) window_fixed W={W}: p per r =", {r: round(depth_res[(f'window_fixed_w{W}', r)]['p'], 3) for r in wfR})
# rise rate vs layer
lam_by_L = {}
for L in [7, 14, 21, 27]:
    li = int(np.where(layers == L)[0][0])
    ri = int(np.argmin(np.abs(radii - 0.1)))
    vals = [curve_rates(dh0[ri, di, li, :], 300)["lam_rise"] for di in range(n_dir)]
    lam_by_L[L] = boot_med_ci(vals)
print("C) lam_rise by layer (r=0.1):", {L: np.round(lam_by_L[L][0], 4) for L in lam_by_L})

# ---------------- figures ----------------
# fig A: windows
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axs[0]
cols = {0.01: "tab:blue", 0.1: "tab:red", 0.3: "tab:purple"}
for r in wfR:
    xs = np.array(WIN, float)
    ys = np.array([win_stats[(W, r, 27, "lam_early")]["med"] for W in WIN])
    lo = np.array([win_stats[(W, r, 27, "lam_early")]["lo"] for W in WIN])
    hi = np.array([win_stats[(W, r, 27, "lam_early")]["hi"] for W in WIN])
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, capsize=3,
                color=cols[r], label=f"r={r:g}")
ax.set_xscale("log"); ax.set_xlabel("attention window W (tokens)"); ax.set_ylabel(r"$\lambda_{\rm early}$ (t$<$50)")
ax.axhline(0, color="k", lw=.5)
ax.set_title("$\\lambda_{early}$ vs W (L=27): window-insensitive"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[1]
for r in wfR:
    ys = np.array([win_stats[(W, r, 27, "lam_rise")]["med"] for W in WIN])
    lo = np.array([win_stats[(W, r, 27, "lam_rise")]["lo"] for W in WIN])
    hi = np.array([win_stats[(W, r, 27, "lam_rise")]["hi"] for W in WIN])
    ax.errorbar(WIN, ys, yerr=[ys - lo, hi - ys], marker="s", ms=4, capsize=3, color=cols[r], label=f"r={r:g}")
ax.set_xscale("log"); ax.set_xlabel("W"); ax.set_ylabel(r"$\lambda_{\rm rise}$ (segment fit)")
ax.set_title("rise rate vs W (L=27)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[2]
for r in wfR:
    ys = np.array([win_stats[(W, r, 27, "tau_exp")]["med"] for W in WIN])
    lo = np.array([win_stats[(W, r, 27, "tau_exp")]["lo"] for W in WIN])
    hi = np.array([win_stats[(W, r, 27, "tau_exp")]["hi"] for W in WIN])
    ax.errorbar(WIN, ys, yerr=[ys - lo, hi - ys], marker="^", ms=4, capsize=3, color=cols[r], label=f"r={r:g}")
ax.set_xscale("log"); ax.set_xlabel("W"); ax.set_ylabel(r"$\tau_{\rm decay}$ (tokens)")
ax.set_title("decay time vs W (L=27)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle("Task 4: attention-window dependence (same tokens, teacher-forced, 8 dirs)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_windows.png"), dpi=160); plt.close(fig)

# fig B: radius
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axs[0]
for L, c in [(7, "tab:blue"), (27, "tab:red")]:
    ys = np.array([rad_stats[(float(r), L, "lam_rise")][0] for r in radii])
    lo = np.array([rad_stats[(float(r), L, "lam_rise")][1] for r in radii])
    hi = np.array([rad_stats[(float(r), L, "lam_rise")][2] for r in radii])
    ax.errorbar(radii, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, capsize=3, color=c, label=f"L={L}")
ax.set_xscale("log"); ax.axhline(0, color="k", lw=.5)
ax.set_xlabel("r"); ax.set_ylabel(r"$\lambda_{\rm rise}$"); ax.legend(fontsize=8)
ax.set_title("rise rate: flat for r$\\le$0.3, drops at r$\\ge$0.5"); ax.grid(alpha=.3)
ax = axs[1]
ys = np.array([rad_stats[(float(r), 27, "lam_decay")][0] for r in radii])
lo = np.array([rad_stats[(float(r), 27, "lam_decay")][1] for r in radii])
hi = np.array([rad_stats[(float(r), 27, "lam_decay")][2] for r in radii])
ax.errorbar(radii, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, capsize=3, color="tab:red")
ax.set_xscale("log"); ax.set_xlabel("r"); ax.set_ylabel(r"$\lambda_{\rm decay}$ (L=27)")
ax.set_title("decay rate: constant for r$\\le$0.3"); ax.grid(alpha=.3)
ax = axs[2]
ax.errorbar(radii, tau_all, yerr=[tau_all - np.array([rad_stats[(float(r), 27, 'tau_exp')][1] for r in radii]),
                                 np.array([rad_stats[(float(r), 27, 'tau_exp')][2] for r in radii]) - tau_all],
            marker="o", ms=4, capsize=3, color="k")
ax.plot(rs, np.exp(np.log(tau_med[0]) + b_tau * np.log(rs / rs[0])), "r--",
        label=f"$\\tau\\sim r^{{{b_tau:+.2f}}}$ (r$\\le$0.3)")
ax.set_xscale("log"); ax.set_xlabel("r"); ax.set_ylabel(r"$\tau_{\rm exp}$ (L=27)")
ax.set_title(f"decay time: $\\beta$={b_tau:+.3f} [{b_lo:+.3f},{b_hi:+.3f}] (no scaling)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle("Task 5: radius dependence of rise/decay (w100_layers, 20 dirs)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_radius.png"), dpi=160); plt.close(fig)

# fig C: depth
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axs[0]
for r, c in [(0.01, "tab:blue"), (0.1, "tab:red"), (0.3, "tab:purple")]:
    A1 = np.array(depth_res[("w100_layers", r)]["A1"]); p = depth_res[("w100_layers", r)]["p"]
    ax.loglog(Lfit, A1, "o", color=c, label=f"r={r:g}: p={p:+.2f}")
    ax.loglog(Lfit, A1[0] * (np.array(Lfit, float) / Lfit[0]) ** p, "-", color=c, alpha=.6)
ax.set_xlabel("layer l"); ax.set_ylabel("A(1) = median $a(1,l)$")
ax.set_title("early gain vs depth (w100)"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
ax = axs[1]
labels, ps, los, his = [], [], [], []
for (ds, r), v in depth_res.items():
    if "boot_lo" not in v: continue
    labels.append(f"{ds[-11:]},{r:g}"); ps.append(v["p"]); los.append(v["boot_lo"]); his.append(v["boot_hi"])
ypos = np.arange(len(labels))
ax.errorbar(ps, ypos, xerr=[np.array(ps) - np.array(los), np.array(his) - np.array(ps)], fmt="o", ms=4, capsize=3)
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7)
ax.axvline(0, color="k", lw=.5); ax.set_xlabel("p in A(l) ~ l^p [95% boot CI]")
ax.set_title("depth exponent stable across radius (w100)"); ax.grid(alpha=.3)
ax = axs[2]
for r in wfR:
    ps = [depth_res[(f"window_fixed_w{W}", r)]["p"] for W in WIN]
    ax.plot(WIN, ps, marker="o", label=f"r={r:g}")
ax.set_xscale("log"); ax.set_xlabel("W"); ax.set_ylabel("p (window_fixed, l=7..27)")
ax.axhline(0, color="k", lw=.5); ax.set_title("p vs window: stable ~ +0.30"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle("Task 6: depth scaling of the perturbation gain")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_depth.png"), dpi=160); plt.close(fig)

# save JSON
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o
with open(os.path.join(OUT, "t456_summary.json"), "w") as f:
    json.dump(clean(dict(win_stats=win_stats, cd_stats={f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in cd_stats.items()},
                         rad_stats=rad_stats, tau_beta=dict(beta=float(b_tau), boot_lo=float(b_lo), boot_hi=float(b_hi), r2=float(r2_tau)),
                         depth_res=depth_res, lam_by_layer={str(k): v for k, v in lam_by_L.items()})), f, indent=1)
print("saved fig_windows.png, fig_radius.png, fig_depth.png, t456_summary.json")
