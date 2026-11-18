"""Tasks 4-6 (fixed): window dependence, radius dependence, depth dependence.

A) lambda_early / rise / decay rates vs attention window W (window_fixed, same
   tokens, W in {100,150,200,300,1000}; cross-check w100_layers/w200/w500).
B) decay tau(r): scaling tau ~ r^beta?
C) per-layer gain A(l) ~ l^p with bootstrap CI; rise-rate depth law; windows.
Dumps t456_summary.json BEFORE making figures; each figure in try/except.
"""
import os, json, traceback, warnings
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
    t = np.arange(T); m = (t >= 1) & (y > 0)
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

_rate_cache = {}
def get_rates(dh, Ls, ri, L, tag):
    ck = (tag, ri, int(L))
    if ck in _rate_cache: return _rate_cache[ck]
    li = Ls.index(L)
    rates = [curve_rates(dh[ri, di, li, :], dh.shape[-1]) for di in range(dh.shape[1])]
    _rate_cache[ck] = rates
    return rates

def loglog_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (np.asarray(y) > 0)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    X = np.column_stack([np.ones(m.sum()), np.log(x[m])])
    beta, cov, rss, r2, n = _ols(X, np.log(y[m]))
    return beta[1], np.sqrt(cov[1, 1]), r2, n

# ---------------- A) window_fixed ----------------
WIN = [100, 150, 200, 300, 1000]
wf, wfL, wfR = {}, None, None
for W in WIN:
    z = np.load(os.path.join(BASE, "window_fixed", f"w{W}", "a1.npz"))
    wf[W] = dict(dh=z["dh_norm"].astype(np.float64), radii=list(z["radii"]), layers=list(z["layers"]))
wfL = wf[100]["layers"]; wfR = wf[100]["radii"]

win_stats = {}
for W in WIN:
    for ri, r in enumerate(wfR):
        for L in wfL:
            rates = get_rates(wf[W]["dh"], wfL, ri, L, f"wf{W}")
            for key in ["lam_early", "lam_rise", "lam_decay", "tau_exp", "t_peak"]:
                med, lo, hi, n = boot_med_ci([q[key] for q in rates])
                win_stats[(W, float(r), int(L), key)] = dict(med=med, lo=lo, hi=hi, n=n)
print("=== A) window_fixed (same tokens) ===")
for key in ["lam_early", "lam_rise", "lam_decay"]:
    for L in [27, 14]:
        print(f"-- {key}, L={L}")
        for r in wfR:
            row = [f"W={W}: {win_stats[(W, r, L, key)]['med']:+.4f}[{win_stats[(W, r, L, key)]['lo']:+.4f},{win_stats[(W, r, L, key)]['hi']:+.4f}]"
                   for W in WIN]
            print(f"   r={r:g}  " + "  ".join(row))
print("note: window truncation starts at t = W-65; early segments identical for W>=150.")

cd_stats = {}
for W, run in [(100, "w100_layers"), (200, "w200"), (500, "w500")]:
    z = np.load(os.path.join(BASE, run, "a1.npz"))
    Rs, Ls = list(z["radii"]), list(z["layers"])
    dh = z["dh_norm"].astype(np.float64)
    for r in [0.01, 0.1, 0.3]:
        if r not in Rs: continue
        ri = Rs.index(r)
        for L in [7, 14, 21, 27]:
            if L not in Ls: continue
            rates = get_rates(dh, Ls, ri, L, run)
            for key in ["lam_early", "lam_rise", "lam_decay"]:
                cd_stats[(W, r, L, key)] = boot_med_ci([q[key] for q in rates])
print("\ncross-dataset (different token streams), lam_rise L=27:")
for r in [0.01, 0.1, 0.3]:
    print(f"  r={r}: " + "  ".join(
        f"W={W}: {cd_stats[(W, r, 27, 'lam_rise')][0]:+.4f}[{cd_stats[(W, r, 27, 'lam_rise')][1]:+.4f},{cd_stats[(W, r, 27, 'lam_rise')][2]:+.4f}]"
        for W in [100, 200, 500] if (W, r, 27, 'lam_rise') in cd_stats))

# ---------------- B) radius dependence ----------------
z = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = list(z["radii"]); layers = list(z["layers"]); dh0 = z["dh_norm"].astype(np.float64)
n_dir = dh0.shape[1]
rad_stats = {}
for ri, r in enumerate(radii):
    for L in [7, 27]:
        rates = get_rates(dh0, layers, ri, L, "w100_layers")
        for key in ["lam_rise", "lam_decay", "tau_exp", "t_peak"]:
            rad_stats[(float(r), L, key)] = boot_med_ci([q[key] for q in rates])

rs = np.array([r for r in radii if r <= 0.31])
rates27 = {float(r): get_rates(dh0, layers, int(np.where(np.array(radii) == r)[0][0]), 27, "w100_layers") for r in rs}
tau_med = np.array([rad_stats[(float(r), 27, "tau_exp")][0] for r in rs])
b_tau, se_tau, r2_tau, n_tau = loglog_slope(rs, tau_med)
rng = np.random.default_rng(3)
bs = []
for b in range(500):
    idx = rng.integers(0, n_dir, n_dir)
    tvals = np.array([np.median([rates27[float(r)][i]["tau_exp"] for i in idx]) for r in rs])
    bs.append(loglog_slope(rs, tvals)[0])
b_lo, b_hi = np.quantile(bs, [.025, .975])
print(f"\n=== B) tau ~ r^beta (L=27, r<=0.3): beta={b_tau:+.3f}, boot 95% [{b_lo:+.3f},{b_hi:+.3f}], R2={r2_tau:.3f}")
rs_all = np.array(radii, float)
tau_all = np.array([rad_stats[(float(r), 27, "tau_exp")][0] for r in radii])
lam_all = np.array([rad_stats[(float(r), 27, "lam_rise")][0] for r in radii])
print("tau_exp vs r:", np.round(tau_all, 1))
print("lam_rise vs r:", np.round(lam_all, 4))

# ---------------- C) depth ----------------
Lfit = [3, 7, 14, 21, 27]
Laxis = [int(np.where(np.array(layers) == L)[0][0]) for L in Lfit]
depth_res = {}
for r in [0.01, 0.1, 0.3, 1.0, 2.0]:
    ri = int(np.argmin(np.abs(np.array(radii) - r)))
    A1 = np.array([np.median(dh0[ri, :, ai, 1]) for ai in Laxis])
    p, se, r2, n = loglog_slope(np.array(Lfit, float), A1)
    bs = []
    for b in range(500):
        idx = rng.integers(0, n_dir, n_dir)
        Ab = np.array([np.median(dh0[ri, idx, ai, 1]) for ai in Laxis])
        bs.append(loglog_slope(np.array(Lfit, float), Ab)[0])
    lo, hi = np.quantile(bs, [.025, .975])
    depth_res[("w100_layers", float(r))] = dict(A1=A1.tolist(), p=float(p), se=float(se), r2=float(r2),
                                                boot_lo=float(lo), boot_hi=float(hi))
    # subfit without l=3
    p2, _, r22, _ = loglog_slope(np.array(Lfit[1:], float), A1[1:])
    depth_res[("w100_layers", float(r))]["p_noL3"] = float(p2)
    depth_res[("w100_layers", float(r))]["r2_noL3"] = float(r22)
    print(f"C) w100_layers r={r:g}: A(1)={[f'{a:.3g}' for a in A1]}  p={p:+.3f} [{lo:+.3f},{hi:+.3f}] R2={r2:.3f} "
          f"| noL3: p={p2:+.3f} R2={r22:.3f}")

# rise-rate depth law (r=0.1)
lamL = {}
for L in [7, 14, 21, 27]:
    lamL[L] = boot_med_ci([q["lam_rise"] for q in get_rates(dh0, layers, int(np.argmin(np.abs(np.array(radii) - 0.1))), L, "w100_layers")])
Ls4 = np.array([7, 14, 21, 27], float)
lam4 = np.array([lamL[L][0] for L in [7, 14, 21, 27]])
p_lam, se_lam, r2_lam, _ = loglog_slope(Ls4, lam4)
bs = []
ri = int(np.argmin(np.abs(np.array(radii) - 0.1)))
for b in range(500):
    idx = rng.integers(0, n_dir, n_dir)
    vals = np.array([np.median([get_rates(dh0, layers, ri, L, "w100_layers")[i]["lam_rise"] for i in idx])
                     for L in [7, 14, 21, 27]])
    bs.append(loglog_slope(Ls4, vals)[0])
plo, phi = np.quantile(bs, [.025, .975])
print(f"C) lam_rise(l) ~ l^p: p={p_lam:+.3f} [{plo:+.3f},{phi:+.3f}] R2={r2_lam:.3f}; lam by L:", 
      {L: round(lamL[L][0], 4) for L in lamL})

# window_fixed depth: p at t=1 (identical, window inactive) and t=150 (active for W<=200)
depth_win = {}
for W in WIN:
    try:
        for r in wfR:
            ri = wfR.index(r)
            for tp in [1, 150]:
                A1 = np.array([np.median(wf[W]["dh"][ri, :, wfL.index(L), tp]) for L in wfL])
                p, se, r2, n = loglog_slope(np.array(wfL, float), A1)
                depth_win[(W, float(r), tp)] = dict(A=A1.tolist(), p=float(p), r2=float(r2))
        ri = wfR.index(0.1)
        lamw = [win_stats[(W, 0.1, L, "lam_rise")]["med"] for L in wfL]
        pw, _, r2w, _ = loglog_slope(np.array(wfL, float), np.array(lamw))
        depth_win[(W, 0.1, "lam")] = dict(p=float(pw), r2=float(r2w), lam=lamw)
        print(f"C) window_fixed W={W}: p(A1,r=0.1)={depth_win[(W, 0.1, 1)]['p']:+.3f}  p(A150)={depth_win[(W, 0.1, 150)]['p']:+.3f}  "
              f"p(lam_rise)={pw:+.3f}")
    except Exception:
        print(f"depth_win failed at W={W}\n" + traceback.format_exc())

# ---- dump before figures ----
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)): return float(o)
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o
with open(os.path.join(OUT, "t456_summary.json"), "w") as f:
    json.dump(clean(dict(win_stats=win_stats, cd_stats=cd_stats, rad_stats=rad_stats,
                         tau_beta=dict(beta=float(b_tau), boot_lo=float(b_lo), boot_hi=float(b_hi), r2=float(r2_tau)),
                         depth_res=depth_res, depth_win=depth_win, lamL=lamL,
                         p_lam=dict(p=float(p_lam), boot_lo=float(plo), boot_hi=float(phi), r2=float(r2_lam)))), f, indent=1)
print("saved t456_summary.json")

# ---------------- figures ----------------
figs = {}
try:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    cols = {0.01: "tab:blue", 0.1: "tab:red", 0.3: "tab:purple"}
    for r in wfR:
        xs = np.array(WIN, float)
        ys = np.array([win_stats[(W, r, 27, "lam_early")]["med"] for W in WIN])
        lo = np.array([win_stats[(W, r, 27, "lam_early")]["lo"] for W in WIN])
        hi = np.array([win_stats[(W, r, 27, "lam_early")]["hi"] for W in WIN])
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, capsize=3, color=cols[r], label=f"r={r:g}")
    ax.set_xscale("log"); ax.set_xlabel("attention window W (tokens)"); ax.set_ylabel(r"$\lambda_{\rm early}$ (t$<$50)")
    ax.axhline(0, color="k", lw=.5)
    ax.set_title("early rate vs W (L=27): W$\\geq$150 identical (window inactive)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axs[1]
    for r in wfR:
        ys = np.array([win_stats[(W, r, 27, "lam_rise")]["med"] for W in WIN])
        lo = np.array([win_stats[(W, r, 27, "lam_rise")]["lo"] for W in WIN])
        hi = np.array([win_stats[(W, r, 27, "lam_rise")]["hi"] for W in WIN])
        ax.errorbar(WIN, ys, yerr=[ys - lo, hi - ys], marker="s", ms=4, capsize=3, color=cols[r], label=f"r={r:g}")
    ax.set_xscale("log"); ax.set_xlabel("W"); ax.set_ylabel(r"$\lambda_{\rm rise}$ (segment fit)")
    ax.axhline(0, color="k", lw=.5)
    ax.set_title("rise exists ONLY at W=100 (truncation-active)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axs[2]
    for r in wfR:
        ys = np.array([win_stats[(W, r, 27, "lam_decay")]["med"] for W in WIN])
        lo = np.array([win_stats[(W, r, 27, "lam_decay")]["lo"] for W in WIN])
        hi = np.array([win_stats[(W, r, 27, "lam_decay")]["hi"] for W in WIN])
        ax.errorbar(WIN, ys, yerr=[ys - lo, hi - ys], marker="^", ms=4, capsize=3, color=cols[r], label=f"r={r:g}")
    ax.set_xscale("log"); ax.set_xlabel("W"); ax.set_ylabel(r"$\lambda_{\rm decay}$")
    ax.axhline(0, color="k", lw=.5)
    ax.set_title("decay weakens monotonically with W; sign flip at W=1000")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Task 4: attention-window dependence (same tokens, teacher-forced, 8 dirs)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_windows.png"), dpi=160); plt.close(fig)
    figs["fig_windows.png"] = "ok"
except Exception:
    print("fig_windows failed\n" + traceback.format_exc())

try:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axs[0]
    for L, c in [(7, "tab:blue"), (27, "tab:red")]:
        ys = np.array([rad_stats[(float(r), L, "lam_rise")][0] for r in radii])
        lo = np.array([rad_stats[(float(r), L, "lam_rise")][1] for r in radii])
        hi = np.array([rad_stats[(float(r), L, "lam_rise")][2] for r in radii])
        ax.errorbar(radii, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, capsize=3, color=c, label=f"L={L}")
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("r"); ax.set_ylabel(r"$\lambda_{\rm rise}$"); ax.legend(fontsize=8)
    ax.set_title("rise rate ~ constant through r=2 (linear regime)"); ax.grid(alpha=.3)
    ax = axs[1]
    ys = np.array([rad_stats[(float(r), 27, "lam_decay")][0] for r in radii])
    lo = np.array([rad_stats[(float(r), 27, "lam_decay")][1] for r in radii])
    hi = np.array([rad_stats[(float(r), 27, "lam_decay")][2] for r in radii])
    ax.errorbar(radii, ys, yerr=[ys - lo, hi - ys], marker="o", ms=4, capsize=3, color="tab:red")
    ax.set_xscale("log"); ax.set_xlabel("r"); ax.set_ylabel(r"$\lambda_{\rm decay}$ (L=27)")
    ax.set_title("decay rate: constant for r$\\leq$0.3"); ax.grid(alpha=.3)
    ax = axs[2]
    tlo = np.array([rad_stats[(float(r), 27, "tau_exp")][1] for r in radii])
    thi = np.array([rad_stats[(float(r), 27, "tau_exp")][2] for r in radii])
    ax.errorbar(radii, tau_all, yerr=[tau_all - tlo, thi - tau_all], marker="o", ms=4, capsize=3, color="k")
    ax.plot(rs, tau_med[0] * (rs / rs[0]) ** b_tau, "r--",
            label=fr"$\tau\sim r^{{{b_tau:+.2f}}}$  CI[{b_lo:+.2f},{b_hi:+.2f}]")
    ax.set_xscale("log"); ax.set_xlabel("r"); ax.set_ylabel(r"$\tau_{\rm exp}$ (L=27)")
    ax.set_title("decay time effectively flat ($\\beta\\approx-0.03$)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Task 5: radius dependence of rise/decay (w100_layers, 20 dirs, 95% boot CIs)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_radius.png"), dpi=160); plt.close(fig)
    figs["fig_radius.png"] = "ok"
except Exception:
    print("fig_radius failed\n" + traceback.format_exc())

try:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    for r, c in [(0.01, "tab:blue"), (0.1, "tab:red"), (0.3, "tab:purple")]:
        A1 = np.array(depth_res[("w100_layers", r)]["A1"])
        p = depth_res[("w100_layers", r)]["p"]
        ax.loglog(Lfit, A1, "o", color=c, label=f"r={r:g}: p={p:+.2f}")
        ax.loglog(Lfit, A1[0] * (np.array(Lfit, float) / Lfit[0]) ** p, "-", color=c, alpha=.5)
    ax.set_xlabel("layer l"); ax.set_ylabel("A(1) = median $a(t{=}1,l)$")
    ax.set_title("early gain vs depth (w100, L3..L27)"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    ax = axs[1]
    labels, ps, los, his = [], [], [], []
    for (ds, r), v in depth_res.items():
        labels.append(f"{ds},{r:g}"); ps.append(v["p"]); los.append(v["boot_lo"]); his.append(v["boot_hi"])
    ypos = np.arange(len(labels))
    ax.errorbar(ps, ypos, xerr=[np.array(ps) - np.array(los), np.array(his) - np.array(ps)], fmt="o", ms=4, capsize=3)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="k", lw=.5); ax.set_xlabel("p in A(l) ~ l^p [95% boot CI]")
    ax.set_title("depth exponent vs radius: rises with r"); ax.grid(alpha=.3)
    ax = axs[2]
    Ls4 = [7, 14, 21, 27]
    lam4 = [lamL[L][0] for L in Ls4]
    llo = [lamL[L][1] for L in Ls4]; lhi = [lamL[L][2] for L in Ls4]
    ax.errorbar(Ls4, lam4, yerr=[np.array(lam4) - np.array(llo), np.array(lhi) - np.array(lam4)],
                marker="o", capsize=3, color="tab:red", label=fr"$\lambda_{{rise}}\sim l^{{{p_lam:+.2f}}}$ (r=0.1)")
    for W in WIN:
        lamw = depth_win[(W, 0.1, "lam")]["lam"]
        ax.plot(Ls4, lamw, "--", alpha=.7, label=f"W={W}")
    ax.set_xlabel("layer l"); ax.set_ylabel(r"$\lambda_{\rm rise}(l)$")
    ax.set_title("rise rate grows with depth, all windows"); ax.legend(fontsize=6.5); ax.grid(alpha=.3)
    fig.suptitle("Task 6: depth scaling of perturbation gain and rise rate")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_depth.png"), dpi=160); plt.close(fig)
    figs["fig_depth.png"] = "ok"
except Exception:
    print("fig_depth failed\n" + traceback.format_exc())
print("figures:", figs)
