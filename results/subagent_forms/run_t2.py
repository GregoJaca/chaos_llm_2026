"""Tasks 2+3 (final): lambda(t) = ln(a(t)/a(1))/t; distributions, bootstrap,
envelope model comparison, Jensen aggregation."""
import os, json, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitlib import _ols, _ic

warnings.filterwarnings("ignore")
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"

d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
T = dh.shape[-1]

def lam_curves(ri, li):
    y = np.maximum(dh[ri, :, li, :], 1e-300)
    t = np.arange(T)[None, :]
    lam = np.log(y / y[:, 1:2]) / np.maximum(t, 1)
    lam[:, 0] = np.nan; lam[:, 1] = 0.0
    return lam

def smooth(x, w=7):
    k = np.ones(w) / w; pad = w // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), k, mode="valid")[:len(x)]

def boot_curve_ci(y_dt, stat=np.median, nboot=2000, seed=0, smooth_w=None):
    """y [n_dir, T]; returns (point, lo, hi) of [smoothed] stat over dirs."""
    rng = np.random.default_rng(seed); n = y_dt.shape[0]
    idx = rng.integers(0, n, (nboot, n))
    vals = stat(y_dt[idx], axis=1)
    if smooth_w: vals = np.array([smooth(v, smooth_w) for v in vals])
    lo, hi = np.quantile(vals, [.025, .975], axis=0)
    pt = stat(y_dt, axis=0)
    if smooth_w: pt = smooth(pt, smooth_w)
    return pt, lo, hi

ri = int(np.argmin(np.abs(radii - 0.1))); li = int(np.where(layers == 27)[0][0])
lam = lam_curves(ri, li)                                   # [20, T]
med, lo, hi = boot_curve_ci(lam, np.median)
meds, los, his = boot_curve_ci(lam, np.median, smooth_w=7)
mean, mlo, mhi = boot_curve_ci(lam, np.mean)
means, mlos, mhis = boot_curve_ci(lam, np.mean, smooth_w=7)

def positive_runs(ts, ok):
    runs, start = [], None
    for i, t in enumerate(ts):
        if ok[i] and start is None: start = t
        if not ok[i] and start is not None: runs.append((start, t - 1)); start = None
    if start is not None: runs.append((start, ts[-1]))
    return runs

ts = np.arange(3, T)
win_med_s = positive_runs(ts, los[3:] > 0)
win_mean_s = positive_runs(ts, mlos[3:] > 0)
print("lambda>0 window, smoothed median (95% boot CI):", win_med_s)
print("lambda>0 window, smoothed mean  (95% boot CI):", win_mean_s)
frac_pos = np.mean(lam[:, 3:] > 0, axis=0)
print("fraction of dirs with lambda_d>0: t=30,50,70,90,110 ->", np.round(frac_pos[[27, 47, 67, 87, 107]], 2))

# ---- envelope model comparison on the median curve (t in [3,299]) ----
ymed = med[ts]
def linfit(X, y, k):
    beta, cov, rss, r2, n = _ols(X, y)
    aic, bic = _ic(rss, n, k)
    return dict(beta=beta.tolist(), se=np.sqrt(np.diag(cov)).tolist(), rss=rss, r2=r2, aic=aic, bic=bic, n=n)
c_models = {}
c_models["const"] = linfit(np.ones((len(ts), 1)), ymed, 1)
c_models["A-Blogt"] = linfit(np.column_stack([np.ones_like(ts, float), np.log(ts)]), ymed, 2)
best = None
for tau in np.exp(np.linspace(np.log(5), np.log(3000), 60)):
    X = np.column_stack([np.ones_like(ts, float), np.exp(-ts / tau)])
    b, cov, rss, r2, n = _ols(X, ymed)
    if best is None or rss < best[2]: best = (tau, b, rss)
X = np.column_stack([np.ones_like(ts, float), np.exp(-ts / best[0])])
c_models["Aexp(-t/tau)"] = linfit(X, ymed, 2); c_models["Aexp(-t/tau)"]["tau"] = float(best[0])
c_models["C/t-lam2"] = linfit(np.column_stack([np.ones_like(ts, float), 1.0 / ts]), ymed, 2)
best = None
for tstar in range(10, 290, 5):
    x1 = np.where(ts <= tstar, 1 - 1.0 / ts, 0.0)
    x2 = np.where(ts <= tstar, 0.0, 1.0 / ts)
    X = np.column_stack([x1, x2, np.ones_like(ts, float)])
    b, cov, rss, r2, n = _ols(X, ymed)
    if best is None or rss < best[2]: best = (tstar, b, rss)
x1 = np.where(ts <= best[0], 1 - 1.0 / ts, 0.0)
x2 = np.where(ts <= best[0], 0.0, 1.0 / ts)
X = np.column_stack([x1, x2, np.ones_like(ts, float)])
c_models["two-phase"] = linfit(X, ymed, 3); c_models["two-phase"]["t*"] = int(best[0])
print("\nenvelope fits (median lambda(t), r=0.1, L=27):")
for k, v in sorted(c_models.items(), key=lambda kv: kv[1]["bic"]):
    print(f"  {k:14s} BIC={v['bic']:8.1f}  R2={v['r2']:.3f}  beta={np.round(v['beta'],4)}")

# ---- Jensen ----
am = dh[ri, :, li, :]
mean_curve = am.mean(axis=0)
ttv = np.maximum(np.arange(T), 1)
lam_of_mean = np.log(np.maximum(mean_curve, 1e-300) / mean_curve[1]) / ttv
lam_of_mean[:2] = np.nan
rng = np.random.default_rng(1)
nb, n = 2000, am.shape[0]
b_lom = np.empty((nb, T)); b_mol = np.empty((nb, T))
for b in range(nb):
    idx = rng.integers(0, n, n)
    mc = am[idx].mean(axis=0)
    b_lom[b] = np.log(np.maximum(mc, 1e-300) / mc[1]) / ttv
    b_mol[b] = lam[idx].mean(axis=0)
b_lom[:, :2] = np.nan
lom_lo, lom_hi = np.nanquantile(b_lom, [.025, .975], axis=0)
mol_lo, mol_hi = np.nanquantile(b_mol, [.025, .975], axis=0)
gap = lam_of_mean - lam.mean(axis=0)
sel_t = [10, 30, 50, 100, 150, 200, 299]
print("\nJensen gap lam_of_mean - mean_of_lam:", {t: round(float(gap[t]), 4) for t in sel_t})

# ---- per-layer + radii collapse (smoothed medians) ----
li7 = int(np.where(layers == 7)[0][0]); li14 = int(np.where(layers == 14)[0][0])
lam7, _, _ = boot_curve_ci(lam_curves(ri, li7), np.median, nboot=1000, smooth_w=7)
lam14, _, _ = boot_curve_ci(lam_curves(ri, li14), np.median, nboot=1000, smooth_w=7)
r_lam = {}
for r in [0.001, 0.01, 0.1, 0.3]:
    rj = int(np.argmin(np.abs(radii - r)))
    m, l, h = boot_curve_ci(lam_curves(rj, li), np.median, nboot=1000, smooth_w=7)
    r_lam[r] = (m, l, h)
spread = np.array([np.ptp([r_lam[r][0][t] for r in r_lam]) for t in range(10, 200)])
ciw = np.median([r_lam[0.1][2][t] - r_lam[0.1][1][t] for t in range(10, 200)])
print(f"collapse across radii (t in [10,200]): median spread={np.median(spread):.4f} < CI width={ciw:.4f}")

# ---- figure 1: lambda(t) ----
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5))
ax = axs[0, 0]
for di in range(20):
    ax.plot(np.arange(T), lam[di], color="gray", lw=.4, alpha=.6)
ax.plot(np.arange(T), meds, "k-", lw=1.8, label="median (smoothed)")
ax.fill_between(np.arange(T), los, his, color="tab:red", alpha=.18, label="95% boot CI")
for a, b in win_med_s:
    ax.axvspan(a, b, color="tab:green", alpha=.12)
ax.axhline(0, color="k", lw=.6)
ax.set_ylim(-.05, .05); ax.set_xlim(0, 300)
ax.set_xlabel("t (tokens)"); ax.set_ylabel(r"$\lambda_d(t)$")
ax.set_title(f"per-direction $\\lambda_d(t)$, r=0.1 L=27; green: CI>0 {win_med_s[0][0]}-{win_med_s[-1][1]}")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs[0, 1]
cols = {0.001: "tab:blue", 0.01: "tab:green", 0.1: "tab:red", 0.3: "tab:purple"}
for r, (m, l, h) in r_lam.items():
    ax.plot(np.arange(T), m, color=cols[r], lw=1.3, label=f"r={r:g}")
    ax.fill_between(np.arange(T), l, h, color=cols[r], alpha=.15)
ax.axhline(0, color="k", lw=.6); ax.set_ylim(-.02, .02); ax.set_xlim(0, 300)
ax.set_xlabel("t"); ax.set_ylabel(r"median $\lambda(t)$ [95% CI]")
ax.set_title("radii collapse: $\\lambda$ independent of amplitude (linear regime)")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs[1, 0]
ax.plot(np.arange(T), lam7, color="tab:blue", lw=1.4, label="layer 7")
ax.plot(np.arange(T), lam14, color="tab:green", lw=1.4, label="layer 14")
ax.plot(np.arange(T), meds, color="tab:red", lw=1.6, label="layer 27")
ax.axhline(0, color="k", lw=.6); ax.set_xlim(0, 300)
ax.set_xlabel("t"); ax.set_ylabel(r"median $\lambda(t)$")
ax.set_title("depth structure (r=0.1): L27 positive ~[40,100], L7 negative")
ax.legend(fontsize=8); ax.grid(alpha=.3)

ax = axs[1, 1]
ax.plot(np.arange(T), lam_of_mean, color="tab:orange", lw=1.6, label=r"$\lambda(\bar a)$: $\lambda$ of mean")
ax.fill_between(np.arange(T), lom_lo, lom_hi, color="tab:orange", alpha=.15)
ax.plot(np.arange(T), lam.mean(axis=0), color="tab:blue", lw=1.6, label=r"$\overline{\lambda_d}$: mean of $\lambda$")
ax.fill_between(np.arange(T), mol_lo, mol_hi, color="tab:blue", alpha=.15)
ax.plot(np.arange(T), meds, "k--", lw=1.3, label=r"median of $\lambda_d$")
ax.axhline(0, color="k", lw=.6); ax.set_xlim(0, 300); ax.set_ylim(-.03, .04)
ax.set_xlabel("t"); ax.set_ylabel(r"$\lambda(t)$")
ax.set_title("Jensen: log-of-mean biased high at spike times; median robust")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.suptitle(r"Finite-time rate $\lambda(t)=\ln(a(t)/a(1))/t$ — w100, teacher-forced, 20 directions")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_lambda.png"), dpi=160); plt.close(fig)

# ---- figure 2: envelope fits + decay example ----
from fitlib import fit_exp, fit_expfloor, fit_power
tpk = 77
t_dec = np.arange(tpk + 5, 300)
ydec = np.median(am[:, t_dec], axis=0)
f_exp = fit_exp(t_dec, np.log(ydec))
f_fl = fit_expfloor(t_dec, ydec)
f_pow = fit_power(t_dec, np.log(ydec))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axs[0]
ax.plot(np.arange(T), med, color="k", lw=1.2, label="median $\\lambda(t)$")
for k, style in [("two-phase", "-"), ("Aexp(-t/tau)", ":"), ("C/t-lam2", "-."), ("A-Blogt", "--")]:
    v = c_models[k]
    if k == "two-phase":
        yh = np.where(ts <= v["t*"], v["beta"][0] * (1 - 1.0 / ts), v["beta"][2] + v["beta"][1] / ts)
    elif k == "Aexp(-t/tau)": yh = v["beta"][0] + v["beta"][1] * np.exp(-ts / v["tau"])
    elif k == "C/t-lam2": yh = v["beta"][0] + v["beta"][1] / ts
    else: yh = v["beta"][0] + v["beta"][1] * np.log(ts)
    ax.plot(ts, yh, style, lw=1.4, label=f"{k}  BIC={v['bic']:.0f}  R$^2$={v['r2']:.2f}")
ax.axhline(0, color="k", lw=.5)
ax.set_xlabel("t"); ax.set_ylabel(r"median $\lambda(t)$")
ax.set_title("envelope fits: two-phase (rise$\\to$decay) wins, log-t form fails")
ax.legend(fontsize=8); ax.grid(alpha=.3)
ax = axs[1]
ax.semilogy(t_dec, ydec, "k.", ms=3, label="median decay segment")
yh = np.exp(f_exp["params"]["a0"] + f_exp["params"]["lam"] * t_dec)
ax.semilogy(t_dec, yh, "-", lw=1.4, label=f"exp: $\\tau$={-1/f_exp['params']['lam']:.0f}  R$^2$={f_exp['r2']:.3f}")
A, tau, C = f_fl["params"]["A"], f_fl["params"]["tau"], f_fl["params"]["C"]
ax.semilogy(t_dec, A * np.exp(-(t_dec - t_dec[0]) / tau) + C, "--", lw=1.4,
            label=f"exp+floor: $\\tau$={tau:.0f}, floor={C:.3g}")
yh = np.exp(f_pow["params"]["a0"] + f_pow["params"]["b"] * np.log(t_dec))
ax.semilogy(t_dec, yh, ":", lw=1.6, label=f"power: $b$={f_pow['params']['b']:.2f}")
ax.set_xlabel("t"); ax.set_ylabel(r"$a(t)$ (median, r=0.1 L=27)")
ax.set_title("decay segment: exp+floor beats pure exp ($\\Delta$BIC<0)")
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_lambda_env.png"), dpi=160); plt.close(fig)

np.savez(os.path.join(OUT, "t2_results.npz"), med=med, lo=lo, hi=hi, meds=meds, los=los, his=his,
         lam_of_mean=lam_of_mean, lom_lo=lom_lo, lom_hi=lom_hi,
         mean_of_lam=lam.mean(axis=0), mol_lo=mol_lo, mol_hi=mol_hi, lam=lam,
         f_exp_decay={k: f_exp[k] for k in ("r2", "bic", "params")},
         f_fl_decay={k: f_fl[k] for k in ("r2", "bic", "params", "se")},
         allow_pickle=True)
with open(os.path.join(OUT, "t2_summary.json"), "w") as f:
    json.dump(dict(positive_window_median_smooth=[[int(a), int(b)] for a, b in win_med_s],
                   positive_window_mean_smooth=[[int(a), int(b)] for a, b in win_mean_s],
                   frac_dirs_pos={int(t + 3): float(frac_pos[t]) for t in [27, 47, 67, 87, 107]},
                   envelope={k: v for k, v in c_models.items()},
                   jensen_gap={int(t): float(gap[t]) for t in sel_t},
                   collapse_spread=float(np.median(spread)), collapse_ciw=float(ciw),
                   decay_median=dict(exp_r2=f_exp["r2"], exp_lam=f_exp["params"]["lam"],
                                     exp_tau=float(-1 / f_exp["params"]["lam"]),
                                     floor_tau=f_fl["params"]["tau"], floor_C=f_fl["params"]["C"],
                                     floor_r2=f_fl["r2"], pow_b=f_pow["params"]["b"], pow_r2=f_pow["r2"])), f, indent=1)
print("saved fig_lambda.png, fig_lambda_env.png, t2_*")
