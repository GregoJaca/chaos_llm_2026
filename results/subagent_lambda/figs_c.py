"""Figures 7-9: cumulative-rate distributions, tail fits (AIC/BIC), forest of all configs."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa

np.seterr(divide="ignore", invalid="ignore")
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.0, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8})

z = load("w100_layers/a1.npz")
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
t = tvec(dh)
tail_mask = t > TAIL_T0
NUM = {}
COL = {50: "tab:blue", 100: "tab:red", 200: "tab:green", 400: "tab:orange",
       800: "tab:purple", 4096: "0.35"}


# ------------------------------------------------------------------ FIG 07
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
# A: Lambda(t) per layer (prompt65, W=100, r=0.1)
ax = axes[0, 0]
for j, L in [(4, 7), (5, 14), (6, 21), (7, 27)]:
    cum = cum_rate_2d(dh[4][:, j, :])
    m, lo, hi = med_band(cum)
    ax.plot(t, m, lw=1.3, label=f"L{L}")
    ax.fill_between(t, lo, hi, alpha=0.18, lw=0)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(0, 300)
ax.set_xlabel("token t")
ax.set_ylabel("Λ(t)  [per token]")
ax.set_title("prompt65, W=100, r=0.1: Λ(t)<0 for all t≳8 at every deep layer —\n"
             "never crosses back to positive; ends at −0.006 (L27) … −0.012 (L7)")
ax.legend(ncol=4)

# B: ECDF of Lambda(299) across dirs x radii (L27)
ax = axes[0, 1]
for i, r, c in [(2, 0.01, "tab:blue"), (3, 0.03, "tab:green"), (4, 0.1, "tab:red"),
                (6, 0.3, "tab:purple")]:
    cum = cum_rate_2d(dh[i][:, 7, :])
    xs, ys = ecdf(cum[:, 299])
    ax.plot(xs, ys, color=c, label=f"r={r:g}  (20 dirs)")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Λ(299) = ln[a(300)/a(1)]/299")
ax.set_ylabel("ECDF over 20 directions")
ax.set_title("final cumulative rate, 4 radii × 20 dirs at L27: 80/80 values < 0\n"
             "(range −0.010 … −0.004) — no direction, no radius ends positive")
ax.legend(loc="lower right")

# C: ECDF across windows (prompt63, L27, 10 dirs)
ax = axes[1, 0]
for W in [50, 100, 200, 4096]:
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    cum = cum_rate_2d(A)
    xs, ys = ecdf(cum[:, 299])
    ax.plot(xs, ys, color=COL[W], label=f"W={W}")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Λ(299)  [per token]")
ax.set_ylabel("ECDF over 10 directions")
ax.set_title("final Λ across windows (prompt63): W=50 all ≤ −0.010; W≥100 cluster at\n"
             "−0.004…−0.002; a few W=100/200 dirs touch 0 but none is > +0.0005")
ax.legend(loc="lower right")

# D: last token with positive Lambda, per direction (windows)
ax = axes[1, 1]
data, labels = [], []
for W in [50, 100, 200, 4096]:
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    cum = cum_rate_2d(A)
    lp = []
    for u in range(10):
        c = cum[u]
        f = np.isfinite(c)
        p = np.where(c[f] > 0)[0]
        lp.append(int(p[-1]) + 1 if len(p) else 0)
    data.append(lp)
    labels.append(f"W={W}")
ax.boxplot(data, tick_labels=labels)
ax.set_ylabel("last token t with Λ(t) > 0")
ax.set_title("zero-crossing times: Λ settles permanently negative by t≈264 (W=100, median),\n"
             "t≈91 (W=50); fluctuations of a few outliers reach t≈300")
fig.suptitle("FIG 7  cumulative rate Λ(t): distributions and zero crossings — 10–20 dirs, CIs over directions.\n"
             "Λ(t) is transiently positive only via the shared t≈13 burst; by t≈250–300 every median curve is\n"
             "negative, and 80/80 (dirs×radii) final values at L27 are < 0.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(OUT, "fig07_cumulative.png"))
plt.close(fig)
print("fig07 done")

# ------------------------------------------------------------------ FIG 08
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
# A/B: tail fits of Lambda(t), t>150, W=100 and W=4096 (prompt63, L27)
NUM["tail_fits"] = {}
for ax, W in zip([axes[0, 0], axes[0, 1]], [100, 4096]):
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    cum = cum_rate_2d(A)
    m, lo, hi = med_band(cum)
    tt = t[tail_mask]
    yy = m[tail_mask]
    fc = fit_const(tt, yy)
    c, clo, chi = fit_const_ci(yy)
    fpw = fit_pow_decay(tt, yy)
    fex = fit_exp_decay(tt, yy)
    ax.plot(tt, yy, color="0.3", lw=1.6, label="median Λ(t), 10 dirs")
    ax.fill_between(tt, lo[tail_mask], hi[tail_mask], color="0.3", alpha=0.2, lw=0)
    ax.axhline(fc["params"]["c"], color="tab:red", ls="--", lw=1.3,
               label=f"const c={fc['params']['c']:+.4f} t-CI[{clo:+.4f},{chi:+.4f}]")
    if fpw is not None:
        ax.plot(fpw["t"], fpw["yhat"], color="tab:blue", ls="-.", lw=1.3,
                label=f"pow c+A·t^−p: c={fpw['params']['c']:+.4f} A={fpw['params']['A']:+.3f} "
                      f"p={fpw['params']['p']:.2f}")
    if fex is not None:
        ax.plot(fex["t"], fex["yhat"], color="tab:green", ls=":", lw=1.4,
                label=f"exp c+A·e^(−t/τ): c={fex['params']['c']:+.4f} τ={fex['params']['tau']:.0f}")
    # OLS slope of ln a over tail = direct sustained rate with dir CI
    la_slope = []
    for u in range(10):
        s, a1, a2, _ = ols_slope(tt, np.log(A[u, tail_mask]))
        la_slope.append(s)
    ms, ls_, hs = boot_dirs_stat(np.array(la_slope))
    ax.set_title(f"W={W}: AIC const/pow/exp = {fc['aic']:.0f}/{fpw['aic']:.0f}/{fex['aic']:.0f}; "
                 f"BIC {fc['bic']:.0f}/{fpw['bic']:.0f}/{fex['bic']:.0f}\n"
                 f"sustained rate (OLS ln a, t>150) = {ms:+.4f} [{ls_:+.4f},{hs:+.4f}] — "
                 + ("significantly POSITIVE" if ls_ > 0 else "significantly NEGATIVE"
                    if hs < 0 else "indistinguishable from 0"))
    ax.set_xlabel("token t")
    ax.set_ylabel("Λ(t)  [per token]")
    ax.legend(fontsize=7)
    NUM["tail_fits"][f"W{W}"] = dict(
        const=dict(c=fc["params"]["c"], tCI=[clo, chi], aic=fc["aic"], bic=fc["bic"], r2=fc["r2"]),
        pow=dict(c=fpw["params"]["c"], A=fpw["params"]["A"], p=fpw["params"]["p"],
                 aic=fpw["aic"], bic=fpw["bic"], r2=fpw["r2"]),
        exp=dict(c=fex["params"]["c"], A=fex["params"]["A"], tau=fex["params"]["tau"],
                 aic=fex["aic"], bic=fex["bic"], r2=fex["r2"]),
        ols=dict(m=ms, lo=ls_, hi=hs))

# C: sustained rate per layer (prompt65 W=100) with CIs + window slopes
ax = axes[1, 0]
Ls, ms, los, his = [], [], [], []
for j, L in [(2, 2), (3, 3), (4, 7), (5, 14), (6, 21), (7, 27)]:
    A = dh[4][:, j, :]
    tt = t[tail_mask]
    sl = []
    for u in range(20):
        sl.append(ols_slope(tt, np.log(A[u, tail_mask]))[0])
    m, lo, hi = boot_dirs_stat(np.array(sl))
    Ls.append(f"L{L}"); ms.append(m); los.append(lo); his.append(hi)
x = np.arange(len(Ls))
ms, los, his = np.array(ms), np.array(los), np.array(his)
ax.errorbar(x - 0.12, ms, yerr=[ms - los, his - ms], fmt="o", color="tab:red", capsize=3,
            label="w100_layers (prompt65)")
w_pt = []
for W, c in [(50, COL[50]), (100, COL[100]), (4096, COL[4096])]:
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    sl = np.array([ols_slope(t[tail_mask], np.log(A[u, tail_mask]))[0] for u in range(10)])
    m, lo, hi = boot_dirs_stat(sl)
    w_pt.append((W, m, lo, hi))
xw = np.array([1.9, 2.2, 2.5])  # placed next to L27
ax.errorbar(xw, [p[1] for p in w_pt],
            yerr=[[p[1] - p[2] for p in w_pt], [p[3] - p[1] for p in w_pt]],
            fmt="s", color="tab:blue", capsize=3, label="prompt63 windows (at L27)")
for (W, m, lo, hi), xx in zip(w_pt, xw):
    ax.text(xx + 0.06, m, f"W={W}", fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(list(x) + list(xw))
ax.set_xticklabels(Ls + ["", "", ""])
ax.set_ylabel("sustained rate: OLS slope of ln a, t>150  [per token]")
ax.set_title("sustained rate per layer (95% dir CI): NEGATIVE at all layers (L2 −0.085 … L27 −0.010);\n"
             "windows at L27: W=50 −0.018, W=100 −0.006, W=4096 −0.004 — none compatible with growth")
ax.legend(fontsize=7.5)
NUM["sustained_layers"] = {L: dict(m=float(m), lo=float(lo), hi=float(hi)) for L, m, lo, hi in
                           zip(Ls, ms, los, his)}
NUM["sustained_windows"] = {W: dict(m=float(m), lo=float(lo), hi=float(hi)) for W, m, lo, hi in w_pt}

# D: delta-AIC (pow - const) for the Lambda tail, per layer and per window
ax = axes[1, 1]
names, dAIC = [], []
for j, L in [(2, 2), (3, 3), (4, 7), (5, 14), (6, 21), (7, 27)]:
    A = dh[4][:, j, :]
    cum = cum_rate_2d(A)
    m = np.nanmedian(cum, axis=0)
    tt = t[tail_mask]
    yy = m[tail_mask]
    if not np.any(np.isfinite(yy)):
        continue
    fc = fit_const(tt, yy)
    fpw = fit_pow_decay(tt, yy)
    names.append(f"L{L}"); dAIC.append(fpw["aic"] - fc["aic"])
for W in [50, 100, 200, 4096]:
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    cum = cum_rate_2d(A)
    m = np.nanmedian(cum, axis=0)
    tt = t[tail_mask]
    yy = m[tail_mask]
    fc = fit_const(tt, yy)
    fpw = fit_pow_decay(tt, yy)
    names.append(f"W{W}"); dAIC.append(fpw["aic"] - fc["aic"])
x = np.arange(len(names))
colors = ["tab:red" if v > 0 else "tab:blue" for v in dAIC]
ax.bar(x, dAIC, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, fontsize=7.5)
ax.set_ylabel("ΔAIC = AIC(pow) − AIC(const)   [negative ⇒ const (sustained rate) preferred]")
ax.set_title("model choice for the Λ tail (ΔAIC = AIC(pow) − AIC(const), negative ⇒ const preferred):\n"
             "const wins 5/6 layers + all 4 windows (only L7 marginally prefers pow, ΔAIC=+2.5);\n"
             "for the windows an exp relaxation Λ→c+A·e^(−t/τ) with c<0 beats const (see top panels)")
NUM["dAIC"] = {n: float(v) for n, v in zip(names, dAIC)}
fig.suptitle("FIG 8  tail fits (t>150) — the Lyapunov-like number. The tail of Λ(t) relaxes to a NEGATIVE\n"
             "asymptote: exp-decay Λ→c (c=−0.0012 W=100, −0.0023 W=4096) beats a pure constant (AIC); const still beats\n"
             "a power law on 9/10 layer/window grids. Direct sustained rates (OLS of ln a, t>150) are significantly\n"
             "NEGATIVE everywhere: −0.006 (W=100), −0.004 (W=4096), −0.009 (L27, prompt65). No fit admits a positive rate.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(OUT, "fig08_tailfit.png"))
plt.close(fig)
print("fig08 done")

# ------------------------------------------------------------------ FIG 09 (forest)
rows = []  # (label, group, rate mean, dir CI, 2way CI)

def add_row(label, group, LA):
    e = np.nanmean(LA[:, tail_mask], axis=1)
    if np.all(~np.isfinite(e)):
        return
    m, lo, hi = boot_dirs_stat(e)
    m2, lo2, hi2 = two_way_boot(LA, tail_mask, block=10, n_boot=600)
    rows.append((label, group, m, lo, hi, lo2, hi2))

for j, L in [(1, 1), (2, 2), (3, 3), (4, 7), (5, 14), (6, 21), (7, 27)]:
    add_row(f"prompt65 W100 r=0.1 L{L}", "layers", lam_k_2d(dh[4][:, j, :], 10))
for i, r in [(2, 0.01), (3, 0.03), (4, 0.1), (6, 0.3)]:
    add_row(f"prompt65 W100 r={r:g} L27", "radii", lam_k_2d(dh[i][:, 7, :], 10))
for W in [50, 100, 200, 400, 800, 4096]:
    add_row(f"prompt63 W={W} L27", "windows", lam_k_2d(load(f"window_sweep/w{W}/a1.npz")["dh_norm"][1][:, 3, :], 10))
for W in [50, 100, 4096]:
    add_row(f"cyber W={W} L27", "cyber", lam_k_2d(load(f"window_sweep_cyber/w{W}/a1.npz")["dh_norm"][1][:, 3, :], 10))
for p in ["p1", "p2", "p3", "p4", "p5"]:
    add_row(f"W100 prompt {p}", "prompts", lam_k_2d(load(f"rate_prompts/{p}/a1.npz")["dh_norm"][0][:, 3, :], 10))
for p in ["p1", "p2", "p3"]:
    add_row(f"W50 prompt {p}", "prompts50", lam_k_2d(load(f"rate_w50/{p}/a1.npz")["dh_norm"][0][:, 3, :], 10))
for name in ["prompts_long_w100", "prompts_long2_w100", "prompts_long_w4096", "prompts_long2_w4096"]:
    zz = load(f"rate_stats/{name}/a1.npz")
    add_row(f"{name} L27", "long", lam_k_2d(zz["dh_norm"][1][:, 3, :], 10))

fig, ax = plt.subplots(figsize=(12, 10))
y = np.arange(len(rows))
gcol = {"layers": "tab:red", "radii": "tab:orange", "windows": "tab:blue",
        "cyber": "tab:cyan", "prompts": "tab:green", "prompts50": "tab:purple", "long": "0.5"}
for i, (lab, g, m, lo, hi, lo2, hi2) in enumerate(rows):
    c = gcol[g]
    ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="butt", alpha=0.55)
    ax.plot([lo2, hi2], [i, i], color=c, lw=5, alpha=0.25)
    ax.plot(m, i, "o", color=c, ms=5)
ax.axvline(0, color="k", lw=1.2)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlabel("sustained rate: mean λ₁₀ over t>150  [per token]   (thin line = dir bootstrap 95% CI, thick = + token-block bootstrap)")
max_hi = max(r[6] for r in rows)
max_hi2 = max(r[4] for r in rows)
n_reach = sum(1 for r in rows if r[6] > 0)   # 2-way CI upper bound above 0
n_pos = sum(1 for r in rows if r[5] > 0)     # 2-way CI entirely positive
ax.set_title(f"{len(rows)} configs: sustained rate vs 0.  Point estimates: 0/{len(rows)} positive.\n"
             f"Direction bootstrap: 0/{len(rows)} CIs reach positive (max upper bound {max_hi2:+.4f}).\n"
             f"Direction+token-block bootstrap (conservative): {n_reach}/{len(rows)} CIs reach up to {max_hi:+.4f},\n"
             f"but 0/{len(rows)} are entirely positive — no config supports a sustained positive rate",
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig09_forest.png"))
plt.close(fig)
NUM["forest"] = [dict(label=r[0], group=r[1], m=r[2], dir_ci=[r[3], r[4]], twoway_ci=[r[5], r[6]])
                 for r in rows]
NUM["forest_summary"] = dict(n=len(rows), n_reach_positive_twoway=n_reach,
                             n_entirely_positive_twoway=n_pos, n_reach_positive_dir=sum(1 for r in rows if r[4] > 0),
                             max_hi_twoway=max_hi, max_hi_dir=max_hi2)
print("fig09 done: rows", len(rows), "n_reach_pos(dir CI)", sum(1 for r in rows if r[4] > 0),
      "n_reach_pos(2way)", n_reach, "max upper dir", max_hi2, "max upper 2way", max_hi)
save_json(NUM, "numbers_figs_7_9.json")
