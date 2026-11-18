"""03: s-dependence of the response: immediate gain, decay rate, shape.

With only 4 injection times (16, 50, 100, 150) this is descriptive, not
statistical: we quantify, per s:
  - immediate response dh(s) (= local gain through 27 layers, since ||v||=1)
  - decay slopes of the log envelope on lag windows [10,50] and [50,120]
    (robust: fitted on a rolling-median envelope; per-direction spread)
  - half-life of the envelope
  - value at fixed lags
  - bumpiness (std of the log residual after removing the envelope)

Figure: figures/fig03_s_dependence.png ; stats: s_dependence.json
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from common import load_state, load_baseline, log10safe, save_json, savefig

st = load_state()
base = load_baseline()
dh, steps = st["dh"], st["steps"]
m = base["margins"]
med = np.median(dh, axis=1)
T = dh.shape[-1]
r = st["r"]
stats = {"r": r}


def fit_slope(y_log, lag0, lag1):
    """OLS slope (dex/token) of y_log over lag in [lag0, lag1); returns (slope, r2, n)."""
    x = np.arange(len(y_log))
    sel = (x >= lag0) & (x < lag1)
    if sel.sum() < 5:
        return None
    k, b = np.polyfit(x[sel], y_log[sel], 1)
    yh = k * x[sel] + b
    ss = 1 - np.sum((y_log[sel] - yh) ** 2) / np.sum((y_log[sel] - y_log[sel].mean()) ** 2)
    return float(k), float(ss), int(sel.sum())


per_s = {}
for si, s in enumerate(steps):
    x = med[si, s:]                                   # median curve, lags 0..T-s-1
    L = log10safe(x)
    env = median_filter(L, size=15, mode="nearest")   # smooth envelope
    lam = {}
    for tag, (a, b) in [("lag10_50", (10, 50)), ("lag50_120", (50, 120))]:
        if len(L) < b - 2:
            b = len(L) - 2
        raw = fit_slope(L, a, b)
        envf = fit_slope(env, a, b)
        # per-direction slopes for spread
        sl = []
        for d in range(dh.shape[1]):
            Ld = log10safe(dh[si, d, s:])
            f = fit_slope(Ld, a, min(b, len(Ld) - 2))
            if f:
                sl.append(f[0])
        lam[tag] = dict(env_slope_dex_per_token=envf[0] if envf else None,
                        env_r2=envf[1] if envf else None,
                        raw_slope_dex_per_token=raw[0] if raw else None,
                        dir_slope_median=float(np.median(sl)),
                        dir_slope_iqr=[float(np.quantile(sl, .25)), float(np.quantile(sl, .75))])
        # convert to e-folds and half-life from envelope slope
        if envf:
            lam_e = envf[0] * np.log(10)            # per-token e-folding rate
            lam[tag]["e_folding_rate_per_token"] = float(lam_e)
            lam[tag]["e_folding_tokens"] = float(1.0 / lam_e) if lam_e < 0 else None
    # half-life: first lag where env falls below env_peak - log10(2)
    pk = env.argmax()
    target = env[pk] - np.log10(2)
    below = np.where(env[pk:] < target)[0]
    halflife = int(below[0] + pk) if len(below) else None
    # bumpiness
    resid = L - env
    # fixed-lag values
    fixed = {}
    for lag in [0, 5, 10, 20, 50, 100]:
        if lag < len(x):
            fixed[lag] = float(np.median(dh[si, :, s + lag]))
    per_s[int(s)] = dict(
        dh_at_s_median=float(np.median(dh[si, :, s])),
        dh_at_s_iqr=[float(np.quantile(dh[si, :, s], .25)), float(np.quantile(dh[si, :, s], .75))],
        immediate_gain_over_r=float(np.median(dh[si, :, s]) / r),
        margin_at_s=float(m[s]),
        slopes=lam, env_peak_lag=int(pk), env_half_life_lag=halflife,
        bumpiness_dex=float(resid.std()), fixed_lag_median_dh=fixed)

stats["per_s"] = per_s
save_json(stats, "s_dependence.json")

# ================================================================ figure
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
ss = [int(s) for s in steps]
c = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

# (a) immediate response + margin at s
ax = axs[0, 0]
vals = [per_s[s]["dh_at_s_median"] for s in ss]
err = [[max(0, vals[i] - per_s[s]["dh_at_s_iqr"][0]) for i, s in enumerate(ss)],
       [max(0, per_s[s]["dh_at_s_iqr"][1] - vals[i]) for i, s in enumerate(ss)]]
ax.bar([str(s) for s in ss], vals, yerr=err, color=c, capsize=4)
ax.set_yscale("log")
ax.set_xlabel("injection time s")
ax.set_ylabel(r"median $\|\delta h_s\|$ at injection")
ax2 = ax.twinx()
ax2.plot([str(s) for s in ss], [per_s[s]["margin_at_s"] for s in ss], "k.--", lw=1, alpha=.7)
ax2.set_yscale("log")
ax2.set_ylabel("baseline margin at s (log)")
ax.set_title(f"(a) immediate response dh(s): range {min(vals):.3f}-{max(vals):.3f} "
             f"(x{max(vals)/min(vals):.0f}); dots = margin(s)", fontsize=9)
ax.grid(alpha=0.25, axis="y")

# (b) decay slopes
ax = axs[0, 1]
w = 0.35
xs = np.arange(len(ss))
for k, (tag, lab) in enumerate([("lag10_50", "lags 10-50"), ("lag50_120", "lags 50-120")]):
    v = [per_s[s]["slopes"][tag]["env_slope_dex_per_token"] for s in ss]
    ax.bar(xs + (k - .5) * w, v, w, label=lab, color=["#4c72b0", "#dd8452"][k])
ax.set_xticks(xs)
ax.set_xticklabels([str(s) for s in ss])
ax.set_xlabel("injection time s")
ax.set_ylabel("envelope log-slope (dex / token)")
ax.axhline(0, color="k", lw=0.6)
ax.legend(fontsize=8)
ax.set_title("(b) decay slopes of the log envelope\n(negative = contraction)", fontsize=9)
ax.grid(alpha=0.25, axis="y")

# (c) fixed-lag values vs s
ax = axs[1, 0]
for lag in [0, 5, 10, 20, 50, 100]:
    y = [per_s[s]["fixed_lag_median_dh"].get(lag) for s in ss]
    ax.plot(ss, y, "o-", label=f"lag {lag}", lw=1.2, ms=4)
ax.set_yscale("log")
ax.set_xlabel("injection time s")
ax.set_ylabel(r"median $\|\delta h_{s+\tau}\|$")
ax.set_title("(c) response at fixed lags vs s\nnon-monotonic, no single trend", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# (d) half-life + bumpiness
ax = axs[1, 1]
hl = [per_s[s]["env_half_life_lag"] if per_s[s]["env_half_life_lag"] is not None else np.nan for s in ss]
ax.plot(ss, hl, "o-", color="#4c72b0", label="envelope half-life (tokens)")
ax.set_xlabel("injection time s")
ax.set_ylabel("half-life (tokens)")
ax2 = ax.twinx()
bp = [per_s[s]["bumpiness_dex"] for s in ss]
ax2.plot(ss, bp, "s--", color="#c44e52", label="bumpiness (dex)")
ax2.set_ylabel("bumpiness (std of log residual, dex)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8)
ax.set_title("(d) persistence and bumpiness vs s", fontsize=9)
ax.grid(alpha=0.25)

fig.suptitle("s-dependence of the current-state response (4 injection times only)", fontsize=11)
fig.tight_layout()
savefig(fig, "fig03_s_dependence.png")

for s in ss:
    p = per_s[s]
    print(f"s={s}: dh(s)={p['dh_at_s_median']:.4f} (gain x{p['immediate_gain_over_r']:.0f}), "
          f"margin={p['margin_at_s']:.3f}, slope(10-50)={p['slopes']['lag10_50']['env_slope_dex_per_token']:.4f} dex/tok "
          f"(R2={p['slopes']['lag10_50']['env_r2']:.2f}), slope(50-120)={p['slopes']['lag50_120']['env_slope_dex_per_token']:.4f}, "
          f"half-life={p['env_half_life_lag']}, bumpy={p['bumpiness_dex']:.3f}")
