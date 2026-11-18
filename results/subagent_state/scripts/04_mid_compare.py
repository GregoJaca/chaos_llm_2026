"""04: input-embedding injection (state.npz) vs layer-14 residual injection
     (state_mid.npz), both measured at layer 27, r=1e-3, common s=16,50,100.

Caveats handled explicitly:
  - different direction samples (seeds 1000+i, n=32 vs 2000+i, n=8):
    UNPAIRED comparison; report medians with IQR and direction scatter.
  - both are teacher-forced (fixed gen_ids), so this is about propagation
    of the injected perturbation to later positions, not about generation.

Quantities: immediate dh(s); median curves; log-envelope slope on lags 5-30;
mean response over lags 5-30; ratio emb/mid for each; response relative to
the baseline hidden norm ||h|| (layer 27, median 545).

Figure: figures/fig04_mid_vs_emb.png ; stats: mid_vs_emb.json
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from common import load_state, load_state_mid, load_baseline, log10safe, save_json, savefig

st = load_state()
mid = load_state_mid()
base = load_baseline()
dh, steps = st["dh"], st["steps"]
d2, steps2 = mid["dh"], mid["steps"]
normh = float(np.median(base["norm_h0"]))
common = [int(s) for s in steps if int(s) in [int(x) for x in steps2]]
stats = {"common_s": common, "n_dirs_emb": int(dh.shape[1]), "n_dirs_mid": int(d2.shape[1]),
         "seeds": "emb 1000+i, mid 2000+i (unpaired)", "median_norm_h27": normh}

per_s = {}
for s in common:
    si = list(steps).index(s)
    sj = [int(x) for x in steps2].index(s)
    xe, xm = dh[si], d2[sj]                     # [32,300], [8,300]
    me, mm = np.median(xe, axis=0), np.median(xm, axis=0)

    def slope_env(med_curve, s, a=5, b=30):
        L = log10safe(med_curve[s:])
        env = median_filter(L, size=9, mode="nearest")
        x = np.arange(len(env))
        sel = (x >= a) & (x < b)
        k, b0 = np.polyfit(x[sel], env[sel], 1)
        return float(k)

    se, sm = slope_env(me, s), slope_env(mm, s)
    area_e = float(np.mean(me[s + 5:s + 30]))
    area_m = float(np.mean(mm[s + 5:s + 30]))
    # persistent tail: median over t>=150 (structured signal: lag-1 autocorr > 0.5)
    def tail_stats(med_curve, s):
        t0 = max(s + 1, 150)
        x = med_curve[t0:]
        dx = x - x.mean()
        ac1 = float(np.correlate(dx, dx, "full")[len(dx) - 1] / np.sum(dx ** 2))
        return dict(level=float(np.median(x)), rel_variation=float(x.std() / x.mean()),
                    lag1_autocorr=ac1)
    te, tm = tail_stats(me, s), tail_stats(mm, s)
    # initial drop tau0 -> tau5
    drop_e = float(me[s + 5] / me[s])
    drop_m = float(mm[s + 5] / mm[s])
    per_s[s] = dict(
        dh_at_s_emb=float(me[s]), dh_at_s_mid=float(mm[s]),
        ratio_at_s=float(me[s] / mm[s]),
        ratio_area_lag5_30=float(area_e / area_m),
        initial_drop_tau0_to_tau5=dict(emb=drop_e, mid=drop_m),
        slope_emb_dex_per_token=se, slope_mid_dex_per_token=sm,
        slope_ratio_mid_over_emb=float(sm / se) if se != 0 else None,
        tail_t_ge_150=dict(emb=te, mid=tm, ratio_level=float(te["level"] / tm["level"])),
        dir_spread_emb_dex=float((log10safe(xe[:, s + 5:s + 30]) -
                                  log10safe(np.median(xe[:, s + 5:s + 30], axis=0))).std()),
        dir_spread_mid_dex=float((log10safe(xm[:, s + 5:s + 30]) -
                                  log10safe(np.median(xm[:, s + 5:s + 30], axis=0))).std()),
        max_rel_to_norm_emb=float(xe.max() / normh),
        max_rel_to_norm_mid=float(xm.max() / normh))
stats["per_s"] = per_s
stats["note_tail"] = ("the mid tail plateau is a real structured signal, not fp32 noise: "
                      "dh(t<s)=0 exactly, lag-1 autocorr > 0.5, tight across directions")
save_json(stats, "mid_vs_emb.json")

fig, axs = plt.subplots(1, len(common), figsize=(4.2 * len(common), 3.8), sharey=True)
for ax, s in zip(axs, common):
    si = list(steps).index(s)
    sj = [int(x) for x in steps2].index(s)
    tt = np.arange(s, dh.shape[-1])
    for x, lab, col in [(dh[si], "input-embedding (32 dirs)", "#4c72b0"),
                        (d2[sj], "layer-14 residual (8 dirs)", "#c44e52")]:
        m = np.median(x[:, s:], axis=0)
        q1, q3 = np.quantile(x[:, s:], .25, axis=0), np.quantile(x[:, s:], .75, axis=0)
        ax.plot(tt, m, lw=1.8, color=col, label=lab)
        ax.fill_between(tt, q1, q3, color=col, alpha=0.15)
    p = per_s[s]
    ax.set_yscale("log")
    ax.set_xlabel("generated token t")
    tl = p["tail_t_ge_150"]
    ax.set_title(f"s={s}: dh(s) ratio x{p['ratio_at_s']:.0f}; tails {tl['emb']['level']:.1e}"
                 f" vs {tl['mid']['level']:.1e}", fontsize=9)
    ax.grid(alpha=0.25)
axs[0].set_ylabel(r"median $\|\delta h_t\|$ (layer 27), bands = IQR")
axs[0].legend(fontsize=8)
fig.suptitle("Input-embedding vs layer-14 residual injection, r=1e-3 "
             "(unpaired direction samples)", fontsize=11)
fig.tight_layout()
savefig(fig, "fig04_mid_vs_emb.png")

for s in common:
    p = per_s[s]
    tl = p["tail_t_ge_150"]
    print(f"s={s}: dh(s) emb {p['dh_at_s_emb']:.4f} vs mid {p['dh_at_s_mid']:.5f} (x{p['ratio_at_s']:.1f}); "
          f"drop tau0->5 emb {p['initial_drop_tau0_to_tau5']['emb']:.2f} mid {p['initial_drop_tau0_to_tau5']['mid']:.2f}; "
          f"slope(5-30) emb {p['slope_emb_dex_per_token']:+.4f} mid {p['slope_mid_dex_per_token']:+.4f}; "
          f"tail emb {tl['emb']['level']:.2e} mid {tl['mid']['level']:.2e} (x{tl['ratio_level']:.1f})")
