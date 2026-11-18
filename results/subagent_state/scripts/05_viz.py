"""05: Visualization that makes R(t,s) intuitive.

fig05_response_field.png (the headline figure, 3 stacked panels, shared t axis):
  (top)    heat map of log10 median ||dh(t,s)|| over the (s,t) plane;
           white = before injection (exactly zero). Reads as a causal "wedge".
  (middle) one representative row (s=100): 3 individual directions (thin) +
           median (thick), log scale.
  (bottom) baseline margin m(t) with near-ties marked, for context.

fig05_lag_view.png: the same field in lag coordinates tau=t-s:
  (a) heat map of log10 R(t,s)=dh(t)/dh(s) over (s,tau), plus the 4 median
      lag curves -> shows the failure of lag collapse and the censoring of
      large s (fewer lags available).
  (b) spread factor across s vs tau (how non-collapsed R is at each lag).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_state, load_baseline, log10safe, save_json, savefig

st = load_state()
base = load_baseline()
dh, steps = st["dh"], st["steps"]
m = base["margins"]
med = np.median(dh, axis=1)
T = dh.shape[-1]
K = len(steps)

# ================================================================ headline figure
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 0.9], hspace=0.35)
ax0 = fig.add_subplot(gs[0])

F = log10safe(med).copy()          # [4, 300]; entries t<s are log10(0)->floor, mask them
mask = np.zeros_like(F, dtype=bool)
for si, s in enumerate(steps):
    mask[si, :int(s)] = True
Fm = np.ma.masked_array(F, mask)
im = ax0.imshow(Fm, aspect="auto", cmap="viridis", origin="lower", vmin=-3.6, vmax=-1.0,
                extent=[0, T, -0.5, K - 0.5])
ax0.set_yticks(range(K), [f"s={int(s)}" for s in steps])
for si, s in enumerate(steps):
    ax0.plot([s, s], [si - 0.5, si + 0.5], color="w", lw=2)
ax0.set_xlabel("generated token t")
ax0.set_title(r"current-state response field: $\log_{10}$ median $\|\delta h_t\|$ "
              "(32 directions, r=1e-3, layer 27); white = before injection", fontsize=10)
cb = fig.colorbar(im, ax=ax0, pad=0.01)
cb.set_label(r"$\log_{10}\|\delta h_t\|$")

ax1 = fig.add_subplot(gs[1], sharex=ax0)
si = list(steps).index(100)
tt = np.arange(int(steps[si]), T)
x = dh[si][:, int(steps[si]):]
tot = np.mean(log10safe(x), axis=1)
d_lo, d_md, d_hi = int(np.argmin(tot)), int(np.argsort(tot)[x.shape[0] // 2]), int(np.argmax(tot))
for d, lab in [(d_hi, f"d{d_hi} (largest)"), (d_md, f"d{d_md} (median)"), (d_lo, f"d{d_lo} (smallest)")]:
    ax1.plot(tt, np.maximum(x[d], 1e-12), lw=0.7, alpha=0.8, label=lab)
ax1.plot(tt, np.median(x, axis=0), lw=2, color="k", label="median over 32 dirs")
ax1.set_yscale("log")
ax1.set_ylabel(r"$\|\delta h_t\|$")
ax1.set_title("row s=100 zoomed: 3 individual directions + aggregate", fontsize=10)
ax1.legend(fontsize=7, loc="upper right", ncol=2)
ax1.grid(alpha=0.25)

ax2 = fig.add_subplot(gs[2], sharex=ax0)
ax2.plot(np.arange(T), np.maximum(m, 1e-3), lw=0.8, color="k")
ax2.set_yscale("log")
nt = np.where(m < 0.2)[0]
ax2.plot(nt, np.maximum(m[nt], 1e-3), "v", color="#c44e52", ms=5, label="margin < 0.2")
ax2.set_xlabel("generated token t")
ax2.set_ylabel("baseline margin")
ax2.set_title("baseline top1-top2 margin (context: bumps do not align with near-ties)",
              fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)
savefig(fig, "fig05_response_field.png")

# ================================================================ lag view
fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False,
                        gridspec_kw=dict(height_ratios=[1.6, 1]))
Lmax = T - int(steps.min())      # 284
Rimg = np.full((K, Lmax), np.nan)
for si, s in enumerate(steps):
    x = med[si, int(s):]
    Rimg[si, :len(x)] = log10safe(x / max(x[0], 1e-30))
Rm = np.ma.masked_array(Rimg, np.isnan(Rimg))
im = axs[0].imshow(Rm, aspect="auto", cmap="magma", origin="lower", vmin=-2.5, vmax=0.2,
                   extent=[0, Lmax, -0.5, K - 0.5])
axs[0].set_yticks(range(K), [f"s={int(s)}" for s in steps])
axs[0].set_xlabel(r"lag $\tau=t-s$")
axs[0].set_ylabel("injection time s")
axs[0].set_title(r"$\log_{10} R(t,s)=\log_{10}[\|\delta h_t\|/\|\delta h_s\|]$ in lag coordinates"
                 "\nrows do not overlap: R is not a function of lag alone", fontsize=10)
cb = fig.colorbar(im, ax=axs[0], pad=0.01)
cb.set_label(r"$\log_{10}R$")

taus = np.arange(Lmax)
common = (taus >= 5) & (taus <= 149)
L = Rimg[:, common]
sd = np.nanstd(L, axis=0)
axs[1].plot(taus[common], 10 ** sd, lw=1.6, color="#4c72b0")
axs[1].axhline(1.0, color="k", lw=0.7, ls="--")
axs[1].axhline(float(np.mean(10 ** sd)), color="#c44e52", lw=0.9, ls=":",
               label=f"mean spread factor {np.mean(10**sd):.2f}")
axs[1].set_xlabel(r"lag $\tau=t-s$")
axs[1].set_ylabel("spread factor across s")
axs[1].set_title("disagreement between the four s curves vs lag "
                 "(1 = perfect collapse of median R)", fontsize=10)
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.25)
fig.tight_layout()
savefig(fig, "fig05_lag_view.png")

save_json(dict(mean_spread_factor=float(np.mean(10 ** sd)),
               spread_factor_at_tau_25=float(10 ** sd[20]),
               spread_factor_at_tau_100=float(10 ** sd[95]),
               spread_factor_at_tau_149=float(10 ** sd[-1])), "lag_view.json")
print("spread factor: mean %.2f, at tau=25 %.2f, tau=100 %.2f, tau=149 %.2f" %
      (np.mean(10 ** sd), 10 ** sd[20], 10 ** sd[95], 10 ** sd[-1]))
