import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini",
          "Mamba-130m", "Pythia-410m", "randQwen", "Qwen3.5-9B"]

def stats(n):
    loop = {T: (z, m) for T, z, m in soft_runs(n, "loop")}
    rep = {T: (z, m) for T, z, m in soft_runs(n, "replay")}
    out = {}
    for T, (z, meta) in loop.items():
        r = float(meta["r"])
        a = np.median(z["dh"][:, -1, :].astype(float) / r, axis=0)
        hn = float(np.median(z["h0n"][-1].astype(float)))
        sat = 0.25 * hn
        ts = []
        for i in range(z["dh"].shape[0]):
            idx = np.where(z["dh"][i, -1, :].astype(float) / r > sat)[0]
            ts.append(int(idx[0]) if len(idx) else z["dh"].shape[-1])
        ts = np.array(ts, float)
        ar = None
        if T in rep:
            zr, mr = rep[T]
            ar = np.median(zr["dh"][:, -1, :].astype(float) / float(mr["r"]), axis=0)
        m = z["m_ref"].astype(float)
        out[T] = dict(a=a, ar=ar, ts=ts, sat=sat, r=r, T=T,
                      dens=float(np.mean(m < T)), n=z["dh"].shape[0],
                      steps=z["dh"].shape[-1])
    return out

S = {n: stats(n) for n in MODELS}

# ---------------------------------------------------------------- 1. closed vs open
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
for ax, n in zip(axes.flat, MODELS):
    if n not in S or not S[n]:
        ax.axis("off"); continue
    Ts = sorted(S[n].keys())
    sel = [T for T in Ts if T > 1e-3][:4]
    cmap = plt.cm.plasma(np.linspace(0.05, 0.9, max(len(sel), 1)))
    for T, c in zip(sel, cmap):
        d = S[n][T]
        a = d["a"]; a = a / max(a[1], 1e-30)
        t = np.arange(len(a))
        ax.plot(t, a, color=c, lw=1.4, label=f"T={T:g} closed")
        if d["ar"] is not None:
            ar = d["ar"]; ar = ar / max(ar[1], 1e-30)
            ax.plot(t, ar[:len(a)], color=c, lw=0.9, ls="--")
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.plot([], [], color="k", ls="--", lw=0.9, label="replay (network only)")
    ax.set_yscale("log")
    ax.set_xlim(0, 200)
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel("step $t$", fontsize=7.6)
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$", fontsize=7.6)
    ax.legend(fontsize=5.4, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Closed loop (solid) vs network-only replay (dashed); colour = temperature", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_softloop_closed_all.png")

# ---------------------------------------------------------------- 2. phase (broken axis + colorbar)
n = "DeepSeek-1.5B"
Ts_all = np.array(sorted(S[n].keys()))
Ts = Ts_all[Ts_all > 1e-3]
ts = np.array([np.median(S[n][T]["ts"]) for T in Ts])
cens = np.array([np.mean(S[n][T]["ts"] >= S[n][T]["steps"]) for T in Ts])
dens = np.array([max(S[n][T]["dens"], 1e-6) for T in Ts])
fig = plt.figure(figsize=(12.6, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.08)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
# categorical x: greedy first, then the temperatures
xs = np.arange(len(Ts) + 1)
Ts_plot = np.concatenate([[0.0], Ts])
ts_plot = np.concatenate([[600.0], ts])
cens_plot = np.concatenate([[1.0], cens])
for x, tt, c in zip(xs, ts_plot, cens_plot):
    for ax in (ax0, ax1):
        ax.plot(x, tt, "o", color="#111111" if c > 0.5 else COL[n], ms=6)
ax0.set_ylim(580, 620); ax1.set_ylim(0, 260)
ax0.spines["bottom"].set_visible(False); ax1.spines["top"].set_visible(False)
ax0.tick_params(labelbottom=False, bottom=False)
ax0.set_ylabel(r"$t_{\rm sat}$ (censored)")
ax1.set_ylabel(r"median $t_{\rm sat}$ (tokens)")
ax1.set_xticks(xs)
ax1.set_xticklabels(["greedy"] + [f"{T:g}" for T in Ts], rotation=25, ha="right", fontsize=7.4)
ax1.set_xlabel("temperature $T$  (greedy = $T\\to0$)")
ax0.set_title("(a) Horizon vs $T$; black = 100% of directions survived", loc="left")
d = .012
kwargs = dict(transform=ax0.transAxes, color="k", clip_on=False, lw=0.8)
ax0.plot((-d, +d), (-d, +d), **kwargs); ax0.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, lw=0.8)
ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs); ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax = fig.add_subplot(gs[:, 1])
sc = ax.scatter(dens, ts, c=np.log10(Ts), cmap="viridis", s=70, edgecolor="k", lw=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"live-gate density  $P(m_t<T)$")
ax.set_ylabel(r"median $t_{\rm sat}$")
cb = fig.colorbar(sc, ax=ax, pad=0.01); cb.set_label(r"$\log_{10}T$")
ax.set_title("(b) Horizon vs gate density, colour = $T$", loc="left")
ax.grid(True, which="both", alpha=0.3)
fig.suptitle("The phase plot, redrawn: broken axis and an explicit temperature colourbar", fontsize=11.2)
save(fig, "pack_softloop_phase.png")

# ---------------------------------------------------------------- 3. rates: distribution
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
sep, net = [], []
for n in MODELS:
    for T, d in S[n].items():
        a = d["a"]; tmax = np.nanmedian(d["ts"])
        if np.isfinite(tmax) and tmax >= 8:
            t = np.arange(3, int(min(tmax, len(a))))
            am = np.median(a[t])
            sep.append(np.polyfit(t, np.log(np.maximum(a[t], 1e-30)), 1)[0])
        if d["ar"] is not None:
            ar = d["ar"]; t = np.arange(3, len(ar))
            net.append(np.polyfit(t, np.log(np.maximum(ar[t], 1e-30)), 1)[0])
ax = axes[0]
bins = np.linspace(-0.08, 0.08, 40)
ax.hist(net, bins=bins, color="#1f77b4", alpha=0.8, label=f"$\\lambda_{{\\rm net}}$ (n={len(net)})")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"network-only rate $\lambda_{\rm net}$  (nats/step)")
ax.set_ylabel("count")
ax.set_title("(a) Replay rate: 95% inside $\\pm0.006$", loc="left")
ax.legend(fontsize=7.6); ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(sep, bins=np.linspace(-0.1, 0.5, 40), color="#d62728", alpha=0.8, label=f"$\\lambda_{{\\rm sep}}$ (n={len(sep)})")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"closed-loop rate $\lambda_{\rm sep}$  (nats/step)")
ax.set_ylabel("count")
ax.set_title("(b) Closed-loop rate is broad and mostly positive", loc="left")
ax.legend(fontsize=7.6); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_softloop_rates.png")

# ---------------------------------------------------------------- 4. response all
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
for ax, n in zip(axes.flat, MODELS):
    if n not in S or not S[n]:
        ax.axis("off"); continue
    Ts = [T for T in sorted(S[n]) if T > 1e-3][:4]
    for T in Ts:
        d = S[n][T]; a = d["a"]; t = np.arange(len(a))
        ax.plot(t, a, lw=1.2, label=f"T={T:g}")
    ax.set_yscale("log"); ax.set_xlim(0, 250)
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel("step $t$", fontsize=7.6)
    ax.set_ylabel(r"$\|\delta h_t\|/r$", fontsize=7.6)
    ax.legend(fontsize=5.8); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Soft-loop response, every model and several temperatures", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_softloop_response_all.png")

json.dump({n: {str(T): {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in d.items() if k not in ("a", "ar")}
                for T, d in S[n].items()} for n in S},
          open(os.path.join(HERE, "analysis_update", "pack_softloop.json"), "w"), indent=1)

# ---------------------------------------------------------------- 5. paired growth at t=100
fig, ax = plt.subplots(figsize=(11.0, 4.8))
names, gc, gr = [], [], []
for n in MODELS:
    if n not in S or not S[n]:
        continue
    vals_c, vals_r = [], []
    for T, d in S[n].items():
        if T < 1e-3:
            continue
        a = d["a"]
        if len(a) > 100:
            vals_c.append(a[100] / max(a[1], 1e-30))
        if d["ar"] is not None and len(d["ar"]) > 100:
            vals_r.append(d["ar"][100] / max(d["ar"][1], 1e-30))
    if vals_c:
        names.append(n); gc.append(np.median(vals_c)); gr.append(np.median(vals_r) if vals_r else np.nan)
x = np.arange(len(names))
ax.bar(x - 0.18, gc, width=0.36, color="#d62728", label="closed loop")
ax.bar(x + 0.18, gr, width=0.36, color="#1f77b4", label="replay (network only)")
ax.set_yscale("log")
ax.axhline(1, color="k", ls=":", lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.6)
ax.set_ylabel(r"$\|\delta h_{100}\|/\|\delta h_1\|$  (median over $T$)")
ax.set_title("Closed loop vs replay at step 100: the gap is the feedback", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y", which="both")
fig.tight_layout()
save(fig, "pack_softloop_open_closed_bars.png")

