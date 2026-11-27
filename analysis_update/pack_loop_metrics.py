import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from looputil import gather
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini",
          "Mamba-130m", "Pythia-410m", "randQwen", "Qwen3.5-9B"]
SAMPLE = {"DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b",
          "Phi-4-mini": "phi4mini", "Mamba-130m": "mamba130", "Pythia-410m": "pythia410"}

def get(n, T=1.0, mode="loop"):
    for TT, z, m in soft_runs(n, mode):
        if abs(TT - T) < 1e-9:
            return z
    return None

def arr(z):
    return z["dh"][:, -1, :].astype(float) if z is not None else None

def metrics(a):
    """a: (dirs, T). Returns dict of per-direction metrics."""
    out = {}
    aa = np.maximum(a, 1e-30)
    la = np.log(aa / aa[:, [1]])
    out["M"] = np.mean(la[:, 1:], axis=1)
    out["lam_end"] = la[:, -1] / (a.shape[1] - 1)
    out["A_max"] = np.max(la, axis=1)
    t10 = []
    for i in range(a.shape[0]):
        idx = np.where(la[i] > np.log(10))[0]
        t10.append(idx[0] if len(idx) else np.nan)
    out["t10"] = np.array(t10, float)
    d = np.diff(la, axis=1)
    out["max_jump"] = np.max(d, axis=1)
    out["jump_frac"] = np.mean(d > np.log(2), axis=1)
    return out

# ---------------------------------------------------------------- bars: M closed/replay/CRN/ind
fig, ax = plt.subplots(figsize=(13.0, 4.9))
labels, series = [], []
for n in MODELS:
    z = get(n, 1.0, "loop"); zr = get(n, 1.0, "replay")
    if z is None:
        continue
    labels.append(n)
    vals = [np.median(metrics(arr(z))["M"])]
    vals.append(np.median(metrics(arr(zr))["M"]) if zr is not None else np.nan)
    base = os.path.join(HERE, "results/softloop_sample", SAMPLE.get(n, ""))
    for mode in ["crn", "ind"]:
        p = os.path.join(base, f"{mode}_T1.0/softloop.npz")
        vals.append(np.median(metrics(dict(np.load(p, allow_pickle=True))["dh"][:, -1, :].astype(float))["M"])
                    if os.path.exists(p) else np.nan)
    series.append(vals)
series = np.array(series)
x = np.arange(len(labels))
for j, (lab, c) in enumerate([("closed", "#d62728"), ("replay", "#1f77b4"),
                              ("CRN", "#2ca02c"), ("independent", "#9467bd")]):
    ax.bar(x + (j - 1.5) * 0.2, series[:, j], width=0.2, color=c, label=lab)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.4)
ax.set_ylabel(r"$M=\langle\log(a_t/a_1)\rangle_t$  (mean log amplification)")
ax.set_title("Mean log amplification at $T=1$: feedback rule comparison", loc="left")
ax.legend(fontsize=7.6, ncol=4); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_loop_metric_M.png")

# ---------------------------------------------------------------- T dependence of M
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.0))
for ax, n in zip(axes.flat, MODELS):
    Ts, mc, mr = [], [], []
    for T, z, m in soft_runs(n):
        if T < 1e-3:
            continue
        mm = metrics(arr(z))["M"]
        Ts.append(T); mc.append(np.median(mm))
    for T, z, m in soft_runs(n, "replay"):
        if T < 1e-3:
            continue
        mr.append((T, np.median(metrics(arr(z))["M"])))
    if not Ts:
        ax.axis("off"); continue
    ax.plot(Ts, mc, "o-", color="#d62728", label="closed")
    if mr:
        tt, vv = zip(*sorted(mr))
        ax.plot(tt, vv, "s--", color="#1f77b4", label="replay")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_title(n, fontsize=8.4)
    ax.set_xlabel("$T$", fontsize=7.4); ax.set_ylabel(r"$M$", fontsize=7.4)
    ax.legend(fontsize=6.0); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Mean log amplification vs temperature, closed loop vs replay", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_metric_T.png")

# ---------------------------------------------------------------- FSLE
# S is the factor by which the response has grown relative to t=1
# (S = 2 means "twice its initial value").  t_S is the first step at which the
# response reaches S; the FSLE is log(S)/t_S, i.e. an average growth rate over
# the interval [1, t_S].  Per-direction: for each threshold we use the median
# t_S over the directions that reach it and mark hollow the thresholds reached by
# fewer than 30% of the directions (the honest way to show incomplete data).
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.0))
for ax, n in zip(axes.flat, MODELS):
    for mode, c, lab in [("loop", "#d62728", "closed"), ("replay", "#1f77b4", "replay")]:
        ac, ar, _u = gather(n, "T1.0")
        a = ac if mode == "loop" else ar
        if a is None:
            continue
        S = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        med, frac = [], []
        for sv in S:
            ts = []
            for d in range(a.shape[0]):
                idx = np.where(a[d] > sv)[0]
                if len(idx):
                    ts.append(idx[0])
            med.append(np.median(ts) if ts else np.nan)
            frac.append(len(ts) / a.shape[0])
        med = np.array(med, float); frac = np.array(frac)
        ok = np.isfinite(med) & (med > 0)
        rr = np.log(S[ok]) / med[ok]
        ax.plot(np.log(S[ok]), rr, "-", color=c, lw=1.2, label=lab)
        solid = ok & (frac >= 0.3)
        hollow = ok & (frac < 0.3)
        ax.plot(np.log(S[solid]), rr[frac[ok] >= 0.3], "o", color=c, ms=4)
        ax.plot(np.log(S[hollow]), rr[frac[ok] < 0.3], "o", mfc="none", mec=c, ms=5)
    ax.set_xlabel(r"$\log S$  ($S$ = response / value at $t=1$)", fontsize=7.2)
    ax.set_ylabel(r"$\log S / t_S$  (nats/step)", fontsize=7.2)
    ax.set_title(n, fontsize=8.4)
    ax.legend(fontsize=6.0); ax.grid(alpha=0.3)
fig.suptitle("Finite-scale growth rate: $t_S$ = first step reaching $S\\times$ the initial response (hollow = <30% of directions reach $S$)", fontsize=10.6)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "pack_loop_fsle.png")

# ---------------------------------------------------------------- staircase
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
for n in MODELS:
    z = get(n, 1.0, "loop")
    if z is None:
        continue
    a = np.median(arr(z), axis=0); a = a / a[1]
    d = np.diff(np.log(np.maximum(a, 1e-30)))
    axes[0].plot(np.arange(len(d)), d, lw=1.0, color=COL[n], label=n)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xlabel("step $t$")
axes[0].set_ylabel(r"$\Delta\log a_t$")
axes[0].set_title("(a) One-step log increments, closed loop $T=1$", loc="left")
axes[0].legend(fontsize=6.0, ncol=2); axes[0].grid(alpha=0.3)
for n in MODELS:
    z = get(n, 1.0, "loop")
    if z is None:
        continue
    a = np.median(arr(z), axis=0); a = a / a[1]
    d = np.diff(np.log(np.maximum(a, 1e-30)))
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / ac[0]
    axes[1].plot(np.arange(len(ac)), ac, lw=1.0, color=COL[n], label=n)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("lag")
axes[1].set_ylabel("autocorrelation of $\\Delta\\log a$")
axes[1].set_title("(b) Increments are not white noise", loc="left")
axes[1].legend(fontsize=6.0, ncol=2); axes[1].grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_staircase.png")

# ---------------------------------------------------------------- endpoint vs feedback
fig, ax = plt.subplots(figsize=(8.4, 5.0))
for n in MODELS:
    z = get(n, 1.0, "loop"); zr = get(n, 1.0, "replay")
    if z is None or zr is None:
        continue
    mc = metrics(arr(z)); mr = metrics(arr(zr))
    ax.scatter(np.median(mr["lam_end"]), np.median(mc["lam_end"]), color=COL[n], s=55, label=n)
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.plot([-0.02, 0.02], [-0.02, 0.02], "k:", lw=1.0)
ax.set_xlabel(r"replay endpoint rate $\lambda_{\rm net}$")
ax.set_ylabel(r"closed endpoint rate $\lambda_{\rm loop}$")
ax.set_title("Endpoint rates: the loop moves off the replay line", loc="left")
ax.legend(fontsize=6.6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_endpoint.png")
print("done")
