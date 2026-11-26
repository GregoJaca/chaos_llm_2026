import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# ------------------------------------------------------------------ greedy A1 runs
A1_ORDER = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Qwen3.5-2B"]
resp = {}
for n in A1_ORDER:
    d = load_a1(n)
    radii = d["radii"].astype(float)
    ri = int(np.argmin(np.abs(radii - 0.1)))
    x = d["dh_norm"][ri].astype(float)          # (dir, layer, token)
    a = np.median(x[:, -1, :], axis=0) / radii[ri]
    lo, hi = np.percentile(x[:, -1, :], [25, 75], axis=0) / radii[ri]
    resp[n] = dict(a=a, lo=lo, hi=hi, r=float(radii[ri]), n_dir=x.shape[0],
                   L=int(x.shape[1]), T=x.shape[2])
    T = len(a)
    sl, _, r2 = loglog_slope(np.arange(max(1, T - 100), T), np.log(a[max(1, T - 100):]))
    resp[n]["late_slope"] = sl
    resp[n]["late_r2"] = r2
    print(f"A1 {n:16s} r={radii[ri]:.3g} dirs={x.shape[0]} L={x.shape[1]} "
          f"a(1)={a[1]:.2f} max={a.max():.2f} late_slope={sl:+.4f}")

# ------------------------------------------------------------------ replay runs
REPLAY = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Qwen3.5-9B",
          "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]
rep = {}
for n in REPLAY:
    run = None
    for rr in softloop_runs(n, mode="replay"):
        if rr["T"] < 1e-6:
            run = rr
    if run is None:
        print("no replay", n); continue
    z = run["data"]
    dh = z["dh"].astype(float)                  # (dir, layer, token)
    r = float(run["meta"]["r"])
    a = np.median(dh[:, -1, :], axis=0) / r
    lo, hi = np.percentile(dh[:, -1, :], [25, 75], axis=0) / r
    rep[n] = dict(a=a, lo=lo, hi=hi, r=r, n_dir=dh.shape[0], L=dh.shape[1], T=dh.shape[2])
    T = len(a)
    sl, _, r2 = loglog_slope(np.arange(max(1, T - 100), T), np.log(a[max(1, T - 100):]))
    rep[n]["late_slope"] = sl
    rep[n]["late_r2"] = r2
    print(f"REP {n:16s} r={r:.3g} dirs={dh.shape[0]} steps={dh.shape[2]} "
          f"a(1)={a[1]:.2f} max={a.max():.2f} late_slope={sl:+.4f}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))

ax = axes[0, 0]
for n in A1_ORDER:
    v = resp[n]
    t = np.arange(len(v["a"]))
    ax.plot(t, v["a"] / v["a"][1], color=COL[n],
            label=f"{n} (r={v['r']:.2g}, a$_1$={v['a'][1]:.1f})")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xlim(0, 300)
ax.set_xlabel("generated token  $t$")
ax.set_ylabel(r"$\|\delta h_{t,L}\| \,/\, \|\delta h_{1,L}\|$  (last block, median)")
ax.set_title("(a) Response shape: all models flat or decaying after an early burst", loc="left")
ax.legend(fontsize=6.9)
ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
for n in REPLAY:
    if n not in rep:
        continue
    v = rep[n]
    t = np.arange(len(v["a"]))
    ax.plot(t, v["a"], color=COL[n], ls="--" if n in ("Mamba-130m", "Pythia-410m", "randQwen") else "-",
            label=n)
ax.set_yscale("log")
ax.set_xlim(0, 300)
ax.set_xlabel("generated token  $t$")
ax.set_ylabel(r"$\|\delta h_{t,L}\|/r$  (replay, last block, median)")
ax.set_title("(b) Network-only replay (identical tokens): LLMs vs controls", loc="left")
ax.legend(fontsize=6.6, ncol=2)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 0]
taus = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120])
for n in REPLAY:
    if n not in rep:
        continue
    a = rep[n]["a"]
    ms, es = [], []
    for tau in taus:
        if len(a) <= tau + 1:
            ms.append(np.nan); es.append(np.nan); continue
        lam = np.log(a[tau:] / a[:-tau]) / tau
        lam = lam[np.isfinite(lam)]
        ms.append(np.mean(lam)); es.append(np.std(lam) / max(np.sqrt(len(lam)), 1))
    ax.errorbar(taus, ms, yerr=es, color=COL[n], marker="o", ms=3, lw=1.4, capsize=2,
                ls="--" if n in ("Mamba-130m", "Pythia-410m", "randQwen") else "-", label=n)
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"averaging window $\tau$ (tokens)")
ax.set_ylabel(r"finite-time exponent  $\Lambda(\tau)=\frac{1}{\tau}\log\frac{a(t+\tau)}{a(t)}$")
ax.set_title("(c) Finite-time exponent converges to $\\leq 0$ for every system", loc="left")
ax.set_ylim(-0.02, 0.03)
ax.legend(fontsize=6.3, ncol=2)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 1]
labels, vals, errs, cols = [], [], [], []
for n in REPLAY:
    if n not in rep:
        continue
    a = rep[n]["a"]
    t = np.arange(len(a))
    k = 20
    lam = (np.log(a[k:]) - np.log(a[:-k])) / k
    late = lam[t[k:] > 150]
    labels.append(n); vals.append(np.mean(late)); errs.append(np.std(late)); cols.append(COL[n])
y = np.arange(len(labels))
ax.barh(y, vals, xerr=errs, color=cols, alpha=0.85, capsize=2)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.6)
ax.set_xlabel(r"late rate $\lambda_{20}$ for $t>150$  (nats/token)")
ax.set_title("(d) Late-time rate is within $\\pm0.01$ for every system", loc="left")
ax.grid(alpha=0.3, axis="x")

fig.suptitle("Time dynamics with identical tokens: the perturbation does not grow in any system tested", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_time_dynamics_all.png")

json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
               for kk, vv in v.items()} for k, v in {**resp, **rep}.items()},
          open(os.path.join(HERE, "analysis_update", "time_numbers.json"), "w"), indent=1)

