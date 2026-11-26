import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Qwen3.5-9B",
          "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]

def stats_for(n):
    loop = {rr["T"]: rr for rr in softloop_runs(n, "loop")}
    repl = {rr["T"]: rr for rr in softloop_runs(n, "replay")}
    out = {}
    for T, rr in loop.items():
        if T not in repl:
            continue
        dl = rr["data"]; dr = repl[T]["data"]
        r = float(rr["meta"]["r"])
        al = dl["dh"][:, -1, :].astype(float) / r
        ar = dr["dh"][:, -1, :].astype(float) / r
        hn = float(np.median(dl["h0n"][-1].astype(float)))
        sat = 0.25 * hn
        ts = []
        for i in range(al.shape[0]):
            idx = np.where(al[i] > sat)[0]
            ts.append(int(idx[0]) if len(idx) else np.nan)
        ts = np.array(ts, float)
        def rate(a, tmax):
            T_ = a.shape[-1]
            t = np.arange(3, int(min(tmax, T_)) if np.isfinite(tmax) else T_)
            if len(t) < 8:
                return np.nan
            am = np.median(a[:, t], axis=0)
            return float(np.polyfit(t, np.log(am), 1)[0])
        tmed = np.nanmedian(ts) if np.isfinite(ts).any() else np.nan
        lam_sep = rate(al, tmed) if (np.isfinite(tmed) and tmed >= 8) else np.nan
        lam_net = rate(ar, np.nan)
        out[T] = dict(ts=ts, al=al, ar=ar, sat=sat, hnorm=hn, lam_sep=lam_sep,
                      lam_net=lam_net, r=r, n=al.shape[0], Tsteps=al.shape[1],
                      med_margin=float(np.median(dl["m_ref"])),
                      frac_sat=float(np.mean(np.isfinite(ts))))
    return out

ALL = {}
for n in MODELS:
    try:
        ALL[n] = stats_for(n)
        print(n, "temps", sorted(ALL[n].keys()))
    except Exception as e:
        print("ERR", n, e)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))

# (a) DeepSeek loop vs replay at selected T
ax = axes[0, 0]
sel = [0.2, 0.5, 1.0, 2.0]
cmap = plt.cm.plasma(np.linspace(0.05, 0.9, len(sel)))
for T, c in zip(sel, cmap):
    if T not in ALL["DeepSeek-1.5B"]:
        continue
    d = ALL["DeepSeek-1.5B"][T]
    t = np.arange(d["Tsteps"])
    al = np.median(d["al"], axis=0); ar = np.median(d["ar"], axis=0)
    ax.plot(t, al / al[1], color=c, label=f"closed loop T={T:g}")
    ax.plot(t, ar / ar[1], color=c, ls="--", lw=1.0)
ax.axhline(1, color="k", lw=0.6, ls=":")
ax.set_yscale("log")
ax.set_xlim(0, 300)
ax.set_xlabel("step  $t$")
ax.set_ylabel(r"$\|\delta h_t\| \,/\, \|\delta h_1\|$  (median over directions)")
ax.set_title("(a) DeepSeek: closed loop (solid) vs network-only replay (dashed)", loc="left")
ax.legend(fontsize=7.2, ncol=2)
ax.grid(True, which="both", alpha=0.3)

# (b) lambda_sep vs lambda_net scatter
ax = axes[0, 1]
for n in MODELS:
    if n not in ALL:
        continue
    xs, ys = [], []
    for T, d in ALL[n].items():
        if np.isfinite(d["lam_sep"]) and np.isfinite(d["lam_net"]):
            xs.append(d["lam_net"]); ys.append(d["lam_sep"])
    if xs:
        ax.scatter(xs, ys, color=COL[n], s=32, label=n, alpha=0.85, edgecolor="white", lw=0.4)
ax.axvline(0, color="#1f77b4", lw=1.0, ls="--", label=r"$\lambda_{\rm net}=0$")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(-0.01, 0.005)
ax.set_ylim(-0.1, 0.5)
ax.set_xlabel(r"network-only rate  $\lambda_{\rm net}$  (nats/step)")
ax.set_ylabel(r"closed-loop rate  $\lambda_{\rm sep}$  (nats/step)")
ax.set_title("(b) Instability appears only on the closed-loop axis", loc="left")
ax.legend(fontsize=6.4, ncol=2)
ax.grid(alpha=0.3)

# (c) lambda_net across models
ax = axes[1, 0]
labels = []
for i, n in enumerate(MODELS):
    if n not in ALL:
        continue
    Ts = sorted(ALL[n].keys())
    y = np.array([ALL[n][T]["lam_net"] for T in Ts])
    ax.plot(np.arange(len(Ts)), y, "o-", color=COL[n], ms=4, label=n)
    labels.append(n)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("temperature index (each model's own grid)")
ax.set_ylabel(r"$\lambda_{\rm net}$  (nats/step)")
ax.set_title("(c) Network-only rate is within $\\pm0.006$ for every model and T", loc="left")
ax.legend(fontsize=6.6, ncol=2)
ax.grid(alpha=0.3)

# (d) horizon vs T, DeepSeek and Qwen
ax = axes[1, 1]
for n in ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini"]:
    if n not in ALL:
        continue
    Ts = [T for T in sorted(ALL[n].keys()) if T > 1e-3]
    med = [np.nanmedian(ALL[n][T]["ts"]) if np.isfinite(ALL[n][T]["ts"]).any() else np.nan for T in Ts]
    ax.plot(Ts, med, "o-", color=COL[n], ms=4, label=n)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"median separation step $t_{\rm sat}$  (tokens)")
ax.set_title("(d) Predictability horizon vs temperature", loc="left")
ax.legend(fontsize=7.4)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Soft-token feedback: the network contracts; only the readout/feedback can destabilise", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_softloop_open_closed.png")

json.dump({n: {str(T): {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in d.items() if k not in ("al", "ar")}
                for T, d in dd.items()} for n, dd in ALL.items()},
          open(os.path.join(HERE, "analysis_update", "softloop_numbers.json"), "w"), indent=1)
print("saved softloop_numbers.json")
