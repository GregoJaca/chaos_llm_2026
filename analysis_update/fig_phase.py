import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-9B", "Phi-4-mini",
          "Mamba-130m", "Pythia-410m", "randQwen"]
rows = []
for n in MODELS:
    for rr in softloop_runs(n, "loop"):
        z = rr["data"]; T = rr["T"]
        if "m_ref" not in z:
            continue
        m = z["m_ref"].astype(float)
        dens = float(np.mean(m < T))
        dh = z["dh"][:, -1, :].astype(float)
        r = float(rr["meta"]["r"])
        hn = float(np.median(z["h0n"][-1].astype(float)))
        sat = 0.25 * hn
        ts = []
        for i in range(dh.shape[0]):
            idx = np.where(dh[i] / r > sat)[0]
            ts.append(int(idx[0]) if len(idx) else dh.shape[-1])
        rows.append(dict(model=n, T=T, dens=dens, ts=float(np.median(ts)),
                         frac=float(np.mean(np.array(ts) < dh.shape[-1]))))
    print(n, "done")

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
ax = axes[0]
for n in MODELS:
    sub = [r for r in rows if r["model"] == n]
    if not sub:
        continue
    Ts = [r["T"] for r in sub]; ts = [r["ts"] for r in sub]
    ax.plot(Ts, ts, "o-", color=COL[n], ms=4, label=n)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"median separation step $t_{\rm sat}$  (capped at run length)")
ax.set_title("(a) Horizon vs temperature: no single trend", loc="left")
ax.legend(fontsize=6.6, ncol=2)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n in MODELS:
    sub = [r for r in rows if r["model"] == n and r["T"] > 1e-3]
    if not sub:
        continue
    ax.scatter([r["dens"] for r in sub], [r["ts"] for r in sub],
               color=COL[n], s=34, label=n, alpha=0.9, edgecolor="white", lw=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"live-gate density: fraction of steps with $m_t<T$")
ax.set_ylabel(r"median $t_{\rm sat}$")
ax.set_title("(b) Horizon vs gate density: a trend, not a collapse", loc="left")
ax.legend(fontsize=6.6, ncol=2)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("The phase structure is set by the gate density, not by temperature directly", fontsize=11.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig_phase_density.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "phase_numbers.json"), "w"), indent=1)
