import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini",
          "Mamba-130m", "Pythia-410m", "randQwen", "Qwen3.5-9B"]

def get(n, T=1.0, mode="loop"):
    for TT, z, m in soft_runs(n, mode):
        if abs(TT - T) < 1e-9:
            return z, m
    return None, None

rows = {}
for n in MODELS:
    zc, mc = get(n, 1.0, "loop")
    zr, mr = get(n, 1.0, "replay")
    if zc is None or zr is None:
        continue
    ac = zc["dh"][:, -1, :].astype(float)
    ar = zr["dh"][:, -1, :].astype(float)
    T = min(ac.shape[-1], ar.shape[-1])
    ac, ar = ac[:, :T], ar[:, :T]
    t = np.arange(10, T)
    # per-direction feedback exponent
    lfb = []
    for i in range(ac.shape[0]):
        r = np.log(np.maximum(ac[i, t], 1e-30)) - np.log(np.maximum(ar[i, t], 1e-30))
        lfb.append(np.polyfit(t, r, 1)[0])
    lfb = np.array(lfb)
    # total rates
    lc = np.log(np.maximum(ac[:, -1], 1e-30) / np.maximum(ac[:, 1], 1e-30)) / (T - 1)
    lr = np.log(np.maximum(ar[:, -1], 1e-30) / np.maximum(ar[:, 1], 1e-30)) / (T - 1)
    rows[n] = dict(lfb=lfb, lc=lc, lr=lr, T=T,
                   ac=np.median(ac, axis=0), ar=np.median(ar, axis=0))
    print(n, "lambda_fb median %.4f" % np.median(lfb), "closed %.4f" % np.median(lc),
          "replay %.4f" % np.median(lr))

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
ax = axes[0]
names = list(rows)
for i, n in enumerate(names):
    v = rows[n]["lfb"]
    med = np.median(v); lo, hi = np.percentile(v, [25, 75])
    ax.errorbar(med, i, xerr=[[med - lo], [hi - med]], fmt="o", color=COL[n], ms=6, capsize=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.6)
ax.set_xlabel(r"feedback exponent $\lambda_{\rm fb}=\mathrm{d}\log(a_{\rm closed}/a_{\rm replay})/\mathrm{d}t$")
ax.set_title("(a) Feedback gain per model, $T=1$", loc="left")
ax.grid(alpha=0.3, axis="x")

ax = axes[1]
for n in names:
    r = rows[n]["ac"] / np.maximum(rows[n]["ar"], 1e-30)
    ax.plot(np.arange(len(r)), r, color=COL[n], lw=1.3, label=n)
ax.set_yscale("log")
ax.set_xlim(0, 200)
ax.set_xlabel("step $t$")
ax.set_ylabel(r"$a_{\rm closed}(t)/a_{\rm replay}(t)$")
ax.set_title("(b) The feedback ratio grows with $t$", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_feedback.png")

json.dump({n: dict(lfb=np.median(rows[n]["lfb"]).item(),
                   lfb_lo=np.percentile(rows[n]["lfb"], 25).item(),
                   lfb_hi=np.percentile(rows[n]["lfb"], 75).item(),
                   lc=np.median(rows[n]["lc"]).item(),
                   lr=np.median(rows[n]["lr"]).item()) for n in rows},
          open(os.path.join(HERE, "analysis_update", "pack_loop_feedback.json"), "w"), indent=1)
print("done")
