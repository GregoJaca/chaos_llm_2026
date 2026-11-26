"""Temperature dependence of the closed-loop feedback gain, DeepSeek-1.5B.

Why only one model: the probe radius differs across models (0.03 for DeepSeek,
0.1 for Phi-4-mini), and at T->0 the response is dominated by *discrete token
flips*, whose rate depends on the radius.  So a cross-model comparison at
T->0 would be confounded.  DeepSeek-1.5B has a full T ladder at one radius in
fp32, which is the clean experiment.

(a) L(t) = log(a_closed/a_replay) for every temperature.
(b) L at t=300 against T.  The greedy point (T=1e-9) is the control: there the
    closed loop IS ordinary autoregressive generation and the replay is teacher
    forcing, so their difference is the "self-generated vs teacher-forced" gap.
"""
import os
import sys
import glob
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

TAG = "ds15"
FLOOR = 1e-6
HORIZON = 300


def pairs(tag):
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, "results/softloop", tag, "T*"))):
        p = os.path.join(d, "softloop.npz")
        r = os.path.join(d, "replay", "softloop.npz")
        if not (os.path.exists(p) and os.path.exists(r)):
            continue
        T = float(os.path.basename(d)[1:])
        a = np.median(np.load(p, allow_pickle=True)["dh"][:, -1, :].astype(float), axis=0)
        b = np.median(np.load(r, allow_pickle=True)["dh"][:, -1, :].astype(float), axis=0)
        n = min(len(a), len(b))
        out.append((T, a[:n] / max(a[1], 1e-30), b[:n] / max(b[1], 1e-30)))
    return sorted(out, key=lambda r: r[0])


def Lcurve(a, b):
    ok = (a > FLOOR) & (b > FLOOR)
    return np.where(ok, np.log(np.maximum(a, 1e-30)) - np.log(np.maximum(b, 1e-30)), np.nan)


P = pairs(TAG)
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
ax = axes[0]
Ts = np.array([r[0] for r in P])
norm = plt.matplotlib.colors.Normalize(np.log10(max(Ts.min(), 1e-3)), np.log10(Ts.max()))
for T, a, b in P:
    L = Lcurve(a, b)
    c = plt.cm.viridis(norm(np.log10(max(T, 1e-3))))
    ax.plot(L, color=c, lw=1.15, ls="--" if T < 1e-3 else "-",
            label=f"$T$={T:g}" + ("  (greedy control)" if T < 1e-3 else ""))
ax.axhline(0, color="k", lw=0.9)
ax.set_xlim(0, 600)
ax.set_xlabel("step $t$")
ax.set_ylabel(r"$\log(a_{\rm closed}/a_{\rm replay})$  (nats)")
ax.set_title("(a) Feedback gain vs $t$, every temperature", loc="left")
ax.legend(fontsize=5.8, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
ts, ls = [], []
for T, a, b in P:
    L = Lcurve(a, b)
    L = L[np.isfinite(L)]
    ts.append(T); ls.append(float(L[min(HORIZON, len(L) - 1)]))
ts, ls = np.array(ts), np.array(ls)
ax.plot(ts[ts >= 1e-3], ls[ts >= 1e-3], "o-", color=COL["DeepSeek-1.5B"], ms=6,
        label="soft loop vs replay")
ax.plot(ts[ts < 1e-3], ls[ts < 1e-3], "s", color="#7f7f7f", ms=8, mec="k",
        label="$T\\to0$: greedy AR vs teacher forcing")
ax.axhline(0, color="k", lw=0.9)
ax.set_xscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(rf"$L(t={HORIZON})$  (nats)")
ax.set_title("(b) Any soft feedback gives the same gain; greed does not", loc="left")
ax.legend(fontsize=7.2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_temperature.png")
json.dump({"T": ts.tolist(), "L300": ls.tolist()},
          open(os.path.join(HERE, "analysis_update", "pack_loop_temperature.json"), "w"), indent=1)
print(list(zip(np.round(ts, 3), np.round(ls, 2))))
