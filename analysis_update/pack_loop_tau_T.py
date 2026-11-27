"""How do L_inf and tau depend on temperature T and on the attention window W?

For every (model, T) with a matched loop/replay pair we fit
    L(t) = L_inf (1 - exp(-t/tau))          (no free offset: L(1)=0)
The window dependence uses the dedicated W-sweep runs (results/longloop_v3/ds15/W*/
or any run whose meta.window differs), at T=1.
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE
from looputil import TAGS, gather, pairs, meta_layers_of
from style import plt, save


def sat(tt, L, tau):
    return L * (1.0 - np.exp(-tt / tau))


def fit_pair(name, T):
    ac, ar, used = gather(name, T)
    if ac is None or ac.shape[1] < 50:
        return None
    Ld = np.log(np.maximum(ac, 1e-30)) - np.log(np.maximum(ar, 1e-30))
    m = np.nanmedian(Ld, axis=0)
    t = np.arange(len(m))
    try:
        popt, _ = curve_fit(sat, t[1:], m[1:], p0=[max(m[-1], 0.5), max(len(t) / 5, 5)],
                            bounds=([0, 0.5], [60, 5e4]), maxfev=40000)
    except Exception:
        return None
    pred = sat(t[1:], *popt)
    r2 = 1 - np.sum((m[1:] - pred) ** 2) / np.sum((m[1:] - m[1:].mean()) ** 2)
    k0 = int(0.8 * len(m))
    return dict(Linf=float(popt[0]), tau=float(popt[1]), r2=float(r2),
                Lend=float(np.nanmedian(m[k0:])), n=int(ac.shape[0]), steps=int(len(m)))


rows = {}
for name in TAGS:
    for T, lab in [("T0.5", 0.5), ("T1.0", 1.0), ("T2.0", 2.0)]:
        r = fit_pair(name, T)
        if r:
            rows.setdefault(name, {})[str(lab)] = r
            print("%-14s T=%-4s Linf=%5.2f tau=%7.1f r2=%5.2f Lend=%5.2f n=%d" % (
                name, lab, r["Linf"], r["tau"], r["r2"], r["Lend"], r["n"]))

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
ax = axes[0]
for name, d in rows.items():
    if len(d) < 2:
        continue
    Ts = sorted(float(k) for k in d)
    ax.plot(Ts, [d[str(T)]["Lend"] for T in Ts], "o-", color=COL.get(name, "gray"), label=name)
ax.set_xscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"$L_{\rm end}$ (nats)")
ax.set_title("(a) Saturation level vs $T$", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for name, d in rows.items():
    if len(d) < 2:
        continue
    Ts = sorted(float(k) for k in d)
    ax.plot(Ts, [d[str(T)]["tau"] for T in Ts], "o-", color=COL.get(name, "gray"), label=name)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"rise time $\tau$ (steps)")
ax.set_title(r"(b) $\tau$ vs $T$", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)

# window dependence from meta.window of every run, T=1
ax = axes[2]
wrow = {}
for name in TAGS:
    for pl, pr in pairs(name, "T1.0"):
        for cand in (os.path.dirname(pl), os.path.dirname(os.path.dirname(pl))):
            mp = os.path.join(cand, "meta.json")
            if os.path.exists(mp):
                W = json.load(open(mp)).get("window")
                if W and W < 4096:
                    Lend = None
                    ac, ar, used = gather(name, "T1.0")
                    if ac is not None:
                        Ld = np.log(np.maximum(ac, 1e-30)) - np.log(np.maximum(ar, 1e-30))
                        k0 = int(0.6 * ac.shape[1])
                        Lend = float(np.median(Ld[:, k0:]))
                    if Lend is not None:
                        wrow.setdefault(name, []).append((W, Lend))
                break
for name, v in wrow.items():
    v = sorted(set(v))
    if len(v) < 2:
        continue
    ax.plot([x[0] for x in v], [x[1] for x in v], "o-", color=COL.get(name, "gray"), label=name)
ax.set_xscale("log")
ax.set_xlabel("attention window $W$ (tokens)")
ax.set_ylabel(r"$L_{\rm end}$ at $T=1$ (nats)")
ax.set_title("(c) Saturation level vs window", loc="left")
ax.legend(fontsize=6.2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_tau_T.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_loop_tau_T.json"), "w"), indent=1)
print("done")
