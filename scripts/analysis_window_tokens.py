"""Window token agreement — four panels, with a fit of the surviving fraction."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "window_tokens")
OUT = os.path.join(ROOT, "notebook", "figures", "veto")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

z = np.load(os.path.join(B, "tokens.npz"))
meta = json.load(open(os.path.join(B, "meta.json")))
P = int(meta["prompt_len"])
ws = sorted([int(k[1:]) for k in z.files])
ref = z[f"w{max(ws)}"]
T = len(ref)
rows = {}
for w in ws[:-1]:
    ok = (z[f"w{w}"] == ref).astype(float)
    first = int(np.argmax(ok == 0)) if (ok == 0).any() else -1
    rows[w] = dict(first_diff=first, frac=float(ok.mean()), agree=ok)

fig, axs = plt.subplots(2, 2, figsize=(13, 8.4))
ax = axs[0, 0]
for w in ws[:-1]:
    ax.plot(np.cumsum(rows[w]["agree"]) / np.arange(1, T + 1), lw=1.1, label=f"$W={w}$")
ax.set_xlabel("generated token $t$")
ax.set_ylabel("running fraction identical to $W=4096$")
ax.set_title("(a) Text agreement with the unwindowed continuation")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25)

wv = np.array([w for w in ws[:-1]])
first = np.array([rows[w]["first_diff"] for w in wv], float)
ax = axs[0, 1]
m = first > 0
ax.plot(wv[m], first[m], "o", ms=8, color="tab:blue", label="measured")
for w in wv[~m]: ax.annotate("identical\nfor 300", (w, 5), ha="center", fontsize=7)
ax.plot(wv, np.maximum(wv - P, 0), "k--", lw=1.2, label=r"$W-P$ (first token that can be lost)")
ax.set_xlabel("window $W$"); ax.set_ylabel("first differing token $t$")
ax.set_title("(b) Departure time: $W-P$ plus a few tokens")
ax.legend(); ax.grid(alpha=0.25)

ax = axs[1, 0]
lag = first - np.maximum(wv - P, 0)
ok = first > 0
ax.plot(wv[ok], lag[ok], "o-", ms=7, color="tab:purple")
ax.set_xlabel("window $W$"); ax.set_ylabel(r"$t_{\rm first} - (W-P)$")
ax.set_title("(c) Extra latency: 2–7 tokens (mean 5)")
ax.grid(alpha=0.25)

ax = axs[1, 1]
fr = np.array([rows[w]["frac"] for w in wv])
ax.semilogx(wv, fr, "o", ms=8, color="tab:red", label="measured")
r = fit_curve(wv, fr, families=["exponential", "exp+offset", "saturating", "stretched_rise",
                                "power", "log"])
xx = np.geomspace(wv.min(), wv.max(), 200)
ax.semilogx(xx, r.best.predict(xx), "-", lw=1.5, color="tab:red",
            label=f"best: {r.best.name} $R^2$={r.best.r2:.3f}")
for f in r.fits[1:3]:
    ax.semilogx(xx, f.predict(xx), "--", lw=1.0, label=f"{f.name} $R^2$={f.r2:.3f}")
ax.set_xlabel("window $W$"); ax.set_ylabel("fraction of 300 tokens identical")
ax.set_title("(d) Surviving fraction vs window")
ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_token_agreement_multi.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print(json.dumps({str(k): {kk: vv for kk, vv in v.items() if kk != "agree"} for k, v in rows.items()}, indent=1))
print("fit:", r.best.name, dict(zip(r.best.pnames, np.round(r.best.params, 4))), "R2", round(r.best.r2, 4))
print("lag:", lag[ok].tolist())
json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "agree"} for k, v in rows.items()},
          open(os.path.join(ROOT, "results", "window_tokens.json"), "w"), indent=1)
