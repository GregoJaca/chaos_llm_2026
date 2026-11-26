"""Does the response grow with depth?  LLMs vs controls.

Response ||dh_l(t)|| at t=50, normalized first by the layer-1 response (to remove
the arbitrary probe radius) and then by the middle-layer value, so the *shape*
along depth is comparable across models with very different depth and scale.
The bar is the last/middle ratio: >1 means deeper layers respond more.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

CTRL = {"Mamba-130m": "results_big_small/ctrl_Mamba_130m.npz",
        "Pythia-410m": "results_big_small/ctrl_Pythia_410m.npz",
        "randQwen": "results_big_small/ctrl_randQwen.npz"}
ORDER = all_names() + list(CTRL)


def prof(n, radius=0.1):
    if n in SMALL:
        z = a1(n)
        ri = int(np.argmin(np.abs(z["radii"] - radius)))
        return z["med"][ri], z["layers"].astype(float)
    if n in BIG:
        z = big(n)
        return z["dh_med"], z["layers"].astype(float)
    z = load(CTRL[n])
    ri = int(np.argmin(np.abs(z["radii"] - radius)))
    return z["dh_med"][ri], z["layers"].astype(float)


rows = {}
fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
ax = axes[0]
for n in ORDER:
    try:
        m, l = prof(n)
    except Exception:
        continue
    t = min(50, m.shape[-1] - 1)
    p = m[:, t] / max(m[1, t], 1e-30)
    mid = p[len(p) // 2]
    x = (l - l[0]) / (l[-1] - l[0])
    ax.plot(x[1:], p[1:] / mid, "o-", ms=3, color=COL[n], lw=1.3,
            ls="--" if n in CTRL else "-", label=n)
    rows[n] = dict(last_over_mid=float(p[-1] / mid), n_layers=int(len(l)))
ax.axhline(1, color="k", ls=":", lw=0.9)
ax.set_yscale("log")
ax.set_xlabel(r"normalized depth  $(\ell-\ell_0)/(L-1)$  (layer 0 dropped)")
ax.set_ylabel(r"$\|\delta h_\ell\|\ /\ \|\delta h_{\rm mid}\|$  at $t=50$")
ax.set_title("(a) Response grows with depth in every trained model", loc="left")
ax.legend(fontsize=6.0, ncol=3); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
names = [n for n in ORDER if n in rows]
vals = [rows[n]["last_over_mid"] for n in names]
y = np.arange(len(names))
for i, n in enumerate(names):
    ax.barh(i, vals[i] - 1, left=1, color=COL[n], alpha=0.85,
            hatch="//" if n in CTRL else None)
    ax.text(vals[i] + 0.15, i, f"{vals[i]:.1f}", va="center", fontsize=6.6)
ax.axvline(1, color="k", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels([n + ("  (control)" if n in CTRL else "") for n in names],
                                    fontsize=7.4)
ax.set_xlabel(r"last-layer / middle-layer response  at $t=50$")
ax.set_title("(b) The random transformer is flat; trained models are not", loc="left")
ax.grid(alpha=0.3, axis="x")
ax.set_xlim(0.8, max(vals) * 1.12)
fig.tight_layout()
save(fig, "pack_layer_gain.png")

json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_layer_gain.json"), "w"), indent=1)
for n in names:
    print(n, round(rows[n]["last_over_mid"], 2))
