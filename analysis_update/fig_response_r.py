import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# DeepSeek: the full 12-radius sweep
z = np.load(os.path.join(HERE, "results/local_data/w100_layers/a1.npz"), allow_pickle=True)
rd = z["radii"].astype(float)
t = 50
y_ds = np.array([np.median(z["dh_norm"][i, :, -1, t]) for i in range(len(rd))]) / rd

# other models: 3 radii (0.03, 0.1, 0.3)
SMALL = ["Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Qwen3.5-2B"]
R = {}
for n in SMALL:
    d = load_a1(n)
    radii = d["radii"].astype(float)
    y = np.array([np.median(d["dh_norm"][i, :, -1, min(t, d["dh_norm"].shape[3] - 1)])
                  for i in range(len(radii))]) / radii
    R[n] = (radii, y)

fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))

ax = axes[0, 0]
ax.plot(rd, y_ds * rd, "o-", ms=5, color=COL["DeepSeek-1.5B"], label="DeepSeek-1.5B (12 radii)")
for n in SMALL:
    r, y = R[n]
    ax.plot(r, y * r, "s--", ms=4, color=COL[n], label=n + " (3 radii)")
rr = np.logspace(-3, 0.4, 50)
ax.plot(rr, rr, "k:", lw=1.0, label=r"linear guide $\|\delta h\|=r$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"perturbation norm $r$  (Euclidean, whole prompt tensor)")
ax.set_ylabel(r"$\|\delta h_{t=50,L}\|$  (absolute response)")
ax.set_title("(a) Only DeepSeek follows $\\delta h\\propto r$; the others are flat", loc="left")
ax.legend(fontsize=6.8)
ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
lr, ly = np.log(rd), np.log(y_ds)
alpha = np.diff(ly) / np.diff(lr)
xm = np.sqrt(rd[:-1] * rd[1:])
ax.plot(xm, alpha, "o-", ms=5, color=COL["DeepSeek-1.5B"])
ax.axhline(0, color="k", lw=0.9)
ax.axhline(1, color="#888888", lw=0.8, ls=":", label=r"$\alpha=+1$: response proportional to r^0 (noise floor)")
ax.axhline(-1, color="#888888", lw=0.8, ls="--", label=r"$\alpha=-1$: response saturates")
ax.set_xscale("log")
ax.set_xlabel(r"$r$  (log scale)")
ax.set_ylabel(r"local slope $\alpha=\mathrm{d}\log(\|\delta h\|/r)/\mathrm{d}\log r$")
ax.set_title(r"(b) $\alpha=0$ between $3\times10^{-3}$ and 0.3: exact linear response", loc="left")
ax.set_ylim(-1.4, 1.4)
ax.legend(fontsize=7.0)
ax.grid(True, which="both", alpha=0.3)
ax.axvspan(3e-3, 0.3, color="#eaf3ea", zorder=0)

# (c) superposition error vs total norm
ax = axes[1, 0]
pts = []
for sub in ["deepseek_r0.003", "deepseek_r0.01", "deepseek_r0.03", "deepseek",
            "deepseek_p2", "deepseek_p3", "deepseek_r003"]:
    p = os.path.join(HERE, "results/superposition", sub, "superposition.npz")
    zz = np.load(p, allow_pickle=True)
    r = float(zz["r"]); P = int(zz["prompt_len"])
    e = np.nanmedian(np.abs(zz["err_L27"]))
    c = np.nanmedian(zz["cos_L27"])
    pts.append((r * np.sqrt(P), e, c, sub))
pts.sort()
xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
ax.plot(xs, ys, "o-", color="#1f77b4", ms=6)
for x, y, c, s in pts:
    ax.annotate(f"cos={c:.3f}", (x, y), fontsize=6.6, xytext=(4, 6), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel(r"total perturbation norm  $r\sqrt{P}$")
ax.set_ylabel(r"superposition error  $\|A-S\|/\|A\|$  (median over tokens)")
ax.set_title("(c) Superposition holds to 5-7% at total norm 0.24", loc="left")
ax.grid(True, which="both", alpha=0.3)

# (d) replication
ax = axes[1, 1]
labels, errs, coss = [], [], []
for sub, lab in [("deepseek_r0.03", "prompt 1"), ("deepseek_p2", "prompt 2"),
                 ("deepseek_p3", "prompt 3"), ("deepseek_r003", "prompt 1 (repeat)")]:
    zz = np.load(os.path.join(HERE, "results/superposition", sub, "superposition.npz"), allow_pickle=True)
    labels.append(lab)
    errs.append(float(np.nanmedian(np.abs(zz["err_L27"]))))
    coss.append(float(np.nanmedian(zz["cos_L27"])))
x = np.arange(len(labels))
ax.bar(x, errs, color="#1f77b4", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$\|A-S\|/\|A\|$ at $r=0.03$")
ax.set_title("(d) Replication across prompts (cos(A,S) > 0.997)", loc="left")
ax.set_ylim(0, 0.09)
ax.grid(alpha=0.3, axis="y")
for i, (e, c) in enumerate(zip(errs, coss)):
    ax.text(i, e + 0.002, f"{e:.3f}\ncos {c:.3f}", ha="center", fontsize=7)

fig.suptitle("Linear response and its limits: where the perturbation is linear in its size", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_response_vs_radius.png")

json.dump(dict(deepseek=dict(r=rd.tolist(), y=y_ds.tolist()),
               models={n: dict(r=v[0].tolist(), y=v[1].tolist()) for n, v in R.items()}),
          open(os.path.join(HERE, "analysis_update", "radius_numbers.json"), "w"), indent=1)
print("saved")

