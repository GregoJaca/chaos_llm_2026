import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# ---------------------------------------------------------------- measured data
# response-vs-state angle: per-model percentiles from the measured A1 runs
ANG = {}
for tag, f, label in [
    ("Qwen3-14B", "notebook_big/figures/qwen314_numbers.json", "Qwen3-14B"),
    ("Phi-4", "notebook_big/figures/phi4_numbers.json", "Phi-4"),
    ("Qwen3.5-9B", "notebook_big/figures/syn_numbers.json", "Qwen3.5-9B"),
    ("DeepSeek-14B", "notebook_big/figures/syn_numbers.json", "DeepSeek-14B"),
]:
    d = json.load(open(os.path.join(HERE, f)))
    key = {"Qwen3-14B": "Qwen3-14B", "Phi-4": "phi-4",
           "Qwen3.5-9B": "Qwen3.5-9B", "DeepSeek-14B": "DS-14B"}[label]
    if "angles" in d:
        a = d["angles"][key]
    else:  # per-model files store a single 'angle' block
        a = d["angle"]
        a = dict(percentiles=a["angle_percentiles"], median=a["median_angle"],
                 mean=a["mean_angle"], dh_over_h_median=a.get("dh_over_h0_median"),
                 n=a.get("n_pairs"), source=f)
    ANG[label] = dict(p=a["percentiles"], med=a["median"], mean=a["mean"],
                      dh_h=a.get("dh_over_h_median"), n=a.get("n"),
                      src=a.get("source", f))

# 1.5B: notebook reports 86-87 deg from a 3-direction full-state run
ANG["DeepSeek-1.5B"] = dict(p=None, med=86.5, mean=None, dh_h=None, n=3,
                            src="notebook.md: depth_angle_test (3 dirs)")

# Jacobian top-singular-vector rotation angles (principal angle between
# consecutive layers), measured
ROT = {}
for tag, p in [("DeepSeek-1.5B", "results/nonnorm_w100_big/nonnorm.npz"),
               ("DeepSeek-14B", "results_big_small/results_big/ds14/nonnorm_w100/nonnorm.npz"),
               ("Qwen3-14B", "results_big_small/results_big/qwen314/nonnorm_w0/nonnorm.npz"),
               ("Phi-4", "results_big_small/results_big/phi4/nonnorm_w0/nonnorm.npz")]:
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    a = z["angles"].astype(float)
    ROT[tag] = dict(all=a[np.isfinite(a)], med=float(np.nanmedian(a)),
                    p10=float(np.nanpercentile(a, 10)), p90=float(np.nanpercentile(a, 90)))

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))

# (a) response-vs-state angle percentiles (p5-p95) per model
ax = axes[0, 0]
names = [n for n in ["DeepSeek-14B", "Phi-4", "Qwen3-14B", "Qwen3.5-9B", "DeepSeek-1.5B"] if n in ANG]
for i, n in enumerate(names):
    a = ANG[n]
    if a["p"] is not None:
        lo, hi = a["p"]["p5"], a["p"]["p95"]
        ax.plot([lo, hi], [i, i], color=COL[n], lw=5, alpha=0.45, solid_capstyle="round")
        ax.plot([a["p"]["p25"], a["p"]["p75"]], [i, i], color=COL[n], lw=9, alpha=0.85, solid_capstyle="round")
        ax.plot(a["med"], i, "o", color="white", ms=6, mec=COL[n], mew=2)
    else:
        ax.plot(a["med"], i, "o", color=COL[n], ms=7)
ax.axvline(90, color="k", ls="--", lw=1.0)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(["%s\n(%s dirs)" % (n, ANG[n]["n"]) for n in names], fontsize=8)
ax.set_xlabel(r"angle between response $\delta h_\ell$ and state $h_\ell$  (degrees)")
ax.set_title("(a) Response-vs-state angle: median is 90$^\\circ$ for every model", loc="left")
ax.set_xlim(55, 125)
ax.grid(alpha=0.35, axis="x")

# (b) radial fraction |cos(angle)|
ax = axes[0, 1]
for i, n in enumerate(names):
    a = ANG[n]
    if a["p"] is not None:
        c = [abs(np.cos(np.radians(a["p"][k]))) for k in ("p5", "p25", "p50", "p75", "p95")]
        ax.plot([c[0], c[4]], [i, i], color=COL[n], lw=1.2)
        ax.plot([c[1], c[3]], [i, i], color=COL[n], lw=6, alpha=0.85, solid_capstyle="round")
        ax.plot(c[2], i, "o", color="white", ms=6, mec=COL[n], mew=2)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(0, 0.42)
ax.set_xlabel(r"radial fraction $|\cos\angle(\delta h_\ell,h_\ell)|$")
ax.set_title("(b) Median radial fraction 0.0002-0.02, quartiles up to 0.1", loc="left")
ax.grid(alpha=0.35)

# (c) Jacobian rotation angles
ax = axes[1, 0]
for i, n in enumerate(ROT):
    a = ROT[n]["all"]
    ax.hist(a, bins=np.linspace(50, 91, 60), histtype="step", lw=1.8,
            color=COL[n], label=f"{n}: median {ROT[n]['med']:.1f}$^\\circ$", density=True)
ax.axvline(90, color="k", ls="--", lw=1.0, label="random vectors (90$^\\circ$)")
ax.set_xlabel("principal angle between top singular vectors of consecutive blocks (degrees)")
ax.set_ylabel("probability density")
ax.set_title("(c) Amplification directions rotate by 78-87$^\\circ$ between blocks", loc="left")
ax.legend(fontsize=7.4)
ax.set_xlim(50, 91)
ax.grid(alpha=0.35)

# (d) naive product of per-block sigma_max vs measured product gain
ax = axes[1, 1]
BS = {"DeepSeek-1.5B": "results/block_sigma/deepseek/block_sigma.npz",
      "Qwen3-4B": "results/block_sigma/qwen3_4b/block_sigma.npz",
      "Phi-4-mini": "results/block_sigma/phi4mini/block_sigma.npz"}
bar = []
for n, p in BS.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    smax = z["smax"].astype(float)                 # (n_tok, n_block)
    naive = np.exp(np.sum(np.log(np.median(smax, axis=0))))
    # measured product gain from the same-model depth run
    d = load_depth(n)
    ell = np.arange(d["lam"].shape[1])
    cum = (d["lam"].astype(float) * (ell + 1)[None, :, None])
    measured = float(np.exp(np.median(cum[:, -1, :])))
    bar.append((n, naive, measured))
x = np.arange(len(bar))
ax.bar(x - 0.18, [b[1] for b in bar], width=0.36, color="#d62728",
       label=r"naive $\prod_\ell \sigma_{\max}(J_\ell)$")
ax.bar(x + 0.18, [b[2] for b in bar], width=0.36, color="#2ca02c",
       label="measured product gain")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([b[0] for b in bar], fontsize=8)
ax.set_ylabel("gain (log scale)")
ax.set_title("(d) Composing per-block maxima overpredicts the measured gain by 16-23 dex", loc="left")
ax.legend(fontsize=7.6)
ax.grid(alpha=0.3, axis="y", which="both")
for i, (n, nv, mv) in enumerate(bar):
    ax.text(i, max(nv, mv) * 1.6, f"{np.log10(nv/mv):.1f} dex", ha="center", fontsize=7.5)

fig.suptitle("Response geometry: what is real and what is the high-dimensional null", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_angle_and_rotation.png")

json.dump(dict(response_angle=ANG,
               rotation={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                             for kk, vv in v.items()} for k, v in ROT.items()},
               product=bar),
          open(os.path.join(HERE, "analysis_update", "angle_numbers.json"), "w"), indent=1)
print("saved angle_numbers.json")
for b in bar:
    print(b[0], "naive %.2e measured %.2e ratio %.1e" % (b[1], b[2], b[1] / b[2]))
