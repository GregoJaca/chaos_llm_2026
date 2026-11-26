import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from chaos_llm.curvefit import fit_curve

rows = {}
for name in DEPTH:
    d = load_depth(name)
    lam = d["lam"].astype(float)          # (n_tok, n_block, k): cum/(l+1)
    nL = lam.shape[1]
    ell = np.arange(nL, dtype=float)
    cum = lam * (ell + 1)[None, :, None]  # log singular value of the product up to l
    cum_m = np.median(cum, axis=(0, 2))   # median over tokens and tracked directions
    cum_tok = np.median(cum, axis=2)      # median over directions, one curve per token
    cum_lo = np.percentile(cum_tok, 25, axis=0)
    cum_hi = np.percentile(cum_tok, 75, axis=0)
    lam_m = np.median(lam, axis=(0, 2))
    local = d["local_gain"].astype(float)  # mean ||J_l v|| on the tracked basis
    local_m = np.median(local, axis=0)
    G = np.exp(cum_m)
    rows[name] = dict(lam=lam_m, cum=cum_m, G=G, local=local_m,
                      cum_lo=cum_lo, cum_hi=cum_hi,
                      n=nL, depth_exp=float(lam_m[-1]),
                      total_gain=float(G[-1]),
                      first_gain=float(G[0]),
                      local_tail=float(np.median(local_m[1:])))
    print(f"{name:16s} L={nL:2d}  G_first={G[0]:7.2f}  G_last={G[-1]:9.1f}  "
          f"depth_exp={lam_m[-1]:.4f}  local_tail={np.median(local_m[1:]):.3f}")

# fit log G vs depth: is it linear?  (cum should be ~ linear)
for name, r in rows.items():
    a, b, r2 = loglog_slope(np.arange(r["n"]), r["cum"])
    r["lin_r2"], r["lin_slope"], r["lin_icpt"] = r2, a, b

# ---------------------------------------------------------------- figure
order = ["DeepSeek-1.5B", "DeepSeek-14B", "Qwen2.5-3B", "Qwen3-4B",
         "Qwen3-14B", "Qwen3.5-2B", "Phi-4-mini", "Phi-4"]
fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))

ax = axes[0, 0]
for n in order:
    r = rows[n]
    x = np.arange(r["n"])
    ax.plot(x, r["lam"], "-o", ms=2.6, color=COL[n], label=n)
ax.set_xlabel("block index $\\ell$")
ax.set_ylabel(r"$\lambda_\ell=\frac{1}{\ell+1}\log\sigma_1(J_\ell\cdots J_1)$")
ax.set_title("(a) The quantity that was fitted: a $1/\\ell$ artefact", loc="left")
ax.legend(fontsize=7.2, ncol=2)
ax.grid(alpha=0.4)

ax = axes[0, 1]
for n in order:
    r = rows[n]
    x = np.arange(r["n"])
    ax.plot(x, r["G"], "-o", ms=2.6, color=COL[n], label=n)
ax.set_yscale("log")
ax.set_xlabel("block index $\\ell$")
ax.set_ylabel(r"$\exp(\mathrm{cum}_\ell)=\sigma_1(J_\ell\cdots J_1)$  (geometric mean)")
ax.set_title("(b) The product gain itself: monotone, not decaying", loc="left")
ax.legend(fontsize=7.2, ncol=2)
ax.grid(True, which="both", alpha=0.35)

ax = axes[1, 0]
for n in order:
    r = rows[n]
    x = np.arange(r["n"])
    ax.plot(x, r["local"], "-o", ms=2.6, color=COL[n], label=n)
ax.axhline(1.0, color="k", ls="--", lw=0.9)
ax.set_yscale("log")
ax.set_xlabel("block index $\\ell$")
ax.set_ylabel(r"mean $\|J_\ell v_i\|$ on the tracked basis  (per-block gain)")
ax.set_title("(c) The tracked response is not amplified by later blocks ($\\approx1$)", loc="left")
ax.legend(fontsize=7.2, ncol=2)
ax.grid(True, which="both", alpha=0.35)

ax = axes[1, 1]
xs, ys, labels = [], [], []
for n in order:
    r = rows[n]
    ax.plot(np.arange(r["n"]) / (r["n"] - 1), r["cum"], "-", color=COL[n], label=n)
    ax.fill_between(np.arange(r["n"]) / (r["n"] - 1), r["cum_lo"], r["cum_hi"],
                    color=COL[n], alpha=0.12, lw=0)
    xs.append(MODEL_SIZE[n]); ys.append(r["depth_exp"]); labels.append(n)
ax.set_xlabel("normalized depth  $\\ell/(L-1)$")
ax.set_ylabel(r"$\log$ product gain  $\mathrm{cum}_\ell$")
ax.set_title("(d) Log gain: fast first block, plateau, late rise (not a straight line)", loc="left")
ax.legend(fontsize=7.0, ncol=2)
ax.grid(alpha=0.4)

fig.suptitle("Depth dynamics: what the Jacobian product actually does\n"
             "(all curves measured from block Jacobians at 8-16 probe tokens; median over tokens/directions)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig_depth_gain_all_models.png")

# second figure: depth exponent vs model size, and shape summary
fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.5))
ax = axes[0]
for n in order:
    r = rows[n]
    ax.scatter(MODEL_SIZE[n], r["depth_exp"], color=COL[n], s=55, zorder=3)
    ax.annotate(n, (MODEL_SIZE[n], r["depth_exp"]), fontsize=7.5,
                xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("model parameters (billions, log scale)")
ax.set_ylabel(r"mean per-block exponent $\log(G_L)/L$  (nats/block)")
ax.set_title("(a) Larger models are closer to marginal", loc="left")
ax.grid(alpha=0.4)

ax = axes[1]
for n in order:
    r = rows[n]
    ax.scatter(MODEL_SIZE[n], r["total_gain"], color=COL[n], s=55, zorder=3)
    ax.annotate(n, (MODEL_SIZE[n], r["total_gain"]), fontsize=7.5,
                xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("model parameters (billions, log scale)")
ax.set_ylabel(r"total product gain $G_L$  (geometric mean)")
ax.set_title("(b) Total gain spans $\\times$4 to $\\times$28", loc="left")
ax.grid(which="both", alpha=0.4)

ax = axes[2]
width = 0.26
for j, (key, lab) in enumerate([("first", "block 0"), ("mid", "middle half"), ("last", "last 10%")]):
    vals = []
    for n in order:
        r = rows[n]
        g = r["local"]
        if key == "first":
            v = g[0]
        elif key == "mid":
            v = np.median(g[len(g) // 4: 3 * len(g) // 4])
        else:
            v = np.median(g[-max(1, len(g) // 10):])
        vals.append(v)
    ax.bar(np.arange(len(order)) + (j - 1) * width, vals, width=width, label=lab)
ax.set_yscale("log")
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([n.replace("DeepSeek", "DS") for n in order], rotation=35, ha="right", fontsize=7.2)
ax.set_ylabel("local gain (mean over tracked directions)")
ax.set_title("(c) Block 0 share: 2% (Phi-4) to 85% (DeepSeek)", loc="left")
ax.legend(fontsize=7.4)
ax.grid(alpha=0.3, axis="y", which="both")
fig.tight_layout()
save(fig, "fig_depth_exponent_vs_size.png")

json.dump({n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in r.items()} for n, r in rows.items()},
          open(os.path.join(HERE, "analysis_update", "depth_numbers.json"), "w"), indent=1)
print("saved depth_numbers.json")

