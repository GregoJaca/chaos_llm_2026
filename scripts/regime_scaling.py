"""Post-clip forgetting rate scales with 1/W."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open(r"C:\Users\hjaca\research\chaos_llm_2026\results\figures_big\regime_rows.json"))
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def summary(key, wins):
    out = []
    for w in wins:
        v = [r[key] for r in rows if r["W"] == w and np.isfinite(r[key])]
        if not v:
            continue
        rng = np.random.default_rng(0)
        boot = [np.mean(rng.choice(v, len(v))) for _ in range(1000)]
        out.append((w, float(np.mean(v)), float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)), len(v)))
    return out


pre = summary("pre", [90, 100, 120, 150, 200, 300, 500, 4096])
post = summary("post", [50, 70, 90, 100, 120, 150, 200, 300])

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))

ax = axs[0]
w = [x[0] for x in pre]; m = np.array([x[1] for x in pre])
lo = m - np.array([x[2] for x in pre]); hi = np.array([x[3] for x in pre]) - m
ax.errorbar(w, m, yerr=[lo, hi], fmt="o-", capsize=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("window $W$")
ax.set_ylabel(r"mean $\lambda_{10}$ before clipping")
ax.set_title("(a) Pre-clip: neutral ($|\\lambda|<0.003$)")
ax.grid(alpha=0.25, which="both")

ax = axs[1]
w = np.array([x[0] for x in post]); m = np.array([x[1] for x in post])
lo = m - np.array([x[2] for x in post]); hi = np.array([x[3] for x in post]) - m
ax.errorbar(w, -m, yerr=[hi, lo], fmt="o-", capsize=3, color="tab:red")
ww = np.linspace(45, 310, 100)
A = 0.674
ax.plot(ww, A / ww, "k--", label=f"$A/W$, $A={A}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("window $W$")
ax.set_ylabel(r"contraction rate $-\lambda_{10}$ (nats/token)")
ax.set_title("(b) Post-clip: contraction $\\propto 1/W$")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

ax = axs[2]
ax.errorbar(w, -m * w, yerr=[lo * w, hi * w], fmt="o-", capsize=3, color="tab:green")
ax.set_xscale("log")
ax.set_xlabel("window $W$")
ax.set_ylabel(r"$-\lambda_{\mathrm{post}}\cdot W$  (tokens$^2$/token)")
ax.set_title("(c) The product is constant:\nforgetting is eviction-driven")
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(OUT + r"\regime_scaling.png", dpi=190)
print("pre", [(x[0], round(x[1], 4), x[4]) for x in pre])
print("post", [(x[0], round(x[1], 4), round(-x[1] * x[0], 2), x[4]) for x in post])
# fit A
ws = np.array([x[0] for x in post]); ms = -np.array([x[1] for x in post])
print("A = mean(-lambda*W) =", round(float(np.mean(ms * ws)), 3), "sd", round(float(np.std(ms * ws)), 3))
