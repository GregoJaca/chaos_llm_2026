import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def H2(m):
    p = 1 / (1 + np.exp(-m))
    return -p * np.log(p + 1e-300) - (1 - p) * np.log(1 - p + 1e-300)

mods = ["DeepSeek-1.5B", "Qwen3-4B", "Phi-4-mini", "randQwen"]
D = {}
for n in mods:
    ms, hs = [], []
    for rr in softloop_runs(n, "loop"):
        z = rr["data"]
        m = z["m_ref"].astype(float); h = z["ent_ref"].astype(float)
        ok = np.isfinite(m) & np.isfinite(h) & (m >= 0)
        ms.append(m[ok]); hs.append(h[ok])
    D[n] = (np.concatenate(ms), np.concatenate(hs))
    print(n, "n", len(D[n][0]), "H med", np.median(D[n][1]),
          "frac H<0.1", np.mean(D[n][1] < 0.1))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
ax = axes[0]
for n in mods:
    m, h = D[n]
    idx = np.arange(0, len(m), max(1, len(m) // 4000))
    ax.scatter(m[idx], h[idx] + 1e-4, s=4, alpha=0.25, color=COL[n], label=n)
mm = np.linspace(0, 12, 300)
ax.plot(mm, H2(mm) + 1e-4, "k--", lw=1.4, label="two-token entropy $H_2(m)$")
ax.set_yscale("log")
ax.set_xlabel(r"logit margin  $m_t=z_{(1)}-z_{(2)}$  (nats)")
ax.set_ylabel(r"entropy of $\mathrm{softmax}(z)$ at $T=1$  (nats)")
ax.set_title("(a) Trained models: low entropy; random weights: $\\log V$", loc="left")
ax.legend(fontsize=7.2, markerscale=3)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n in mods:
    h = D[n][1]
    x = np.sort(h) + 1e-5
    ax.plot(x, 1 - np.arange(len(x)) / len(x), color=COL[n], lw=1.6, label=n)
ax.axvline(np.log(151936), color="k", ls=":", lw=1.0, label=r"$\log V=11.93$")
ax.set_xscale("log")
ax.set_xlabel(r"entropy  $H$  (nats, log scale)")
ax.set_ylabel(r"$P(H>x)$")
ax.set_title("(b) $H$ is a property of the logits, not of the run temperature", loc="left")
ax.legend(fontsize=7.4)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Entropy: trained models are concentrated; the random control is uniform over the vocabulary",
             fontsize=11.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig_entropy.png")
