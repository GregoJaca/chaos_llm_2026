"""Settle: is sigma_1 of the same-position product the real stretching? Compare three objects."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})
NUM = {}

# 1. depth-map sigma_1 (same-position, from the embedding)
z = np.load(os.path.join(ROOT, "results", "depth2", "deepseek128", "depth.npz"))
L = z["layers"]
sig = np.exp(np.median(z["lam"][:, :, 0], axis=0) * (L + 1))
sig_block = np.median(z["block_sv"][:, :, 0], axis=0) if "block_sv" in z else None

# 2. probe: same-position vs cross-position, injection at block 0's output
def probe(tag, pos):
    f = os.path.join(LD, "probe2", tag, f"pos{pos}", "layer_probe.npz")
    zz = np.load(f)
    dh = zz["dh"]; r = float(zz["r"]); p = int(zz["pos"])
    out = {}
    for name, off in [("same", 0), ("next", 1), ("plus20", 20)]:
        out[name] = np.median(dh[0, :, :, p + off], axis=0) / r
    return out

ds = probe("deepseek", 16)
ds50 = probe("deepseek", 50)
ph = probe("phi4mini", 16)
q4 = probe("qwen3_4b", 16)

# 3. the real prompt-perturbation response at a generated position (a1, all layers)
a1 = np.load(os.path.join(LD, "big", "all_layers", "a1.npz"))
A = a1["dh_norm"].astype(float); La = np.asarray(a1["layers"])
ri = int(np.argmin(np.abs(a1["radii"] - 0.1)))
a_t = np.median(A[ri, :, :, 50], axis=0) / 0.1
h_t = a1["norm_h0"][np.arange(len(La)), 50]

fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))

ax = axs[0]
ax.semilogy(L, sig, "o-", ms=3, color="tab:blue", label=r"depth map: $\sigma_1(\ell)$ (128 tokens)")
if sig_block is not None:
    ax.semilogy(L, sig_block, "s--", ms=2.5, color="0.55",
                label=r"depth map: block-$\ell$ singular value")
ax.semilogy(La[1:], a_t[1:] / a_t[1], "o-", ms=3, color="tab:red",
            label=r"a1: actual prompt response (norm. at L1)")
ax.set_xlabel(r"block $\ell$"); ax.set_ylabel("stretch (log)")
ax.set_title("(a) Same-position chain vs the real response")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")

ax = axs[1]
for d, name, c in [(ds, "DeepSeek s=16", "tab:blue"), (ds50, "DeepSeek s=50", "tab:cyan"),
                   (ph, "Phi s=16", "tab:green"), (q4, "Qwen3-4B s=16", "tab:orange")]:
    ax.semilogy(np.arange(len(d["same"])), np.maximum(d["same"], 1e-3), "-", lw=1.4, color=c,
                label=f"{name}: same position")
    ax.plot(np.arange(len(d["next"])), np.maximum(d["next"], 1e-3), ":", lw=1.1, color=c,
            label=f"{name}: next token")
ax.set_xlabel("block $\\ell$"); ax.set_ylabel(r"measured $\|\delta h\|/r$")
ax.set_title("(b) One-token kick: the same position vs the next token")
ax.legend(fontsize=6.5, ncol=1); ax.grid(alpha=0.25, which="both")

ax = axs[2]
for d, name, c in [(ds, "DeepSeek s=16", "tab:blue"), (ph, "Phi s=16", "tab:green"),
                   (q4, "Qwen3-4B s=16", "tab:orange")]:
    ax.plot(np.arange(len(d["plus20"])), np.maximum(d["plus20"], 1e-3), "-", lw=1.4, color=c,
            label=f"{name}: $t+20$")
ax.set_yscale("log")
ax.set_xlabel("block $\\ell$"); ax.set_ylabel(r"measured $\|\delta h\|/r$")
ax.set_title("(c) Twenty tokens later: decays (DeepSeek) or grows (Phi)")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sigma_vs_real.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

NUM["sigma1_L1"] = float(sig[1]); NUM["sigma1_L27"] = float(sig[-1])
NUM["a1_L1"] = float(a_t[1]); NUM["a1_L27"] = float(a_t[-1])
NUM["a1_growth"] = float(a_t[-1] / a_t[1])
NUM["sigma_growth"] = float(sig[-1] / sig[1])
NUM["probe_same_last"] = {k: float(np.median(v["same"][-1])) for k, v in
                          dict(ds=ds, ds50=ds50, ph=ph, q4=q4).items()}
NUM["probe_plus20_last"] = {k: float(np.median(v["plus20"][-1])) for k, v in
                            dict(ds=ds, ds50=ds50, ph=ph, q4=q4).items()}
json.dump(NUM, open(os.path.join(ROOT, "results", "sigma_vs_real.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1))
