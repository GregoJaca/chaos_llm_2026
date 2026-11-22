"""Figures for the veto evaluation: controlled margins, regime scaling, layer probe, latent agreement."""
import glob
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
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "veto")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

# ------------------------------------------------------------------ 1. controlled margins (same tokens)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axs[0]
for w, c in [(100, "tab:blue"), (150, "tab:orange"), (200, "tab:green"), (300, "tab:red"),
             (1000, "tab:purple")]:
    f = os.path.join(LD, "window_fixed", f"w{w}", "a1.npz")
    if not os.path.exists(f):
        continue
    m = np.load(f)["margins"].astype(float)
    m = m[m > 0]
    xs = np.sort(m)
    ccdf = 1 - np.arange(len(xs)) / len(xs)
    ax.loglog(xs, ccdf, lw=1.2, color=c, label=f"$W={w}$")
ax.set_xlabel("margin $m$ (top-two logit gap)")
ax.set_ylabel(r"$P(m>x)$")
ax.set_title("(a) Same tokens, five windows:\nmargin distribution is window-independent")
ax.legend(); ax.grid(alpha=0.25, which="both")

ax = axs[1]
w = 100
f = os.path.join(LD, "window_fixed", f"w{w}", "a1.npz")
m = np.load(f)["margins"].astype(float)
ax.plot(np.arange(len(m)), m, lw=0.8, color="0.3")
ax.set_yscale("log"); ax.set_xlabel("generated token $t$")
ax.set_ylabel("margin $m_t$ (same tokens, $W=100$)")
ax.set_title("(b) The margins dip at isolated tokens\nthe perturbation waits for one of these")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "margins_controlled.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 2. post-clip rate vs W
fig, ax = plt.subplots(figsize=(7.6, 4.6))
rows = []
for tag, c, name in [("deepseek", "tab:blue", "DeepSeek-1.5B"),
                     ("qwen3_4b", "tab:orange", "Qwen3-4B"),
                     ("phi4mini", "tab:green", "Phi-4-mini")]:
    for w in [60, 100, 200]:
        if tag == "deepseek":
            f = os.path.join(LD, "window_sweep", f"w{w}", "a1.npz")
        else:
            f = os.path.join(LD, "window4d", tag, f"w{w}", "a1.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        P = int(z["prompt_ids"].shape[0])
        a = np.median(z["dh_norm"][0, :, -1, :], axis=0)
        start = max(w - P + 5, 10)
        if start >= len(a) - 20:
            continue
        seg = a[start:]
        k = 10
        t = np.arange(k, len(seg) - k)
        lam = np.log(np.maximum(seg[t + k], 1e-12) / np.maximum(seg[t - k], 1e-12)) / (2 * k)
        rows.append((name, w, float(np.median(lam)), c))
for name, w, lam, c in rows:
    ax.loglog(w, -lam, "o", color=c, ms=9)
    ax.annotate(name, (w, -lam), textcoords="offset points", xytext=(6, 4), fontsize=7, color=c)
ws = np.array([r[1] for r in rows]); ys = -np.array([r[2] for r in rows])
if len(ws) > 1:
    A = float(np.exp(np.mean(np.log(ys) + np.log(ws))))
    xx = np.geomspace(ws.min(), ws.max(), 50)
    ax.loglog(xx, A / xx, "k--", lw=1.2, label=f"$A/W$, $A={A:.2f}$")
ax.set_xlabel("window $W$")
ax.set_ylabel(r"$-\lambda_{\rm post}$  (contraction rate after the window clips, nats/token)")
ax.set_title("The post-clip contraction falls as $1/W$")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regime_scaling_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 3. layer probe (three views)
B = os.path.join(LD, "probe2")
MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Phi-4-mini", "phi4mini", "tab:green")]
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
# (a) amplification vs injection block, several positions, DeepSeek
ax = axs[0]
for pos, c in [(8, "tab:blue"), (16, "tab:orange"), (40, "tab:green")]:
    f = os.path.join(B, "deepseek", f"pos{pos}", "layer_probe.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); r = float(z["r"]); p = int(z["pos"])
    v = [np.median(z["dh"][i][:, -1, p + 1:p + 51].max(axis=1)) / r for i in range(z["dh"].shape[0])]
    ax.plot(z["layers"], v, "o-", ms=4, color=c, label=f"$s={pos}$")
ax.set_xlabel("injection block $\\ell_0$"); ax.set_ylabel("amplification")
ax.set_title("(a) DeepSeek: flat in injection depth"); ax.set_yscale("log")
ax.legend(); ax.grid(alpha=0.25, which="both")
# (b) same-position vs downstream
ax = axs[1]
for name, tag, c in MODELS:
    f = os.path.join(B, tag, "pos16", "layer_probe.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); r = float(z["r"]); p = int(z["pos"])
    same = np.median(z["dh"][0][:, :, p], axis=0) / r
    later = np.median(z["dh"][0][:, :, p + 20], axis=0) / r
    ax.plot(np.arange(len(same)), same, "-", color=c, lw=1.3, label=f"{name} same")
    ax.plot(np.arange(len(later)), later, "--", color=c, lw=1.1, label=f"{name} $t+20$")
ax.set_yscale("log"); ax.set_xlabel("block $\\ell$"); ax.set_ylabel(r"$\|\delta h\|/r$")
ax.set_title("(b) $s=16$: same position vs 20 tokens later")
ax.legend(fontsize=6.5); ax.grid(alpha=0.25, which="both")
# (c) heatmap injection block x position
ax = axs[2]
pos_l, inj_l, M = [], [], []
for sub in sorted(os.listdir(os.path.join(B, "phi4mini"))):
    f = os.path.join(B, "phi4mini", sub, "layer_probe.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); r = float(z["r"]); p = int(z["pos"])
    pos_l.append(p)
    M.append([np.median(z["dh"][i][:, -1, p + 1:p + 51]) / r for i in range(z["dh"].shape[0])])
    inj_l = list(z["layers"])
M = np.array(M)[np.argsort(pos_l)]
pos_l = np.array(pos_l)[np.argsort(pos_l)]
im = ax.imshow(np.log10(M.T), aspect="auto", origin="lower", cmap="viridis",
               extent=[pos_l.min(), pos_l.max(), -0.5, len(inj_l) - 0.5])
ax.set_yticks(range(len(inj_l))); ax.set_yticklabels([str(int(x)) for x in inj_l])
ax.set_xlabel("position $s$"); ax.set_ylabel("injection block $\\ell_0$")
ax.set_title("(c) Phi: amplification vs (position, injection block)")
fig.colorbar(im, ax=ax, pad=0.01, label=r"$\log_{10}$ amplification")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "layer_probe_three_views.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 4. latent agreement + segmented fit
z = np.load(os.path.join(LD, "state_freerun", "sf.npz"))
ref = z["ref"]; pred = z["pred"]; steps = z["steps"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axs[0]
for di in range(min(8, pred.shape[1])):
    ok = (pred[0, di] == ref).astype(float)
    ax.plot(np.cumsum(ok) / np.arange(1, len(ok) + 1), lw=0.9, alpha=0.8)
ax.set_xlabel("generated token $t$"); ax.set_ylabel("running fraction of tokens identical")
ax.set_title(f"(a) Latent perturbation at $s={int(steps[0])}$: agreement is a cumulative fraction\n"
             "(smooth by construction, not by physics)")
ax.grid(alpha=0.25)
ax = axs[1]
worst = None
for di in range(pred.shape[1]):
    ok = (pred[0, di] == ref).astype(float)
    if worst is None or ok.sum() < worst[1]:
        worst = (di, ok.sum(), ok)
ag = np.cumsum(worst[2]) / np.arange(1, len(worst[2]) + 1)
ax.plot(ag, lw=1.4, color="tab:red")
ax.set_xlabel("generated token $t$"); ax.set_ylabel("running fraction identical")
ax.set_title("(b) One direction: flat, then a step at the flip\n(no exponential decay)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "latent_agreement.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved 4 figures")
