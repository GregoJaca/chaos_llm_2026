"""Six ways to look at the superposition test."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "superposition")
OUT = os.path.join(ROOT, "notebook", "figures", "transport")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})


def load(d, layer="last"):
    z = np.load(os.path.join(d, "superposition.npz"))
    L = list(z["layers"])
    l = L[-1]
    return dict(r=float(z["r"]), T=int(z["T"]), nA=z[f"nA_L{l}"], nS=z[f"nS_L{l}"],
                err=z[f"err_L{l}"], cos=z[f"cos_L{l}"])


ds = {}
for d in sorted(glob.glob(os.path.join(B, "deepseek_r*"))) + [os.path.join(B, "deepseek")]:
    if not os.path.isdir(d):
        continue
    x = load(d)
    ds[x["r"]] = x
rs = sorted(ds)
print("radii", rs)

fig, axs = plt.subplots(2, 3, figsize=(16.5, 8.6))

# (a) scatter ||S|| vs ||A||
ax = axs[0, 0]
for r in rs:
    x = ds[r]
    ax.loglog(x["nA"].ravel(), x["nS"].ravel(), ".", ms=2, alpha=0.4, label=f"$r={r}$")
lo, hi = 1e-4, 1e2
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
ax.set_xlabel(r"$\|A\|$: response to all positions at once")
ax.set_ylabel(r"$\|S\|$: sum of the single-position responses")
ax.set_title("(a) Does the sum predict the joint response?")
ax.legend(fontsize=6.5, markerscale=3); ax.grid(alpha=0.25, which="both")

# (b) ratio ||S||/||A||
ax = axs[0, 1]
for r in rs:
    x = ds[r]
    ratio = (x["nS"] / np.maximum(x["nA"], 1e-12)).ravel()
    ax.hist(np.log10(np.maximum(ratio, 1e-6)), bins=60, histtype="step", lw=1.2,
            label=f"$r={r}$")
ax.axvline(0, color="k", ls=":", lw=0.9)
ax.set_xlabel(r"$\log_{10}\|S\|/\|A\|$"); ax.set_ylabel("tokens × trials")
ax.set_title("(b) The ratio is 1 only in the linear regime")
ax.legend(fontsize=7); ax.grid(alpha=0.25)

# (c) cosine
ax = axs[0, 2]
for r in rs:
    x = ds[r]
    ax.hist(x["cos"].ravel(), bins=50, histtype="step", lw=1.2, label=f"$r={r}$")
ax.set_xlabel(r"$\cos(A,S)$"); ax.set_ylabel("tokens × trials")
ax.set_title("(c) Direction agreement")
ax.legend(fontsize=7); ax.grid(alpha=0.25)

# (d) error relative to the state norm
a1 = np.load(os.path.join(ROOT, "results", "local_data", "big", "all_layers", "a1.npz"))
hL = a1["norm_h0"][-1]
ax = axs[1, 0]
for r in rs:
    x = ds[r]
    d = x["err"] * x["nA"] / np.maximum(hL[None, :], 1e-9)
    ax.plot(np.arange(x["T"]), np.median(d, axis=0), lw=1.1, label=f"$r={r}$")
ax.set_yscale("log")
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|A-S\|/\|h_t\|$")
ax.set_title("(d) Error relative to the state norm")
ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")

# (e) prediction gain cos*||S||/||A||
ax = axs[1, 1]
for r in rs:
    x = ds[r]
    g = x["cos"] * x["nS"] / np.maximum(x["nA"], 1e-12)
    ax.plot(np.arange(x["T"]), np.median(g, axis=0), lw=1.1, label=f"$r={r}$")
ax.axhline(1, color="k", ls=":", lw=0.9)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\cos(A,S)\,\|S\|/\|A\|$")
ax.set_title("(e) Gain of the linear prediction (1 = exact)")
ax.legend(fontsize=7); ax.grid(alpha=0.25)

# (f) error vs amplitude
ax = axs[1, 2]
for r in rs:
    x = ds[r]
    a = x["nA"].ravel(); e = x["err"].ravel()
    bins = np.geomspace(max(a.min(), 1e-6), a.max(), 12)
    idx = np.digitize(a, bins) - 1
    med = [np.median(e[idx == i]) if np.any(idx == i) else np.nan for i in range(len(bins) - 1)]
    ax.semilogx(0.5 * (bins[1:] + bins[:-1]), med, "o-", ms=4, lw=1.1, label=f"$r={r}$")
ax.set_xlabel(r"$\|A\|$ (amplitude of the joint response)")
ax.set_ylabel(r"$\|A-S\|/\|A\|$ (median)")
ax.set_title("(f) Where linearity holds")
ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "superposition_six_views.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# model comparison at r=0.03
fig, ax = plt.subplots(figsize=(8, 4.6))
names = {"deepseek": "DeepSeek", "qwen3_4b": "Qwen3-4B", "qwen25_3b": "Qwen2.5-3B",
         "phi4mini": "Phi-4-mini", "qwen3_5": "Qwen3.5-2B"}
res = {}
for tag, name in names.items():
    d = os.path.join(B, f"{tag}_r003")
    if not os.path.isdir(d):
        continue
    x = load(d)
    ax.plot(np.arange(x["T"]), np.median(x["err"], axis=0), lw=1.2, label=name)
    res[tag] = float(np.median(x["err"]))
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|A-S\|/\|A\|$ (median over trials)")
ax.set_title("Superposition error at $r=0.03$ per position (total norm 0.24)")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "superposition_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "results", "superposition_models.json"), "w"), indent=1)
