"""Cross-model sliding window with the explicit 4D mask: first departure and flush cascade."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "window4d")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9})

WS = [60, 100, 200, 4096]
COL = {60: "tab:blue", 100: "tab:orange", 200: "tab:green", 4096: "0.35"}


def load(model, w):
    d = np.load(os.path.join(B, model, f"w{w}", "a1.npz"))
    return d, np.median(d["dh_norm"][0], axis=0), np.asarray(d["layers"])


fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))

# (a,b) response curves at layer 1
for ax, model, name in [(axs[0], "qwen3_4b", "Qwen3-4B"),
                        (axs[1], "phi4mini", "Phi-4-mini")]:
    for w in WS:
        d, a, L = load(model, w)
        li = int(np.where(L == 1)[0][0])
        ax.semilogy(np.arange(a.shape[1]), np.maximum(a[li], 1e-9), lw=1.2,
                    color=COL[w], label=f"$W={w}$")
        nz = np.where(a[li] > 0)[0]
        if len(nz):
            ax.axvline(int(nz[-1]), color=COL[w], ls=":", lw=0.8)
    ax.set_xlim(0, 300)
    ax.set_xlabel(r"generated token $t$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell=1}\|\,/\,r$")
    ax.set_title(f"({'a' if model == 'qwen3_4b' else 'b'}) {name}, layer 1")
    ax.legend(title="window")
    ax.grid(alpha=0.25, which="both")

# (c) flush law collapse
ax = axs[2]
res = []
for model, mk in [("qwen3_4b", "o"), ("phi4mini", "s")]:
    for w in WS[:-1]:
        d, a, L = load(model, w)
        last = {}
        for li, l in enumerate(L):
            nz = np.where(a[li] > 0)[0]
            last[int(l)] = int(nz[-1]) if len(nz) else None
        t1 = last[int(L[0])]
        if t1 is None:
            continue
        for li, l in enumerate(L):
            if last[int(l)] is None or last[int(l)] >= a.shape[1] - 1:
                continue
            pred = t1 + li * (w - 1)
            ax.scatter(pred, last[int(l)], s=32, marker=mk, color=COL[w],
                       edgecolor="k", linewidth=0.4, zorder=3)
            res.append(last[int(l)] - pred)
ax.plot([0, 300], [0, 300], "k--", lw=1.0, label="identity")
ax.set_xlabel(r"prediction  $t_1(W)+(\ell-1)(W-1)$")
ax.set_ylabel(r"measured last non-zero $t$")
ax.set_title("(c) Flush cascade law, both models")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window4d_models.png"), dpi=190)
plt.close(fig)
res = np.array(res)
print("n points", len(res), "max |residual|", float(np.abs(res).max()), "mean", float(res.mean()))

# (d) W=60 normalised by the unwindowed run: cliff vs no cliff
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for model, name, c, lay in [("qwen3_4b", "Qwen3-4B", "tab:orange", 1),
                            ("phi4mini", "Phi-4-mini", "tab:green", 1),
                            ("qwen3_5", "Qwen3.5-2B", "tab:red", 1)]:
    d1, a1, L1 = load(model, 60)
    d2, a2, L2 = load(model, 4096)
    li = int(np.where(L1 == lay)[0][0])
    y = np.maximum(a1[li], 1e-9) / np.maximum(a2[li], 1e-9)
    ax.semilogy(np.arange(len(y)), np.maximum(y, 1e-9), lw=1.2, color=c, label=name)
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$\|\delta h_{t,\ell=1}(W=60)\|\,/\,\|\delta h_{t,\ell=1}(W=4096)\|$")
ax.set_title("A window either flushes exactly or does nothing:\nQwen3-4B and Phi cut off, the hybrid does not")
ax.legend()
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window4d_qwen35.png"), dpi=190)
print("saved")
