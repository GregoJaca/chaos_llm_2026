"""Window spectroscopy: isolate the contribution of each prompt position."""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "wspec")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9})


def main():
    if not os.path.isdir(B):
        print("no local wspec data yet:", B)
        return
    ws = sorted(int(re.search(r"w(\d+)$", d).group(1))
                for d in os.listdir(B) if re.match(r"w\d+$", d))
    ws = [w for w in ws if os.path.exists(os.path.join(B, f"w{w}", "a1.npz"))]
    print("windows", len(ws), ws[:3], "...", ws[-3:])
    if len(ws) < 10:
        return

    ref = np.load(os.path.join(B, f"w{max(ws)}", "a1.npz"))
    P = int(ref["prompt_ids"].shape[0])
    layers = list(ref["layers"])
    T = ref["dh_norm"].shape[-1]
    A = {}
    for w in ws:
        d = np.load(os.path.join(B, f"w{w}", "a1.npz"))
        A[w] = np.median(d["dh_norm"][0], axis=0)      # [layer, T]
    print("P", P, "layers", layers, "T", T)

    # (a) response vs W at fixed t (layer 7)
    li = layers.index(7) if 7 in layers else 0
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.7))
    ax = axs[0]
    for t, c in [(20, "tab:blue"), (60, "tab:orange"), (100, "tab:green"), (150, "tab:red")]:
        y = [A[w][li, t] for w in ws]
        ax.semilogy(ws, np.maximum(y, 1e-9), "o-", ms=2.5, lw=1.0, color=c, label=f"$t={t}$")
    ax.set_xlabel(r"window $W$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell=7}\|\,/\,r$")
    ax.set_title("(a) Response vs window at fixed $t$")
    ax.legend()
    ax.grid(alpha=0.25, which="both")

    # (b) contribution map: -d a / dW  ->  prompt position p = P + t - W
    ax = axs[1]
    dA = np.full((len(ws), T), np.nan)
    for i, w in enumerate(ws):
        j = ws.index(w + 1) if (w + 1) in A else None
        if j is None:
            continue
        dA[i] = np.abs(A[w][li] - A[ws[j]][li])
    # map each (w,t) to p = P + t - w
    pmap = np.full((len(ws), T), np.nan)
    for i, w in enumerate(ws):
        pmap[i] = P + np.arange(T) - w
    m = (pmap >= 0) & (pmap < P) & np.isfinite(dA)
    x = pmap[m]; y = np.arange(T)[None, :].repeat(len(ws), 0)[m]; z = dA[m]
    sc = ax.scatter(x, y, c=np.log10(np.maximum(z, 1e-9)), s=6, cmap="inferno")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label(r"$\log_{10}|\Delta\|\delta h\||$ for $\Delta W=1$")
    ax.set_xlabel(r"prompt position $p$")
    ax.set_ylabel(r"generated token $t$")
    ax.set_title(r"(b) Contribution map: $p=P+t-W$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wspec_map.png"), dpi=190)
    plt.close(fig)
    print("saved wspec_map")


if __name__ == "__main__":
    main()
