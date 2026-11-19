"""New analyses: per-layer gain, window first departure, margin coordinate, flux variants."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})


# ---------------------------------------------------------------- 1. per-layer gain
def per_layer_gain():
    d = np.load(os.path.join(LD, "big", "all_layers", "a1.npz"))
    A = d["dh_norm"].astype(np.float64)          # [r, dir, layer, T]
    layers = np.asarray(d["layers"])
    ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    a = np.median(A[ri], axis=0)                 # [layer, T]

    fig, axs = plt.subplots(1, 3, figsize=(17.5, 4.6))
    ax = axs[0]
    for t, c in [(5, "tab:blue"), (20, "tab:orange"), (50, "tab:green"),
                 (100, "tab:red"), (200, "tab:purple")]:
        num = a[1:, t]; den = a[:-1, t]
        g = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        ax.semilogy(layers[1:], g, "o-", ms=3, lw=1.2, color=c, label=f"$t={t}$")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"layer $\ell$  (gain from $\ell-1$ to $\ell$)")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,\|\delta h_{t,\ell-1}\|$")
    ax.set_title("(a) Layer-to-layer gain of the perturbation")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25, which="both")

    ax = axs[1]
    h = d["norm_h0"].astype(np.float64)
    for t, c in [(5, "tab:blue"), (20, "tab:orange"), (50, "tab:green"),
                 (100, "tab:red"), (200, "tab:purple")]:
        num = a[1:, t] / np.maximum(h[1:, t], 1e-30)
        den = a[:-1, t] / np.maximum(h[:-1, t], 1e-30)
        ex = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        ax.plot(layers[4:], ex[3:], "o-", ms=3, lw=1.2, color=c, label=f"$t={t}$")
    ax.axhline(1, color="k", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 1.7)
    ax.set_xlabel(r"layer $\ell$  (layers 4–27)")
    ax.set_ylabel(r"$\dfrac{\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|}{\|\delta h_{t,\ell-1}\|/\|h_{t,\ell-1}\|}$")
    ax.set_title("(b) Beyond layer 4: gain equals the state's own growth")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25, which="both")

    ax = axs[2]
    for t, c in [(5, "tab:blue"), (20, "tab:orange"), (50, "tab:green"),
                 (100, "tab:red"), (200, "tab:purple")]:
        q = a[:, t] / np.maximum(h[:, t], 1e-9)
        ax.semilogy(layers, np.maximum(q, 1e-12), "o-", ms=3, lw=1.2, color=c,
                    label=f"$t={t}$")
    ax.set_xlabel(r"layer $\ell$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,\|h_{t,\ell}\|$")
    ax.set_title("(c) Relative perturbation is flat in depth")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "perlayer_gain.png"), dpi=190)
    plt.close(fig)

    g_sum = a[1:].sum(axis=1) / np.maximum(a[:-1].sum(axis=1), 1e-30)
    print("cumulative layer gain L1->L27:", [round(float(x), 3) for x in g_sum[::3]])


# ---------------------------------------------------------------- 2. window first departure
def window_first_departure():
    base = None
    series = {}
    for w in [50, 100, 200, 400, 800, 4096]:
        p = os.path.join(LD, "window_sweep", f"w{w}", "a1.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
        a = d["dh_norm"][ri]                     # [dir, layer, T]
        series[w] = a
        if w == 4096:
            base = a
        layers = np.asarray(d["layers"])
        P = int(d["prompt_ids"].shape[0])
    assert base is not None

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axs[0]
    cols = {50: "tab:blue", 100: "tab:orange", 200: "tab:green", 400: "tab:red", 800: "tab:purple"}
    for w, a in series.items():
        if w == 4096:
            continue
        for li, l in enumerate(layers):
            diff = np.abs(a[:, li, :] - base[:, li, :])
            rel = diff / np.maximum(base[:, li, :].max(axis=1, keepdims=True), 1e-30)
            first = []
            for di in range(a.shape[0]):
                idx = np.where(rel[di] > 1e-3)[0]
                first.append(int(idx[0]) if len(idx) else np.nan)
            med = np.nanmedian(first)
            if np.isnan(med):
                continue
            ax.plot(w, med, "o", color=cols[w], ms=6,
                    mfc="white" if l != layers[-1] else cols[w])
            if l == layers[-1]:
                ax.annotate(f"$W={w}$", (w, med), textcoords="offset points",
                            xytext=(8, -4), fontsize=9)
    ws = np.array([w for w in series if w != 4096 and w - P <= 320])
    ax.plot(ws, ws - P, "k--", lw=1.2, label=r"$t=W-P$")
    ax.set_xlabel(r"window $W$")
    ax.set_ylabel(r"first token $t$ at which $W$ differs from $W=4096$")
    ax.set_title(f"(a) First departure: $(W-P)$, not $W$  (prompt length $P={P}$)")
    ax.plot([], [], "o", color="0.4", label="layers 7, 14, 21, 27")
    ax.plot([], [], "o", mfc="white", color="0.4", label="last layer (filled)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axs[1]
    for w, a in series.items():
        if w == 4096:
            continue
        li = len(layers) - 1
        diff = np.abs(a[:, li, :] - base[:, li, :])
        rel = diff / np.maximum(base[:, li, :].max(axis=1, keepdims=True), 1e-30)
        med = np.median(rel, axis=0)
        ax.semilogy(np.arange(len(med)), np.maximum(med, 1e-12), lw=1.2, color=cols[w],
                    label=f"$W={w}$ (departs at $t={w-P}$)")
        ax.axvline(w - P, color=cols[w], ls=":", lw=0.8)
    ax.set_xlim(0, 300)
    ax.set_xlabel(r"generated token $t$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}(W)-\delta h_{t,\ell}(4096)\|$ (relative)")
    ax.set_title(f"(b) Separation from the unwindowed run, layer {layers[-1]}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "window_first_departure.png"), dpi=190)
    plt.close(fig)
    print("first-departure table done")


# ---------------------------------------------------------------- 3. margin coordinate
def margin_coordinate():
    d = np.load(os.path.join(LD, "window_sweep", "w200", "a1.npz"))
    dm = d["dm"].astype(np.float64)              # [r, dir, T]
    m = d["margins"].astype(np.float64)          # [T]
    radii = d["radii"]
    u = dm / np.maximum(m, 1e-9)
    ri = len(radii) - 1

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    U = u[ri]
    tt = np.arange(U.shape[1])
    for di in range(min(10, U.shape[0])):
        ax.plot(tt[:150], U[di][:150], lw=0.9, alpha=0.8, marker=".", ms=2)
    ax.axhline(-1, color="r", lw=1.2, ls="--", label=r"flip threshold $u=-1$")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlim(0, 150)
    ax.set_ylim(-1.1, 0.4)
    ax.set_xlabel(r"generated token $t$ (first 150)")
    ax.set_ylabel(r"$u_t=\delta m_t/m_t$")
    ax.set_title(fr"(a) Margin coordinate, $r={radii[ri]}$, 10 directions")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    ax = axs[1]
    far = []
    for i, r in enumerate(radii):
        far.append(-np.min(u[i], axis=1))
    for r, f, c in zip(radii, far, ["tab:blue", "tab:orange", "tab:green"]):
        ax.scatter(np.full_like(f, r), f, s=18, color=c, alpha=0.8)
    rr = np.linspace(0, max(radii) * 1.9, 10)
    env = np.array([np.max(f) for f in far]) / np.array(radii)
    slope = float(np.median(env))
    ax.plot(rr, slope * rr, "k--", lw=1.0,
            label=fr"envelope fit, $r^*={1/slope:.2f}$")
    ax.axhline(1, color="r", ls="--", lw=1.1, label="flip")
    ax.set_xlim(0, 1.9 / slope * 1.15)
    ax.set_xlabel(r"perturbation radius $r$")
    ax.set_ylabel(r"$-\min_t u_t$  (closest approach to a flip)")
    ax.set_title("(b) The approach to a flip is linear in $r$")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axs[2]
    for r, c in zip(radii, ["tab:blue", "tab:orange", "tab:green"]):
        i = int(np.argmin(np.abs(radii - r)))
        x = -np.min(u[i], axis=1) / r
        ax.hist(x, bins=12, histtype="step", lw=1.4, color=c, label=f"$r={r}$")
    ax.set_xlabel(r"$-\min_t u_t\,/\,r$")
    ax.set_ylabel("directions")
    ax.set_title("(c) Closest approach per direction, divided by $r$")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "margin_coordinate.png"), dpi=190)
    plt.close(fig)
    print("margin coordinate: r* =", round(float(1 / slope), 3),
          "min u per r:", [round(float(u[i].min()), 3) for i in range(len(radii))])


# ---------------------------------------------------------------- 4. flux variants, all layers
def flux_variants():
    d = np.load(os.path.join(LD, "big", "all_layers", "a1.npz"))
    A = d["dh_norm"].astype(np.float64)
    layers = np.asarray(d["layers"])
    ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    a = np.maximum(np.median(A[ri], axis=0), 1e-12)          # [layer, T]
    T = a.shape[1]
    t = np.arange(T)
    cmap = plt.get_cmap("viridis")

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axs[0]
    for l in layers[1:]:
        ax.semilogy(t, a[l] / a[l, 1], lw=0.9, color=cmap(l / layers[-1]), alpha=0.9)
    ax.set_xlabel(r"generated token $t$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,\|\delta h_{1,\ell}\|$")
    ax.set_title(r"(a) Normalised by $t=1$, semilog")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, layers[-1]))
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label(r"layer $\ell$")
    ax.grid(alpha=0.25, which="both")

    ax = axs[1]
    for l in layers[1:]:
        r = a[l, 1:] / np.maximum(a[l, :-1], 1e-12)
        ax.semilogy(t[1:], r, lw=0.9, color=cmap(l / layers[-1]), alpha=0.9)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"generated token $t$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,\|\delta h_{t-1,\ell}\|$")
    ax.set_title(r"(b) Consecutive-step ratio (same colours)")
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label(r"layer $\ell$")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flux_variants_all_layers.png"), dpi=190)
    plt.close(fig)
    print("flux variants saved")


if __name__ == "__main__":
    per_layer_gain()
    window_first_departure()
    margin_coordinate()
    flux_variants()
