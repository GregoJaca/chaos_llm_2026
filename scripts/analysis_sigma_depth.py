"""sigma_1(l) instead of lambda_1(l): the actual gain, plus the state-norm form test."""
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
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})
NUM = {}

z = np.load(os.path.join(ROOT, "results", "depth2", "deepseek128", "depth.npz")) \
    if os.path.exists(os.path.join(ROOT, "results", "depth2", "deepseek128", "depth.npz")) \
    else np.load(os.path.join(ROOT, "results", "depth2", "deepseek", "depth.npz"))
lam, L, tok = z["lam"], z["layers"], z["tokens"]
nt = lam.shape[0]
sig = np.exp(np.median(lam[:, :, 0], axis=0) * (L + 1))     # geometric median of sigma_1
sig_lo = np.exp(np.percentile(lam[:, :, 0] * (L + 1), 25, axis=0))
sig_hi = np.exp(np.percentile(lam[:, :, 0] * (L + 1), 75, axis=0))
NUM["n_tokens"] = int(nt)

fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axs[0]
ax.plot(L, sig, "o-", ms=3, color="tab:blue", label=f"median over {nt} tokens")
ax.fill_between(L, sig_lo, sig_hi, color="tab:blue", alpha=0.18, label="25–75%")
ax.set_xlabel(r"block index $\ell$")
ax.set_ylabel(r"$\sigma_1(\ell)$ = stretch of the best direction")
ax.set_title("(a) The gain itself: flat, not decaying")
ax.legend(); ax.grid(alpha=0.25)

# zoom: is there structure?
ax = axs[1]
ax.plot(L, sig / sig[0], "o-", ms=3, color="tab:blue")
ax.axhline(1, color="k", ls=":", lw=0.9)
ax.set_xlabel(r"block index $\ell$")
ax.set_ylabel(r"$\sigma_1(\ell)/\sigma_1(0)$")
ax.set_title("(b) Normalised: the shape is a dip and a late rise")
ax.grid(alpha=0.25)

# fits of sigma_1(l)
r = fit_curve(L, sig, families=["linear", "quadratic", "power", "power+offset",
                                "exp+offset", "stretched_rise", "stretched_decay",
                                "saturating", "hill"])
ax = axs[2]
ax.plot(L, sig, "o", ms=4, color="0.3", label="data")
xx = np.linspace(0, 27, 300)
ax.plot(xx, r.best.predict(xx), "-", lw=1.8, color="tab:red",
        label=f"best: {r.best.name}  R2={r.best.r2:.4f}")
ax.plot(xx, r.fits[1].predict(xx), "--", lw=1.2, color="tab:green",
        label=f"2nd: {r.fits[1].name}  dBIC={r.fits[1].bic-r.best.bic:.1f}")
ax.set_xlabel(r"block index $\ell$"); ax.set_ylabel(r"$\sigma_1(\ell)$")
ax.set_title("(c) Fitted form of the gain")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sigma1_vs_depth.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
NUM["sigma1_best"] = r.best.name
NUM["sigma1_params"] = dict(zip(r.best.pnames, r.best.params.tolist()))
NUM["sigma1_table"] = [dict(name=f.name, r2=f.r2, dbic=float(f.bic - r.best.bic))
                       for f in r.fits[:6]]

# ---- per-block singular values (where does the gain happen?)
if "block_sv" in z:
    bsv = z["block_sv"]                 # [tokens, layers, k]
    med = np.median(bsv, axis=0)        # [layers, k]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axs[0]
    ax.semilogy(L, med[:, 0], "o-", ms=3, color="tab:blue", label=r"$\sigma_1$ of the block")
    ax.semilogy(L, med[:, 1], "s-", ms=2.5, color="tab:orange", label=r"$\sigma_2$")
    ax.semilogy(L, med[:, 3], "^-", ms=2.5, color="tab:green", label=r"$\sigma_4$")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"block $\ell$"); ax.set_ylabel("singular value of one block's Jacobian")
    ax.set_title("(a) Per-block gain: one big block, then ~1")
    ax.legend(); ax.grid(alpha=0.25, which="both")
    ax = axs[1]
    ax.plot(L, med[:, 0] / med[:, -1], "o-", ms=3, color="tab:purple")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"block $\ell$"); ax.set_ylabel(r"$\sigma_1/\sigma_8$ within the block")
    ax.set_title("(b) Is a single block anisotropic?")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "block_spectrum.png"), dpi=185, bbox_inches="tight")
    plt.close(fig)
    NUM["block_sv_med"] = med[:, 0].tolist()

# ---- state-norm form test
res = {}
for name, path, r_, c in [("DeepSeek-1.5B", os.path.join(ROOT, "results", "local_data", "big",
                                                         "all_layers", "a1.npz"), 0.1, "tab:blue"),
                          ("Qwen3-4B", os.path.join(ROOT, "results", "local_data", "models",
                                                    "qwen3_4b", "a1.npz"), 0.1, "tab:orange")]:
    d = np.load(path, allow_pickle=True)
    Lm = np.asarray(d["layers"]); h = d["norm_h0"].astype(float)
    for t in (10, 50):
        rr = fit_curve(Lm, h[np.arange(len(Lm)), t], families=["linear", "quadratic",
                                                              "power", "power+offset",
                                                              "exp+offset", "stretched_rise"])
        res[f"{name}_t{t}"] = dict(best=rr.best.name, r2=rr.best.r2,
                                   second=rr.fits[1].name,
                                   dbic=float(rr.fits[1].bic - rr.best.bic),
                                   params=dict(zip(rr.best.pnames, rr.best.params.tolist())))
json.dump(dict(sigma=NUM, norms=res), open(os.path.join(ROOT, "results", "sigma_depth.json"), "w"),
          indent=1)
print(json.dumps(dict(n_tokens=nt, sigma1=dict(best=NUM["sigma1_best"], table=NUM["sigma1_table"]),
                      norms=res), indent=1)[:1800])
