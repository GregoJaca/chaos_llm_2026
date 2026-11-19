"""All-directions run: 1536 perturbation directions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\all_dirs.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(B)
A = d["dh_norm"][0]             # [dirs, layers, T]
layers = list(d["layers"])
T = A.shape[-1]
g = np.median(A[:, :, :50], axis=2)   # [dirs, layers]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axs[0]
v = g[:, -1]
ax.hist(np.log10(np.maximum(v, 1e-6)), bins=60, color="tab:blue")
ax.set_xlabel(r"$\log_{10}$ gain at layer 27 (median over first 50 tokens)")
ax.set_ylabel("number of directions")
ax.set_title(f"(a) {len(v)} directions: median {np.median(v):.3f},\n"
             f"5–95% {np.percentile(v,5):.3f}–{np.percentile(v,95):.3f}")
ax.grid(alpha=0.25)

ax = axs[1]
for li, c in zip(range(1, len(layers)), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    ax.semilogy(np.arange(T), np.percentile(A[:, li, :], 50, axis=0), lw=1.2, color=c,
                label=f"layer {layers[li]}")
    lo = np.percentile(A[:, li, :], 10, axis=0)
    hi = np.percentile(A[:, li, :], 90, axis=0)
    ax.fill_between(np.arange(T), np.maximum(lo, 1e-8), np.maximum(hi, 1e-8), color=c, alpha=0.08)
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t,\ell)$: median and 10–90% over 1536 directions")
ax.set_title("(b) Response is highly direction-dependent,\nbut the temporal shape is shared")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axs[2]
w = g[:, 0][:, None] * 0 + 1
X = np.log(np.maximum(g, 1e-6))
X = X - X.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
ax.plot(np.arange(1, len(S) + 1), np.cumsum(S ** 2) / np.sum(S ** 2), "o-", ms=3)
ax.set_xlabel("number of modes")
ax.set_ylabel("cumulative variance of log-gain")
ax.set_title(f"(c) Effective dimension of the direction space:\n"             f"{np.sum(np.cumsum(S**2)/np.sum(S**2) < 0.9)+1} modes explain 90%")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUT + r"\all_directions.png", dpi=190)
print("saved; eff dim 90% =", int(np.sum(np.cumsum(S**2)/np.sum(S**2) < 0.9)+1))
