"""Window flush: predictive law t*_1 = p_last + (W-P), t*_l = t*_1 + (l-1)(W-1)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def first_zero(a):
    z = np.where(a <= 0)[0]
    return int(z[0]) if len(z) else -1


def run(folder, w):
    path = (os.path.join(B, folder, "a1.npz") if folder == "w100_layers"
            else os.path.join(B, folder, f"w{w}", "a1.npz"))
    d = np.load(path, allow_pickle=True)
    L = list(d["layers"]); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    P = int(d["prompt_ids"].shape[0]); D = 1536
    pmax = int((d["subspace"] // D).max())
    ts = {}
    for l in L:
        A = d["dh_norm"][ri, :, L.index(l), :]
        vals = [first_zero(A[di]) for di in range(A.shape[0])]
        vals = [v for v in vals if v >= 0]
        ts[l] = int(np.median(vals)) if vals else -1
    return ts, P, pmax, d, ri, L


WINS = [100, 150, 200, 300, 500]
res = {}
for w in WINS:
    if os.path.exists(os.path.join(B, "flush_sweep", f"w{w}", "a1.npz")):
        res[w] = run("flush_sweep", w)
# 20-direction reference at W=100
res20 = run("w100_layers", 100)

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5))

ax = axs[0]
xs, ys = [], []
for w, (ts, P, pmax, *_ ) in res.items():
    xs.append(pmax + (w - P))          # prediction
    ys.append(ts[1])                   # measured
xs.append(res20[2] + (100 - res20[1])); ys.append(res20[0][1])
ax.plot([70, 500], [70, 500], "k--", lw=1, label="prediction $p_{last}+(W-P)$")
ax.scatter(xs, ys, s=70, color="tab:blue", zorder=3, label="measured")
ax.set_xlabel(r"predicted $t^*_1$")
ax.set_ylabel(r"measured $t^*_1$")
ax.set_title("(a) The death time is predicted by the\nlast perturbed prompt position")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axs[1]
for w, (ts, P, pmax, *_ ) in res.items():
    ls = [l for l in sorted(ts) if ts[l] >= 0]
    x = [l - 1 for l in ls]
    y = [(ts[l] - ts[1]) / (w - 1) for l in ls]
    ax.plot(x, y, "o-", label=f"$W={w}$")
ax.plot([0, 4], [0, 4], "k--", lw=1, label=r"$\ell-1$")
ax.set_xlabel(r"layer index $\ell-1$")
ax.set_ylabel(r"$(t^*_\ell-t^*_1)/(W-1)$")
ax.set_title("(b) Each deeper layer dies exactly\none window length later")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

ax = axs[2]
ts, P, pmax, d, ri, L = res20
for l, c in zip([1, 2, 3], ["tab:red", "tab:orange", "tab:green"]):
    a = np.median(d["dh_norm"][ri, :, L.index(l), :], axis=0)
    ax.semilogy(np.arange(len(a)), np.maximum(a, 1e-9), lw=1.3, color=c, label=fr"$\ell={l}$")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t,\ell)$")
ax.set_title("(c) $W=100$, 20 directions:\nthree sharp deaths, 99 tokens apart")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_flush_scaling.png"), dpi=190)
print("saved")
for w, (ts, P, pmax, *_ ) in res.items():
    print(f"W={w}: pmax={pmax}, predicted={pmax + w - P}, measured={ts[1]}, "
          f"layers={ {l: ts[l] for l in sorted(ts)} }")
print("20 dirs: pmax", res20[2], "predicted", res20[2] + 100 - res20[1], "measured", res20[0])
