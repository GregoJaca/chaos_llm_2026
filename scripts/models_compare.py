"""Cross-model comparison figures. Run after the model suites finish."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/models"
OUT = "results/figures_models"
os.makedirs(OUT, exist_ok=True)


def load_run(d):
    f = os.path.join(d, "a1.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    return z, meta


runs = {}
for d in sorted(glob.glob(os.path.join(ROOT, "*", "a1"))):
    tag = os.path.basename(os.path.dirname(d))
    r = load_run(d)
    if r:
        runs[tag] = r
print("models:", list(runs))

if runs:
    # 1. gain vs relative layer
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for tag, (z, meta) in runs.items():
        L = list(z["layers"]); radii = z["radii"]
        ri = int(np.argmin(np.abs(radii - float(meta["radii"][len(meta["radii"]) // 2]))))
        e = 10
        g = np.median(z["dh_norm"][ri, :, :, e] / radii[ri], axis=0)
        ax.semilogy(np.array(L) / max(L), g, "o-", ms=3, label=tag)
    ax.set_xlabel("layer / number of layers")
    ax.set_ylabel(r"median $a(t=10,\ell)/r$")
    ax.set_title("Response gain vs relative depth, all models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_gain_layers.png"), dpi=190); plt.close(fig)

    # 2. lambda(t) at the middle saved layer
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for tag, (z, meta) in runs.items():
        radii = z["radii"]; ri = len(radii) // 2
        li = z["dh_norm"].shape[2] // 2
        a = np.median(z["dh_norm"][ri, :, li, :], axis=0)
        k = 10
        t = np.arange(k, len(a) - k)
        lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
        ax.plot(t, lam, lw=1.1, label=tag)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"generated token $t$")
    ax.set_ylabel(r"$\lambda_{10}(t)$ (nats/token)")
    ax.set_title("Local stretching rate across models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_lambda.png"), dpi=190); plt.close(fig)

    # 3. normalised response (flux-tube style), two normalisations
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for tag, (z, meta) in runs.items():
        radii = z["radii"]; ri = len(radii) // 2
        li = z["dh_norm"].shape[2] // 2
        a = np.maximum(np.median(z["dh_norm"][ri, :, li, :], axis=0), 1e-12)
        axs[0].semilogy(np.arange(len(a)), a / a[1], lw=1.1, label=tag)
        axs[1].plot(np.arange(1, len(a)), a[1:] / np.maximum(a[:-1], 1e-12), lw=0.9, alpha=0.8, label=tag)
    axs[0].set_xlabel(r"$t$"); axs[0].set_ylabel(r"$a(t)/a(1)$")
    axs[0].set_title("(a) Normalised by $a(1)$"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
    axs[1].set_xlabel(r"$t$"); axs[1].set_ylabel(r"$a(t)/a(t-1)$")
    axs[1].set_title("(b) Step-to-step ratio"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_flux.png"), dpi=190); plt.close(fig)

    # 4. direction-gain distribution
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for tag, (z, meta) in runs.items():
        radii = z["radii"]; ri = len(radii) // 2
        li = z["dh_norm"].shape[2] - 1
        g = np.median(z["dh_norm"][ri, :, li, :50] / radii[ri], axis=1)
        g = g[g > 0]
        ax.hist(np.log10(g), bins=25, alpha=0.5, label=f"{tag}")
    ax.set_xlabel(r"$\log_{10}$ gain (last layer)")
    ax.set_ylabel("directions")
    ax.set_title("Direction gain distribution per model")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_direction_hist.png"), dpi=190); plt.close(fig)
    print("wrote figures to", OUT)
