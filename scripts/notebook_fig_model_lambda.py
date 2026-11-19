"""Cross-model: response and finite-time rate."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
RUNS = [
    ("DeepSeek-1.5B", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz", 27),
    ("Qwen3-4B", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models\qwen3_4b\a1.npz", 32),
    ("Phi-4-mini", r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models\phi\a1.npz", 28),
]

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for name, path, layer in RUNS:
    d = np.load(path, allow_pickle=True)
    L = list(d["layers"]); li = L.index(layer) if layer in L else len(L) - 1
    ri = int(np.argmin(np.abs(d["radii"] - 0.3)))
    a = np.maximum(np.median(d["dh_norm"][ri, :, li, :], axis=0), 1e-12)
    axs[0].semilogy(np.arange(len(a)), a, lw=1.2, label=name)
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(a[t + k] / a[t - k]) / (2 * k)
    axs[1].plot(t, lam, lw=1.1, label=name)
axs[0].set_xlabel(r"$t$")
axs[0].set_ylabel(r"$a(t,\ell)$  (absolute)")
axs[0].set_title("(a) Response at the last layer, $r=0.3$")
axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25, which="both")
axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_xlabel(r"$t$")
axs[1].set_ylabel(r"$\lambda_{10}(t)$ (nats/token)")
axs[1].set_title("(b) Phi has a positive transient;\nDeepSeek decays, Qwen3 is near zero")
axs[1].legend(fontsize=8); axs[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT + r"\model_lambda_compare.png", dpi=190)
print("saved")
