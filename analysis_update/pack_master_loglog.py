import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
# (a) closed vs replay, log-log
ax = axes[0]
for T, c in [(0.1, "#1f77b4"), (1.0, "#d62728"), (2.0, "#9467bd")]:
    loop = [z for TT, z, m in soft_runs("DeepSeek-1.5B") if abs(TT - T) < 1e-9]
    rep = [z for TT, z, m in soft_runs("DeepSeek-1.5B", "replay") if abs(TT - T) < 1e-9]
    if loop:
        a = np.median(loop[0]["dh"][:, -1, :].astype(float), axis=0)
        ax.plot(np.arange(1, len(a)), a[1:] / a[1], color=c, label=f"$T$={T:g} closed")
    if rep:
        a = np.median(rep[0]["dh"][:, -1, :].astype(float), axis=0)
        ax.plot(np.arange(1, len(a)), a[1:] / a[1], color=c, ls="--",
                label=f"$T$={T:g} replay")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1, 300); ax.set_ylim(1e-3, 2e4)
ax.set_xlabel("step $t$  (log)")
ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$  (log)")
ax.set_title("(a) Closed loop vs replay, log-log", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(True, which="both", alpha=0.3)

# (b) sampling, log-log
ax = axes[1]
soft = [z for T, z, m in soft_runs("DeepSeek-1.5B") if abs(T - 1.0) < 1e-9]
if soft:
    a = np.median(soft[0]["dh"][:, -1, :].astype(float), axis=0)
    ax.plot(np.arange(1, len(a)), a[1:] / a[1], color="#d62728", label="soft")
for mode, c, lab in [("crn", "#1f77b4", "CRN"), ("ind", "#2ca02c", "independent")]:
    p = os.path.join(HERE, f"results/softloop_sample/ds15/{mode}_T1.0/softloop.npz")
    if os.path.exists(p):
        z = dict(np.load(p, allow_pickle=True))
        a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
        ax.plot(np.arange(1, len(a)), a[1:] / a[1], c, ls="--", label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1, 300); ax.set_ylim(1e-3, 1e3)
ax.set_xlabel("step $t$  (log)")
ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$  (log)")
ax.set_title("(b) Feedback rule, log-log", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_master_loglog.png")
print("done")
