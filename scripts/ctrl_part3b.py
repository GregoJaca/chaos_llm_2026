"""Part 3b: (i) teacher-forcing as a flip simulator on the negative control,
(ii) direction-gain spread, Mamba-130m vs DeepSeek-1.5B.

(i) TF-exactness scorecard. For each radius r in {0.3, 1, 2, 3, 5} and each of
the 20 prompt-perturbation directions, one teacher-forced (TF) pass predicts a
per-step token sequence ('pred'); the actual free run emits 'ar'. A flip at
step t is pred != ref[t] (TF) or ar != ref[t] (free run). Categories per
direction-run:
    bit-identical  no flip in either (both sequences == ref for all 300 steps)
    exact          both flip; first-flip step AND token agree
    mismatched     both flip; first-flip step or token disagrees
    false alarm    TF flips, the free run never flips
    miss           free run flips, TF never flips
Agreement after the first flip is not counted (the free run has departed).

(ii) Direction gains. g_u = median_t ||dh_L(t,u)||/r at r=0.1, last measured
layer, t in [1,50]; spread = max_u g_u / min_u g_u. Time stability: Spearman
rank corr of per-direction medians, window [1,50] vs [150,299].

Figures -> notebook/figures_controls/{c3b_tfp.png, c3b_dirs.png}.
Numbers -> stdout (copied into results/ctrl_local/part3b_draft.md).
"""
import os

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFP = os.path.join(ROOT, "results", "ctrl_local", "mamba130_tfp")
MAM = os.path.join(ROOT, "results", "ctrl_local", "mamba130", "a1i_txt.npz")
DS = os.path.join(ROOT, "results", "local_data_ds", "w100_layers", "a1.npz")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)

C_MAM = "#0072B2"
C_DS = "#D55E00"
RADII_TFP = ["0.3", "1", "2", "3", "5"]

# ---------------------------------------------------------------- (i) scorecard
CATS = ["bit-identical", "exact flip", "mismatched flip", "false alarm", "miss"]
counts = {R: dict(zip(CATS, [0] * 5)) for R in RADII_TFP}
flips_at_r5 = []

for R in RADII_TFP:
    z = np.load(os.path.join(TFP, f"tfp_r{R}.npz"))
    pred, ar, ref = z["pred"][0], z["ar"][0], z["ref"].astype(np.int64)
    assert not (pred == -1).any(), "unset TF predictions"
    for k in range(pred.shape[0]):
        arok = ar[k] >= 0  # free run emitted this step (-1 = unset)
        tf = np.where(pred[k] != ref)[0]
        ob = np.where((ar[k] != ref) & arok)[0]
        if len(tf) == 0 and len(ob) == 0:
            counts[R]["bit-identical"] += 1
        elif len(tf) == 0 and len(ob) > 0:
            counts[R]["miss"] += 1
        elif len(tf) > 0 and len(ob) == 0:
            counts[R]["false alarm"] += 1
        elif tf[0] == ob[0] and pred[k, tf[0]] == ar[k, ob[0]]:
            counts[R]["exact flip"] += 1
            if R == "5":
                flips_at_r5.append((k, int(ob[0]), int(ar[k, ob[0]])))
        else:
            counts[R]["mismatched flip"] += 1

print("== TF-exactness scorecard, Mamba-130m, 20 directions x 5 radii ==")
for R in RADII_TFP:
    c = counts[R]
    print(f"r={R:>4}: " + ", ".join(f"{k}={v}" for k, v in c.items()))
tot = {k: sum(counts[R][k] for R in RADII_TFP) for k in CATS}
print("total:", tot)
print("r=5 exact flips (direction, step, token):", flips_at_r5)

# ------------------------------------------------------------- (ii) gain spread
def gains(path, r0):
    z = np.load(path)
    ri = int(np.argmin(np.abs(z["radii"] - r0)))
    dh = z["dh_norm"][ri, :, -1, :] / z["radii"][ri]  # [20, T], last layer
    g_early = np.median(dh[:, 1:51], axis=1)
    g_late = np.median(dh[:, 150:300], axis=1)
    rho = spearmanr(g_early, g_late).statistic
    return g_early, g_late, rho, float(z["radii"][ri]), int(z["layers"][-1])

g_m, gl_m, rho_m, r_m, L_m = gains(MAM, 0.1)
g_d, gl_d, rho_d, r_d, L_d = gains(DS, 0.1)

# full-horizon spread, for comparison with the established x7-12 figure
def spread_full(path, r0):
    z = np.load(path)
    ri = int(np.argmin(np.abs(z["radii"] - r0)))
    dh = z["dh_norm"][ri, :, -1, :] / z["radii"][ri]
    g = np.median(dh[:, 1:300], axis=1)
    return g.max() / g.min()

print("\n== direction gains, r=0.1, last layer, median over t in [1,50] ==")
print(f"Mamba-130m  (layer {L_m}): min {g_m.min():.3g}  max {g_m.max():.3g}  "
      f"spread {g_m.max()/g_m.min():.2f}x   rho(early,late) {rho_m:.3f}  "
      f"full-horizon spread {spread_full(MAM, 0.1):.2f}x")
print(f"DeepSeek-1.5B (layer {L_d}): min {g_d.min():.3g}  max {g_d.max():.3g}  "
      f"spread {g_d.max()/g_d.min():.2f}x   rho(early,late) {rho_d:.3f}  "
      f"full-horizon spread {spread_full(DS, 0.1):.2f}x")

# ------------------------------------------------------------------- figures
# c3b_tfp.png : stacked-bar scorecard per radius
fig, ax = plt.subplots(figsize=(7.6, 4.8))
labels = [f"r={R}" for R in RADII_TFP]
colors = ["#999999", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
bottom = np.zeros(len(RADII_TFP))
for cat, col in zip(CATS, colors):
    vals = np.array([counts[R][cat] for R in RADII_TFP], dtype=float)
    ax.bar(labels, vals, bottom=bottom, color=col, label=cat, width=0.62,
           edgecolor="white", lw=0.6)
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, bottom[i] + v / 2, f"{int(v)}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bottom += vals
ax.set_ylabel("direction-runs (20 per radius)")
ax.set_xlabel("perturbation radius $r$ (norm-$r$ kick to the prompt embedding)")
ax.set_ylim(0, 21)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax.set_title(
    "One teacher-forced pass is an EXACT flip simulator for the control too:\n"
    "3/3 flips exact (step and token), 0 false alarms, 0 misses, "
    "97/100 direction-runs bit-identical",
    fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "c3b_tfp.png"), dpi=190)
plt.close(fig)

# c3b_dirs.png : sorted per-direction gains, both models
fig, ax = plt.subplots(figsize=(7.6, 4.8))
for g, col, name, sp, rho in [
        (g_m, C_MAM, "Mamba-130m (no attention)", g_m.max() / g_m.min(), rho_m),
        (g_d, C_DS, "DeepSeek-1.5B (transformer)", g_d.max() / g_d.min(), rho_d)]:
    ax.plot(np.arange(1, len(g) + 1), np.sort(g), "o-", color=col, ms=4,
            label=(f"{name}: spread {sp:.1f}$\\times$, "
                   f"Spearman(early,late) $\\rho$={rho:.2f}"))
ax.set_yscale("log")
ax.set_xlabel("direction rank (per-direction gains $g_u$, sorted)")
ax.set_ylabel(
    r"$g_u = \mathrm{median}_{t\in[1,50]}\,\|\delta h_L(t,u)\|/r$   "
    "(last layer, $r$=0.1)")
ax.legend(fontsize=9, loc="upper left")
ax.set_title(
    "Both architectures have strongly anisotropic perturbation directions:\n"
    "gain spread Mamba $6.5\\times$ vs DeepSeek $21\\times$ (early window) — "
    "anisotropy alone is not an attention signature",
    fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "c3b_dirs.png"), dpi=190)
plt.close(fig)
print(f"\nfigures written to {OUT}")
