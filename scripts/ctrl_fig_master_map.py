"""Part 8: map the control results onto master.md's LLM results.

Every quantity uses master.md's convention:
  - late lambda: mean of lambda_10(t) over t in [150, 290], last layer
  - direction spread (5-95 pct): single-position kick at s=16, r=0.1, last layer
  - amplification at s=16: ||dh_{t=s+1}||/r? master reads the response at the
    perturbed position; we read dh/r at t=0..50 after the kick and report the
    immediate value and the +20 value (their "twenty tokens later")
  - relative size at t=50: ||dh(t=50)||/||h(t=50)||, prompt perturbation
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebook", "figures_controls")
MAM = os.path.join(ROOT, "results", "ctrl_local", "mamba130")
DS = os.path.join(ROOT, "results", "local_data_ds", "w100_layers", "a1.npz")

z = np.load(os.path.join(MAM, "a1i_txt.npz"))
s2 = np.load(os.path.join(MAM, "state_s16_50", "state2.npz"), allow_pickle=True)
zds = np.load(DS)

k = 10


def lam_late(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    a = np.median(zz["dh_norm"][ri, :, -1] / zz["radii"][ri], axis=0)
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    sel = (t >= 150) & (t <= 290)
    return float(lam[sel].mean())


# late lambda, master table + controls
master_lam = {"DeepSeek-1.5B": -0.0105, "Qwen2.5-3B": 0.0002, "Qwen3-4B": 0.0003,
              "Phi-4-mini": 0.0003, "Qwen3.5-2B": -0.0014}
ctrl_lam = {}
for tag in ["txt", "childhood", "cyber"]:
    ctrl_lam[f"Mamba-130m {tag}"] = lam_late(np.load(os.path.join(MAM, f"a1i_{tag}.npz")), 0.1)
z790 = np.load(os.path.join(ROOT, "results", "ctrl_local", "mscale", "mamba790_a1i_txt.npz"))
ctrl_lam["Mamba-790m txt"] = lam_late(z790, 0.1)
zpy = np.load(os.path.join(ROOT, "results", "ctrl_local", "pythia410", "a1i_wide.npz"), allow_pickle=True)
ctrl_lam["Pythia-410m txt"] = lam_late(zpy, 0.3)
ctrl_lam["Pythia-410m step0"] = lam_late(
    np.load(os.path.join(ROOT, "results", "ctrl_local", "pythia410", "step0_a1i.npz"), allow_pickle=True), 0.1)
ctrl_lam["randQwen g=1.0"] = lam_late(
    np.load(os.path.join(ROOT, "results", "ctrl_local", "randqwen", "randqwen_g1.0", "a1i_txt", "a1.npz")), 0.1)

fig, ax = plt.subplots(figsize=(10.5, 4.8))
names = list(master_lam) + list(ctrl_lam)
vals = list(master_lam.values()) + list(ctrl_lam.values())
cols = ["#D55E00"] * len(master_lam) + ["#0072B2"] * len(ctrl_lam)
ax.bar(range(len(names)), vals, 0.6, color=cols)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel(r"late $\lambda_{10}$ (mean over $t\in[150,290]$, nats/token)")
ax.set_title("Late stretching rate (master.md table + controls): no system grows -\nthe LLM values sit inside the range spanned by provably non-chaotic models")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_lambda_all.png"), dpi=190); plt.close(fig)
print("late lambda:", {**master_lam, **{k2: round(v2, 4) for k2, v2 in ctrl_lam.items()}})

# direction spread at s=16 (master: 5-95 pct over 24 dirs; ours: 8 dirs, same kick)
# NOTE: inject_layer=0 lands the kick at GENERATED position s; the response at the
# perturbed position is dh[..., t=s..s+2]; the pre-kick region is the noise floor.
dh = s2["dh"]  # [nr, ns, K, nl, T]
sp = []
amp_at_s = []
amp_plus20 = []
for si in range(dh.shape[1]):
    s = int(s2["steps"][si])
    a_s = dh[0, si, :, -1, s:s + 3] / 0.1  # dh/r at the perturbed position
    amp_at_s.append(float(np.median(a_s)))
    amp_plus20.append(float(np.median(dh[0, si, :, -1, s + 20]) / 0.1))
    aa = np.median(a_s, axis=1) if a_s.shape[1] > 1 else a_s[:, 0]
    sp.append(float(np.quantile(aa, 0.95) / max(np.quantile(aa, 0.05), 1e-12)))
mam_spread = float(np.mean(sp))
print("mamba s=16/50 amplification at the perturbed position:", [round(x, 2) for x in amp_at_s])
print("mamba s=16/50 response +20 tokens later:", [round(x, 2) for x in amp_plus20])
print("mamba s=16 direction spread (5-95):", round(mam_spread, 2),
      " individual:", [round(x, 2) for x in sp])

# amplification at s=16 (immediate and +20)
amp_imm = float(np.mean(amp_at_s))
amp_20 = float(np.mean(amp_plus20))
print(f"mamba amplification at s: {amp_imm:.2f}, +20 tokens {amp_20:.2f}")

# relative size at t=50 (prompt perturbation, r=0.1, last layer)
ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
rel = float(np.median(z["dh_norm"][ri, :, -1, 50] / z["radii"][ri]) /
            np.median(z["norm_h0"][-1][50]))
print("mamba relative size at t=50:", "%.2g" % rel)
ri = int(np.argmin(np.abs(zds["radii"] - 0.1)))
rel_ds = float(np.median(zds["dh_norm"][ri, :, -1, 50] / zds["radii"][ri]) /
               np.median(zds["norm_h0"][-1][50]))
print("deepseek relative size at t=50 (own data):", "%.2g" % rel_ds)

# figure: s=16 amplification across models + direction spread
master_amp = {"DeepSeek-1.5B": 0.76, "Qwen2.5-3B": 4.63, "Qwen3-4B": 5.84,
              "Phi-4-mini": 7.25, "Qwen3.5-2B": 0.48}
master_spread = {"DeepSeek-1.5B": 5.85, "Qwen2.5-3B": 1.36, "Qwen3-4B": 1.19,
                 "Phi-4-mini": 1.19, "Qwen3.5-2B": 1.93}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axs[0]
names = list(master_amp) + ["Mamba-130m"]
vals = list(master_amp.values()) + [amp_imm]
cols = ["#D55E00"] * len(master_amp) + ["#0072B2"]
ax.bar(range(len(names)), vals, 0.6, color=cols)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel(r"amplification at $s=16$ ($\|\delta h\|/r$ at the perturbed position)")
ax.set_title("Single-position kick response (master.md section 8 table + control):\nthe control is comparable to DeepSeek; amplification is not attention-specific")
ax.grid(alpha=0.3, axis="y")
ax = axs[1]
names = list(master_spread) + ["Mamba-130m"]
vals = list(master_spread.values()) + [mam_spread]
cols = ["#D55E00"] * len(master_spread) + ["#0072B2"]
ax.bar(range(len(names)), vals, 0.6, color=cols)
ax.set_yscale("log")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel(r"direction spread (5-95 pct) at $s=16$")
ax.set_title("Direction anisotropy (master.md section 8 + control): the no-attention\ncontrol is as anisotropic as DeepSeek - spread is not an attention property")
ax.grid(alpha=0.3, axis="y", which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_s16_spread.png"), dpi=190); plt.close(fig)

json.dump(dict(late_lambda={**master_lam, **ctrl_lam}, mamba_spread_s16=mam_spread,
               mamba_amp_s16=amp_at_s, mamba_amp_s16_plus20=amp_plus20,
               mamba_rel_size=rel, ds_rel_size=rel_ds),
          open(os.path.join(ROOT, "results", "ctrl_local", "master_map_numbers.json"), "w"), indent=1)
print("wrote", OUT)
