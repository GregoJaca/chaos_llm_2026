"""Part 7: attention vs no-attention at matched training data (Pile),
plus the Pythia step0 random-init checkpoint (ordered phase, standard scale)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "results", "ctrl_local", "pythia410")
MAM = os.path.join(ROOT, "results", "ctrl_local", "mamba130")
DS = os.path.join(ROOT, "results", "local_data_ds", "w100_layers", "a1.npz")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)

C_PY = "#009E73"
C_MAM = "#0072B2"
C_DS = "#D55E00"
C_S0 = "#CC79A7"


def signed_rc(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    m = zz["margins"][None, :]
    neg = np.maximum(-zz["dm"][ri], 0.0)
    rc = np.where(neg > 0, m * float(zz["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc, float(zz["radii"][ri])


z410 = np.load(os.path.join(PY, "a1i_wide.npz"), allow_pickle=True)
z1b = np.load(os.path.join(PY, "pythia1b_a1i.npz"), allow_pickle=True)
zs0 = np.load(os.path.join(PY, "step0_a1i.npz"), allow_pickle=True)
zm = np.load(os.path.join(MAM, "a1i_txt.npz"))
zd = np.load(DS)

# ---- response curves, all models ----
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
for z, r, c, lab in [(zm, 0.1, C_MAM, "Mamba-130m (SSM)"),
                     (z410, 0.3, C_PY, "Pythia-410m (attn)"),
                     (z1b, 0.1, C_PY, "Pythia-1b (attn)"),
                     (zd, 0.1, C_DS, "DeepSeek-1.5B (attn)")]:
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    ax.plot(np.arange(len(a)), a, lw=1.3, color=c, label=lab)
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t)/r$ (last layer, median)")
ax.set_title("(a) Pile-trained, same prompt, same protocol: the small attention\nmodel decays like the control; DeepSeek's persistence is not just attention")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

ax = axs[1]
ri = int(np.argmin(np.abs(zs0["radii"] - 0.1)))
a = np.median(zs0["dh_norm"][ri, :, -1] / zs0["radii"][ri], axis=0)
ax.plot(np.arange(len(a)), a, lw=1.3, color=C_S0, label="Pythia-410m step0 (random init)")
ri = int(np.argmin(np.abs(z410["radii"] - 0.3)))
a = np.median(z410["dh_norm"][ri, :, -1] / z410["radii"][ri], axis=0)
ax.plot(np.arange(len(a)), a, lw=1.1, color=C_PY, alpha=0.6, label="Pythia-410m trained")
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t)/r$")
ax.set_title("(b) The random-init transformer at standard scale is in the\nORDERED phase: response decays, no growth phase")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c7_pythia_resp.png"), dpi=190); plt.close(fig)

# ---- obs/pred scorecard for pythia wide ----
fig, ax = plt.subplots(figsize=(9.5, 4.6))
rows = []
for R, tag in [(0.19, "frw_r019"), (0.3, "frw_r03"), (0.7, "frw_r07"), (1.4, "frw_r14"), (3, "frw_r3")]:
    fr = np.load(os.path.join(PY, tag + ".npz"), allow_pickle=True)
    obs = fr["obs"][0]
    rc, rmeas = signed_rc(z410, R)
    hit = rc <= R
    pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    both = (obs >= 0) & (pred_T >= 0)
    rows.append(dict(r=R, noflip=int((obs < 0).sum()), exact=int((obs[both] == pred_T[both]).sum()),
                     miss=int(((pred_T < 0) & (obs >= 0)).sum()),
                     pred=int((pred_T >= 0).sum()), obs=int((obs >= 0).sum())))
xs = np.arange(len(rows))
w = 0.28
ax.bar(xs - w, [r["noflip"] for r in rows], w, color="#bbb", label="no flip (bit-identical)")
ax.bar(xs, [r["exact"] for r in rows], w, color=C_PY, label="flip predicted exactly (step+token)")
ax.bar(xs + w, [r["miss"] for r in rows], w, color="#f4a582", label="miss (flipped unpredicted)")
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['r']:g}" for r in rows])
ax.set_xlabel("free-run radius $r$")
ax.set_ylabel("direction-runs (out of 20)")
ax.set_title("Pythia-410m free runs vs the signed threshold rule (measured at each $r$):\n19/19 exact among predicted, 0 false alarms, 3 misses at $r\\geq1.4$")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c7_pythia_obspred.png"), dpi=190); plt.close(fig)

# ---- r* comparison ----
fig, ax = plt.subplots(figsize=(8.6, 4.4))
names, rstars = [], []
for z, r, name in [(zs0, 0.1, "pythia step0"), (zm, 1.0, "mamba txt"), (z410, 0.19, "pythia410"),
                   (z1b, 0.1, "pythia1b"), (zd, 0.1, "deepseek")]:
    rc, _ = signed_rc(z, r)
    names.append(name)
    rstars.append(float(np.min(rc)))
ax.bar(names, rstars, color=[C_S0, C_MAM, C_PY, C_PY, C_DS], alpha=0.8)
ax.set_yscale("log")
ax.set_ylabel(r"signed flip threshold $r^*$ (measured at sub-flip radius)")
ax.set_title("Per-model $r^*$ on the same prompt: governed by each model's margin\nnear-ties (random init: $2.7\\times10^{-3}$), not by architecture")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c7_rstar.png"), dpi=190); plt.close(fig)

for n, rs in zip(names, rstars):
    print(f"{n:12s} r* = {rs:.4g}")
print("pythia obs/pred rows:", rows)
