"""06: Larger-r behavior (limitation handled explicitly).

No current-state injection data at r != 1e-3 exists locally, so the
nonlinearity onset CANNOT be measured for the state protocol. What we do
have on the SAME baseline trajectory (w100, margins argmin t=123):
results/local_data/w100_layers/a1.npz -- a PROMPT-perturbation,
teacher-forced r-scan (12 radii 1e-3..2.0, 20 directions, 8 layers,
300 generated tokens). We use it as an indirect proxy: where does the
teacher-forced response at layer 27 stop being linear in r?

Protocol differences (state vs prompt injection) mean the numbers are a
proxy, not a measurement of the state protocol. r=1e-3 (used for state) sits
at the bottom of the linear range found here.

Figure: figures/fig06_r_scaling.png ; stats: r_scaling.json
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import REPO, save_json, savefig, log10safe

d = np.load(REPO + r"\results\local_data\w100_layers\a1.npz")
radii = d["radii"].astype(float)
layers = d["layers"]
dh = d["dh_norm"].astype(float)          # [12, 20, 8, 300]
li = list(layers).index(27)
med = np.median(dh[:, :, li, :], axis=1)  # [12, 300]
T = med.shape[1]

stats = dict(protocol="prompt perturbation, teacher-forced (proxy for state injection)",
             radii=radii.tolist(), layer=27, note="no state-injection data at r!=1e-3 exists locally")

# local exponent alpha(r) = d log dh / d log r between consecutive radii,
# summarized over t (median + IQR over t of per-t exponents)
rows = []
for i in range(len(radii) - 1):
    a = log10safe(med[i]); b = log10safe(med[i + 1])
    num = b - a
    den = np.log10(radii[i + 1] / radii[i])
    alpha_t = num / den
    rows.append(dict(r_lo=float(radii[i]), r_hi=float(radii[i + 1]),
                     alpha_median_over_t=float(np.median(alpha_t)),
                     alpha_iqr_over_t=[float(np.quantile(alpha_t, .25)), float(np.quantile(alpha_t, .75))],
                     frac_t_alpha_within_0p9_1p1=float(np.mean(np.abs(alpha_t - 1) < 0.1))))
stats["local_alpha"] = rows

# first r_hi where median alpha leaves [0.9, 1.1]
onset = None
for rw in rows:
    if abs(rw["alpha_median_over_t"] - 1) > 0.1:
        onset = rw["r_hi"]
        break
stats["nonlinearity_onset_r_median_alpha"] = onset

save_json(stats, "r_scaling.json")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ax = axs[0]
for t in [20, 50, 100, 150, 250]:
    ax.plot(radii, med[:, t], "o-", lw=1.2, ms=3, label=f"t={t}")
ax.plot(radii, med[0, 100] * (radii / radii[0]), "k:", lw=1.5, label=r"$\propto r$ (from r=1e-3)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("perturbation radius r")
ax.set_ylabel(r"median $\|\delta h_t\|$ (L27)")
ax.set_title("Prompt-perturbation r-scan on the same w100 trajectory\n(proxy; state protocol untested at r>1e-3)", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = axs[1]
rmid = [np.sqrt(rw["r_lo"] * rw["r_hi"]) for rw in rows]
amed = [rw["alpha_median_over_t"] for rw in rows]
aq1 = [rw["alpha_iqr_over_t"][0] for rw in rows]
aq3 = [rw["alpha_iqr_over_t"][1] for rw in rows]
ax.plot(rmid, amed, "o-", color="#4c72b0", label="median over t")
ax.fill_between(rmid, aq1, aq3, color="#4c72b0", alpha=0.2, label="IQR over t")
ax.axhline(1.0, color="k", ls="--", lw=0.8)
if onset:
    ax.axvline(onset, color="#c44e52", ls=":", lw=1.2,
               label=f"median alpha leaves [0.9,1.1] at r~{onset:g}")
ax.set_xscale("log")
ax.set_xlabel(r"geometric mean r of the pair")
ax.set_ylabel(r"local exponent $\alpha=d\log\delta h/d\log r$")
ax.set_title("Linearity of the teacher-forced response in r (L27)", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
savefig(fig, "fig06_r_scaling.png")

print("alpha median over t per r pair:")
for rw in rows:
    print(f"  r {rw['r_lo']:g}->{rw['r_hi']:g}: alpha={rw['alpha_median_over_t']:.2f} "
          f"(IQR {rw['alpha_iqr_over_t'][0]:.2f}-{rw['alpha_iqr_over_t'][1]:.2f}, "
          f"frac |alpha-1|<0.1 = {rw['frac_t_alpha_within_0p9_1p1']:.2f})")
print("nonlinearity onset (median alpha leaves [0.9,1.1]):", onset)
