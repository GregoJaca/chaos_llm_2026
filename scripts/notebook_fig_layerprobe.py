"""Where does a perturbation get amplified? Inject at layer L, read every layer.

HF convention: hidden_states[l] is the INPUT of block l, so injecting into the
output of block L first shows up in hidden_states[L+1].
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\layer_probe"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})
colors = {"pos50": "tab:blue", "pos16": "tab:red"}

for ax, (name, pos_label) in zip(axs[:2], [("pos50", r"$s=50$ (ordinary position)"),
                                          ("pos16", r"$s=16$ (sensitive position)")]):
    d = np.load(os.path.join(B, f"{name}.npz"))
    dh = d["dh"]; inj = list(d["layers"]); pos = int(d["pos"]); r = float(d["r"])
    for i, L in enumerate(inj):
        if L + 1 >= 28:
            continue
        y = np.array([np.median(dh[i, :, l, pos]) for l in range(L + 1, 28)]) / r
        ax.semilogx(np.arange(1, 28 - L) + 0, y, "o-", ms=3, lw=1.0,
                    label=fr"inject $L={L}$")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel(r"layers after the injection, $\ell-L$")
    ax.set_ylabel(r"$a(\ell)/r$  (response relative to the injected size)")
    ax.set_title(f"({ 'a' if name=='pos50' else 'b'}) {pos_label}")
    ax.legend(fontsize=6.0, ncol=2)
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(0, 6)

# (c) summary: total gain at the last layer vs injection layer
ax = axs[2]
for name, col, lab in [("pos50", "tab:blue", r"$s=50$"),
                       ("pos16", "tab:red", r"$s=16$")]:
    d = np.load(os.path.join(B, f"{name}.npz"))
    dh = d["dh"]; inj = list(d["layers"]); pos = int(d["pos"]); r = float(d["r"])
    xs, ys = [], []
    for i, L in enumerate(inj):
        if L + 1 >= 28:
            continue
        xs.append(L)
        ys.append(np.median(dh[i, :, 27, pos]) / r)
    ax.semilogx(np.array(xs) + 1, ys, "o-", color=col, label=lab)
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel(r"injection layer $L+1$")
ax.set_ylabel(r"response at the last layer, $a(27)/r$")
ax.set_title("(c) Total gain depends on where the perturbation enters,\nand is ≈1 or below at an ordinary position")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "layer_probe.png"), dpi=190)
print("saved")
for name in ["pos50", "pos16"]:
    d = np.load(os.path.join(B, f"{name}.npz")); dh = d["dh"]; inj = list(d["layers"])
    pos = int(d["pos"]); r = float(d["r"])
    print(name, "gains at L27:", [(L, round(float(np.median(dh[i, :, 27, pos]) / r), 2))
                                  for i, L in enumerate(inj) if L + 1 < 28])
