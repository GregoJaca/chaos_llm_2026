"""A1 window-size confirmation: w100 vs w200 vs w500 (same prompt, same injection).

Clean natural experiments (identical base streams up to each clip event):
  t=35  : w100's sliding window starts clipping (context 101 > 100).
          The w100 response JUMPS x1.7; w200/w500 are byte-identical to each
          other and unchanged. The free-run streams diverge only afterwards
          (first differing generated token at t=40).
  t=135 : w200 starts clipping (context 201 > 200): response jumps x1.5 with
          w500 (not yet clipping) as control; streams diverge at t=142.

Outputs:
  fig8_windows.png
  window_stats.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib.pyplot as plt

Ds = {w: C.load(w) for w in ("w100", "w200", "w500")}
COL = {"w100": "tab:red", "w200": "tab:blue", "w500": "tab:green"}
CLIP = {"w100": 35, "w200": 135, "w500": 435}   # context 65+t+1 > w  ->  t = w-35


def G_of(w, r=0.1, idir=None, li=-1):
    D = Ds[w]
    ir = int(np.where(np.isclose(D["radii"], r))[0][0])
    if idir is None:
        return D["dhm"][ir, li, :] / r
    return D["dh"][ir, idir, li, :] / r


G100, G200, G500 = G_of("w100"), G_of("w200"), G_of("w500")

# numbers
same_gen = {a + "_" + b: int(np.where(Ds[a]["gen_ids"] != Ds[b]["gen_ids"])[0].min())
            for a, b in [("w100", "w200"), ("w100", "w500"), ("w200", "w500")]}
jump35 = G100[35] / G200[35]
jump135 = G_of("w200")[135] / G_of("w500")[135]
id_34 = np.allclose(G100[:35], G200[:35]) and np.allclose(G200[:135], G500[:135])
rates = {}
for w in COL:
    tt = np.arange(150, 281)
    rates[w] = np.polyfit(tt, np.log(G_of(w)[tt]), 1)[0]
lines = [
    f"identical prefix: w100==w200 for t<=34 and w200==w500 for t<=134: {id_34}\n",
    f"t=35 clip event (w100): G_w100(35)/G_w200(35) = {jump35:.2f}; "
    f"G_w100(35)/G_w100(34) = {G100[35]/G100[34]:.2f}\n",
    f"t=135 clip event (w200): G_w200(135)/G_w500(135) = {jump135:.2f}\n",
    f"first differing generated token: {same_gen}\n",
    "tail exp rates [150,280] (CONFOUNDED: streams differ after t=40): " +
    str({w: round(v, 4) for w, v in rates.items()}) + "\n",
]
C.txt("window_stats.txt", "".join(lines))
print("".join(lines))

# ---------------------------------------------------------------- fig 8
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))

# (a) full curves
ax = axes[0, 0]
for w in ("w100", "w200", "w500"):
    ax.plot(G_of(w), lw=1.7, color=COL[w], label=f"{w} (window starts clipping "
           f"at t={CLIP[w]})")
ax.axvline(35, color=COL["w100"], ls=":", lw=1.3)
ax.axvline(135, color=COL["w200"], ls=":", lw=1.3)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("G(t,L27) = $\\delta h/r$ (r=0.1)")
ax.set_title("(a) same prompt + injection, three window sizes;\n"
             "identical until each window starts clipping", loc="left")
ax.legend(fontsize=8.5, loc="lower left")
ax.annotate("w100 clips", xy=(36, 12), fontsize=9, color=COL["w100"])
ax.annotate("w200 clips", xy=(136, 25), fontsize=9, color=COL["w200"])

# (b) zoom on the w100 clip event
ax = axes[0, 1]
for w in ("w100", "w200", "w500"):
    ax.plot(np.arange(25, 61), G_of(w)[25:61], lw=1.8, color=COL[w], label=w)
ax.axvline(35, color="k", ls="--", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("G(t,L27)")
ax.set_title(f"(b) t=35: context length exceeds w=100 $\\Rightarrow$\n"
             f"w100 jumps $\\times${jump35:.1f}; w200/w500 identical", loc="left")
ax.legend(fontsize=8.5)
ax.annotate("w200 = w500 exactly\n(unclipped, byte-identical\nforward passes)",
            xy=(37.2, 1.05), fontsize=8.5, color="0.25")
ax.annotate("first clipped token:\noldest prompt token dropped\n"
            "from attention", xy=(35.2, 5.2), fontsize=8.5, color="0.15")

# (c) zoom on the w200 clip event (w500 control)
ax = axes[1, 0]
tt = np.arange(125, 161)
for w in ("w200", "w500"):
    ax.plot(tt, G_of(w)[125:161], lw=1.8, color=COL[w], label=w)
ax.axvline(135, color="k", ls="--", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("G(t,L27)")
ax.set_title(f"(c) t=135: same event for w=200 with w=500\nas control: "
             f"jump $\\times${jump135:.1f}", loc="left")
ax.legend(fontsize=8.5)
ax.annotate("stream divergence follows\nclipping (first differing token\nt=40, t=142)",
            xy=(140, 6.5), fontsize=8.5, color="0.25")

# (d) individual directions show the same clip event
ax = axes[1, 1]
for idir, ls in [(0, "-"), (10, "--")]:
    for w in ("w100", "w200"):
        ax.plot(np.arange(25, 61), G_of(w, idir=idir)[25:61], ls, lw=1.6,
                color=COL[w], label=f"{w}, dir {idir}")
ax.axvline(35, color="k", ls="--", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("G(t,L27), single direction")
ax.set_title("(d) single directions (0 solid, 10 dashed):\nthe clip jump is "
             "direction-general", loc="left")
ax.legend(fontsize=7.5, ncol=2, loc="lower left")

fig.suptitle("Window mechanism confirmed by two clip events: the A1 response "
             "changes regime exactly when the sliding window starts discarding "
             "context (t=35 for w=100, t=135 for w=200),\nnot when the "
             "perturbation is injected; w200/w500 are byte-identical until "
             "their own clip step. Tail rates fall with window size "
             "(confounded after stream divergence at t=40).", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.945))
C.save(fig, "fig8_windows.png")
