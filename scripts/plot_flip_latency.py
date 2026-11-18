"""Two kinds of token flips: immediate (at the response peak) and delayed
(at a later, closer decision).  Uses teacher-forced responses and observed
free-run flips.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"), allow_pickle=True)
fr = np.load(os.path.join(BASE, "fr_w100_large.npz"), allow_pickle=True)
margins = d["margins"]
radii = d["radii"]
layers = list(d["layers"])
li = layers.index(27)
obs = fr["obs"]
fr_radii = fr["radii"]

rows = []
for fri, r in enumerate(fr_radii):
    ri = int(np.argmin(np.abs(radii - r)))
    for di in range(obs.shape[1]):
        T = int(obs[fri, di])
        if T < 0:
            continue
        resp = d["dh_norm"][ri, di, li]
        rows.append(dict(r=float(r), di=di, T=T, m_T=float(margins[T]),
                         ratio=float(resp[T] / resp[: T + 1].max()),
                         t_peak=int(resp[: T + 1].argmax())))

late = [x for x in rows if x["ratio"] < 0.3]
early = [x for x in rows if x["ratio"] >= 0.3]
print("flips:", len(rows), "immediate:", len(early), "delayed:", len(late))
print("delayed steps:", sorted(set(x["T"] for x in late)))
print("delayed margins:", sorted(set(round(x["m_T"], 4) for x in late)))

fig, axs = plt.subplots(2, 2, figsize=(13, 9))

# (a) margin distribution, log x
ax = axs[0, 0]
bins = np.logspace(np.log10(max(margins.min(), 1e-3)), np.log10(margins.max()), 35)
ax.hist(margins, bins=bins, color="lightgrey", label="all 300 steps")
ax.hist([x["m_T"] for x in rows], bins=bins, color="tab:red", alpha=0.85,
        label="steps where a flip happens")
ax.set_xscale("log")
ax.set_xlabel("baseline top1-top2 logit gap at that step\n(small = a close call, log scale)")
ax.set_ylabel("number of steps / flips")
ax.set_title("(a) Flips need a close call: they sit at the small-gap end")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# (b) two classes
ax = axs[0, 1]
for r, c in zip(sorted(set(x["r"] for x in rows)), ["tab:blue", "tab:orange", "tab:green"]):
    ax.hist([x["ratio"] for x in rows if x["r"] == r], bins=np.linspace(0, 1, 16),
            alpha=0.55, color=c, label=f"r={r:g}")
ax.set_xlabel("response size at the flip step / largest response before it")
ax.set_ylabel("number of flips")
ax.set_title("(b) Two kinds of flips: at the response peak (~1),\nor long after it has faded (~0.1)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# (c) flip step vs ratio
ax = axs[1, 0]
for r, c in zip(sorted(set(x["r"] for x in rows)), ["tab:blue", "tab:orange", "tab:green"]):
    sel = [x for x in rows if x["r"] == r]
    ax.scatter([x["T"] for x in sel], [x["ratio"] for x in sel], s=40, color=c,
               alpha=0.85, label=f"r={r:g}")
ax.set_xlabel("generated token t where the first flip happens")
ax.set_ylabel("response size at that step / its largest value before")
ax.set_title("(c) Delayed flips cluster at t=116-143,\nthe closest calls of this prompt")
ax.axvspan(118, 148, color="grey", alpha=0.15)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.set_ylim(-0.03, 1.05)

# (d) a delayed example
ax = axs[1, 1]
ex = max(late, key=lambda x: x["T"]) if late else rows[0]
di = ex["di"]
ri = int(np.argmin(np.abs(radii - ex["r"])))
t = np.arange(len(margins))
resp = d["dh_norm"][ri, di, li]
ax.semilogy(t, np.maximum(margins, 1e-4), color="k", lw=0.9,
            label="baseline top1-top2 gap")
ax2 = ax.twinx()
ax2.plot(t, resp / resp.max(), color="tab:blue", lw=1.3,
         label="perturbation response (right axis, normalized)")
ax.axvline(ex["T"], color="tab:red", ls="--", lw=1.2)
ax.annotate(f"flip at t={ex['T']}\ngap={ex['m_T']:.4f}\nresponse={ex['ratio']:.2f} of its peak",
            xy=(ex["T"], ex["m_T"]), xytext=(175, 0.25), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlabel("generated token t")
ax.set_ylabel("baseline top1-top2 gap (log scale)")
ax2.set_ylabel("perturbation response / its maximum", color="tab:blue")
ax.set_title(f"(d) Delayed flip example (direction {di}, r={ex['r']:g})")
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, ["baseline gap", "response (right axis)"], fontsize=8, loc="lower left")
ax.grid(alpha=0.25)

fig.suptitle("A flip happens when a decision becomes close; the perturbation can sit latent and act much later", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flip_is_a_close_call.png"), dpi=190)
print("saved")
