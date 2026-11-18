"""fig3 -- WHAT THE PREDICTION IS WORTH: the cost of a flip, measured on the
real generated texts (47 flipped runs with stored sequences, 3 prompts).

(a) token agreement with the unperturbed run as a function of words after the
    first changed word: 3 individual examples + the pooled mean. A flip is not
    a one-word typo: after it, the two continuations stay ~97% different and
    0/47 runs ever come back into sync.
(b) fraction of the whole 300-word answer that is still identical vs the first
    changed word's position: the dots hug the 'everything after the flip is
    lost' line, so predicting the flip step = predicting how much text
    survives.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common2 as C

# gather all flipped runs with sequences
runs = []   # dict(prompt, r, T, agree(bool array), frac_all, resynced)
for e in C.load_free("all"):
    if e["npz"] is None:
        continue
    f = e["npz"]
    ri = int(np.argmin(np.abs(f.radii - e["r"])))
    for d in range(f.K):
        T = int(e["obs"][d])
        if T < 0:
            continue
        ref, row = f.rows(ri, d)
        a = C.agreement_curve(ref, row)
        st = C.after_flip_stats(ref, row, T, tail=50)
        runs.append(dict(prompt=e["prompt"], r=float(e["r"]), d=d, T=T,
                         agree=a, frac_all=st["frac_identical_all"],
                         resynced=st["resynced"]))
print(f"flipped runs with sequences: {len(runs)}; re-synced: "
      f"{sum(r['resynced'] for r in runs)}")
n_noflip = 320 - 75  # 75 flips observed overall; 46 of them have sequences
pooled_after = np.mean([r["agree"][r["T"]:].mean() for r in runs])
print(f"mean agreement after the flip: {pooled_after:.3f}")

def roll(x, w=10):
    """rolling mean, window w, same length (edge-clamped)"""
    c = np.convolve(x.astype(float), np.ones(w) / w, mode="same")
    half = w // 2
    for i in range(half):
        c[i] = x[: i + half + 1].mean()
        c[-(i + 1)] = x[-(i + half + 1):].mean()
    return c

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.7),
                              gridspec_kw=dict(width_ratios=[1.25, 1]))

# ---------------------------------------------------------------- (a)
col_of = dict(propulsion="#1f77b4", cyber="#d62728", childhood="#2ca02c")
examples = [
    ("childhood", 0.1, 12, "childhood prompt, r=0.1, dir 12 (flip at word 173):\n"
     "brief echo of agreement, then permanently lost"),
    ("propulsion", 0.5, 12, "propulsion prompt, r=0.5, dir 12 (flip at word 123):\n"
     "lost immediately and permanently"),
    ("cyber", 0.3, 1, "cyber prompt, r=0.3, dir 1 (flip at word 39)"),
]
for prompt, r, d, lab in examples:
    rr = [x for x in runs if x["prompt"] == prompt and abs(x["r"] - r) < 1e-9
          and x["d"] == d][0]
    aft = rr["agree"][rr["T"]:]
    ax.plot(np.arange(len(aft)), roll(aft), color=col_of[prompt], lw=1.2,
            alpha=0.85, label=lab)

# pooled mean curve (align at flip; average over runs still long enough)
L = 130
pooled = np.zeros(L)
ncov = np.zeros(L)
for rr in runs:
    aft = rr["agree"][rr["T"]:].astype(float)
    n = min(len(aft), L)
    pooled[:n] += aft[:n]
    ncov[:n] += 1
pooled /= ncov
ax.plot(np.arange(L), roll(pooled), color="k", lw=3.0, zorder=5,
        label=f"average over all {len(runs)} flipped runs\n"
              f"(mean agreement {pooled_after*100:.0f}% after the flip)")
ax.text(L * 0.62, 0.66,
        f"none of the {len(runs)} flipped runs ever came back\n"
        f"into sync (last 50 words never matched again)\n\n"
        f"all {n_noflip} runs where nothing flipped were\n"
        f"100% identical - the flip is the only thing\n"
        f"that separates the two texts",
        fontsize=10.6, ha="left", va="top",
        bbox=dict(fc="#f2f2f2", ec="0.6"))
ax.set_xlabel("words after the first changed word", fontsize=12.5)
ax.set_ylabel("fraction of words identical to the unperturbed run\n"
              "(10-word rolling window)", fontsize=12.5)
ax.set_ylim(-0.04, 1.06)
ax.set_xlim(0, L)
ax.axhline(1, color="0.5", lw=0.8, ls=":", alpha=0.6)
ax.set_title("(a) after the first changed word, the two texts stay ~97% "
             "different forever", fontsize=12.5)
ax.legend(loc="upper right", fontsize=9.8, framealpha=0.95)
ax.grid(alpha=0.25, lw=0.5)

# ---------------------------------------------------------------- (b)
Ts = np.linspace(0, 180, 10)
ax2.plot(Ts, Ts / 300, "k--", lw=1.4,
         label="everything after the first change is lost (T / 300)")
for prompt in ["propulsion", "cyber", "childhood"]:
    sub = [r for r in runs if r["prompt"] == prompt]
    ax2.scatter([r["T"] for r in sub], [r["frac_all"] for r in sub],
                s=46, color=col_of[prompt], edgecolors="k", linewidths=0.6,
                zorder=5, label=f"{prompt} prompt ({len(sub)} runs)")
ax2.set_xlim(-6, 186)
ax2.set_ylim(-0.04, 1.06)
ax2.set_xlabel("position T of the first changed word in the 300-word answer",
               fontsize=12.5)
ax2.set_ylabel("fraction of the whole 300-word answer\nstill identical to the "
               "unperturbed run", fontsize=12.5)
ax2.set_title("(b) knowing the flip step = knowing how much of the answer "
              "survives", fontsize=12.5)
ax2.legend(loc="lower right", fontsize=10.2, framealpha=0.95)
ax2.grid(alpha=0.25, lw=0.5)
ax2.annotate("the dots hug the line: a flip at word T costs\n"
             "essentially the whole remaining answer\n"
             "(the small excess is ~3% accidental re-matches)",
             xy=(123, 0.423), xytext=(16, 0.02), fontsize=10.4,
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1),
             bbox=dict(fc="white", ec="0.7"))
ax2.annotate("flips at the very first words (5 runs)\nlose the entire answer",
             xy=(2, 0.007), xytext=(60, 0.115), fontsize=10,
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0),
             bbox=dict(fc="white", ec="0.7"))

fig.suptitle("What the prediction is worth: the first changed word is a point "
             "of no return (measured on 320 real generations, 3 prompts)",
             fontsize=13.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("fig3_cost_of_a_flip.png", dpi=165)
print("saved fig3_cost_of_a_flip.png")
