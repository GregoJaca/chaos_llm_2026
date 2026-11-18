"""Clean summary of threshold-rule accuracy on the same 320 direction-runs.

Counts derived from results/subagent_threshold2/eval_summary.txt:
  signed rule (exact radius): 57 exact, 261 no-flip correct, 2 wrong step, 0 false, 0 missed
  delivered sign-blind rule : 45 exact, 240 no-flip correct, 13 wrong step, 21 false, 1 missed
(childhood rows, scored with the delivered rule only, are excluded from both.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"
cats = ["exact flip\nstep", "correctly says\n'no flip'", "flip step\nwrong", "predicts a flip\nthat never happens", "misses a\nreal flip"]
signed = [57, 261, 2, 0, 0]
blind = [45, 240, 13, 21, 1]
colors = ["#2a9d8f", "#8ecae6", "#f4a261", "#e63946", "#6a4c93"]

x = np.arange(len(cats))
w = 0.38
fig, ax = plt.subplots(figsize=(10.5, 4.8))
b1 = ax.bar(x - w / 2, signed, w, label="signed rule, measured at the kick size being predicted (99.4% correct)", color="#2a9d8f")
b2 = ax.bar(x + w / 2, blind, w, label="sign-blind rule (the version evaluated before) — 89.1% correct", color="#adb5bd")
for bars, vals in [(b1, signed), (b2, blind)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, str(v), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel("number of direction-runs (out of 320)")
ax.set_title("One teacher-forced pass predicts the first changed word: 320 direction-runs, 3 prompts", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(alpha=0.25, axis="y")
ax.set_ylim(0, 305)
fig.tight_layout()
fig.savefig(rf"{OUT}\threshold_rule_scorecard_clean.png", dpi=190)
print("saved")
