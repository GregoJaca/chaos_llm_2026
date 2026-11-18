"""Fixed-input window experiment: identical prompt AND identical token sequence,
only the attention window changes.  Everything is bit-identical until the
context exceeds the window, then that window's curve departs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_fixed"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"
WINDOWS = [100, 150, 200, 300, 1000]
R = 0.1

resp, P = {}, None
for w in WINDOWS:
    d = np.load(os.path.join(BASE, f"w{w}", "a1.npz"), allow_pickle=True)
    P = int(d["prompt_ids"].shape[0])
    L = list(d["layers"]); li = L.index(27)
    ri = int(np.argmin(np.abs(d["radii"] - R)))
    resp[w] = np.median(d["dh_norm"][ri, :, li, :], axis=0)

T = len(resp[1000])
t = np.arange(T)
ref = resp[1000]                      # never clips in this range: the control

print("prompt length", P, "tokens", T)
dep = {}
for w in WINDOWS[:-1]:
    ratio = resp[w] / np.maximum(ref, 1e-12)
    dev = np.where(np.abs(ratio - 1.0) > 1e-6)[0]
    dep[w] = int(dev[0]) if len(dev) else -1
    print(f"window {w}: first departure at t={dep[w]}  predicted {w - P}")

fig = plt.figure(figsize=(14, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.3)

# (a) response curves
ax = fig.add_subplot(gs[0, 0])
for w in WINDOWS:
    ax.semilogy(t, np.maximum(resp[w], 1e-12) / ref[0], lw=1.2, label=f"window {w}")
ax.set_xlabel("generated token t")
ax.set_ylabel("response size / its value at t=0\n(median over 8 directions, last layer)")
ax.set_title("(a) Same prompt, same tokens, five windows")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.25)

# (b) ratio to the never-clipping window
ax = fig.add_subplot(gs[0, 1])
for w, c in zip(WINDOWS[:-1], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    ax.semilogy(t, resp[w] / np.maximum(ref, 1e-12), lw=1.0, color=c,
                label=f"window {w} / window 1000 (departs t={dep[w]})")
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel("generated token t")
ax.set_ylabel("ratio of response sizes (1 = identical)")
ax.set_title("(b) Each window is exact until its own boundary")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

# (c) departure step vs window
ax = fig.add_subplot(gs[0, 2])
ww = np.array([50, 100, 150, 200, 300, 500])
ax.plot(ww, np.maximum(0, ww - P), "k--", lw=1, label="prediction: window - prompt")
ws = list(dep)
ax.scatter(ws, [dep[w] for w in ws], s=70, color="tab:blue", zorder=3,
           label="measured departure")
ax.set_xlabel("sliding-window size (tokens)")
ax.set_ylabel("first step where the curve departs")
ax.set_title("(c) Departure step = window - prompt (exact)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
ax.set_xlim(80, 520)
ax.set_ylim(-10, 480)

fig.suptitle("The rise/decay shape is produced by the sliding window: with identical inputs, the curve departs exactly when the context exceeds the window",
             fontsize=11.5)
fig.savefig(os.path.join(OUT, "a1_window_fixed_inputs.png"), dpi=190, bbox_inches="tight")
print("saved")
