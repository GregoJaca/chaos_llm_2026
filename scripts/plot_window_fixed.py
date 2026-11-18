"""Fixed-input window experiment on two prompts: identical prompt AND identical
token sequence, only the attention window changes.  Departure step collapses
onto  window - prompt_length  for both prompts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"
R = 0.1

SETS = {
    "propulsion review (65-token prompt)": (os.path.join(BASE, "window_fixed"), [100, 150, 200, 300, 1000]),
    "cybersecurity review (62-token prompt)": (os.path.join(BASE, "window_cyber"), [100, 200, 1000]),
}

res = {}
for label, (folder, wins) in SETS.items():
    resp, P = {}, None
    for w in wins:
        d = np.load(os.path.join(folder, f"w{w}", "a1.npz"), allow_pickle=True)
        P = int(d["prompt_ids"].shape[0])
        L = list(d["layers"]); li = L.index(27)
        ri = int(np.argmin(np.abs(d["radii"] - R)))
        resp[w] = np.median(d["dh_norm"][ri, :, li, :], axis=0)
    ref = resp[wins[-1]]
    dep = {}
    for w in wins[:-1]:
        r = resp[w] / np.maximum(ref, 1e-12)
        dev = np.where(np.abs(r - 1.0) > 1e-6)[0]
        dep[w] = int(dev[0]) if len(dev) else -1
    res[label] = dict(P=P, resp=resp, dep=dep, wins=wins, ref=ref)

fig = plt.figure(figsize=(13, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.05, 1.0], wspace=0.3)
cols = {"propulsion review (65-token prompt)": "tab:blue",
        "cybersecurity review (62-token prompt)": "tab:orange"}

# (a) response curves, propulsion
ax = fig.add_subplot(gs[0, 0])
label = "propulsion review (65-token prompt)"
r0 = res[label]
t = np.arange(len(r0["ref"]))
for w in r0["wins"]:
    ax.semilogy(t, np.maximum(r0["resp"][w] / r0["ref"][0], 1e-12), lw=1.1, label=f"window {w}")
for w in r0["wins"][:-1]:
    ax.axvline(r0["dep"][w], color="k", ls=":", lw=0.7, alpha=0.5)
ax.set_xlabel("generated token t")
ax.set_ylabel("response size / its value at t=0")
ax.set_title("(a) Same prompt, same tokens, five windows.\nDotted lines are the predicted departure steps.")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25)

# (b) ratio to the no-clipping control, both prompts
ax = fig.add_subplot(gs[0, 1])
for label, r0 in res.items():
    ref = r0["ref"]
    for w in r0["wins"][:-1]:
        if w not in (100, 200):
            continue
        t = np.arange(len(ref))
        ax.semilogy(t, r0["resp"][w] / np.maximum(ref, 1e-12), lw=1.0,
                    color=cols[label], alpha=0.9,
                    ls="-" if w == 100 else "--",
                    label=f"w={w}, {label.split(' (')[0]}")
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel("generated token t")
ax.set_ylabel("ratio to the widest window (1 = identical)")
ax.set_title("(b) Every narrow window is exact\nuntil its own boundary")
ax.legend(fontsize=6.5, ncol=1)
ax.grid(alpha=0.25)

# (c) departure collapse
ax = fig.add_subplot(gs[0, 2])
for label, r0 in res.items():
    xs = [w - r0["P"] for w in r0["dep"]]
    ys = [r0["dep"][w] for w in r0["dep"]]
    ax.scatter(xs, ys, s=80, color=cols[label], label=label, zorder=3)
lim = [0, 250]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("window size minus prompt length")
ax.set_ylabel("measured first departure step")
ax.set_title("(c) Same law on two prompts:\ndeparture = window - prompt length")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)

fig.suptitle("The rise/decay shape is produced by the sliding window, not by intrinsic dynamics", fontsize=12, y=0.98)
fig.subplots_adjust(top=0.80)
fig.savefig(os.path.join(OUT, "a1_window_fixed_inputs.png"), dpi=190)
print("saved")
for label, r0 in res.items():
    print(label, "P=", r0["P"], "departures", r0["dep"])
