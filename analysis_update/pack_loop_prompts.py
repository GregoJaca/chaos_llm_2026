"""Multi-prompt replication of the loop gain.

Each prompt is an independent orbit, so each prompt gives an independent L_inf.
Pooling prompts is the unit of replication we were missing (everything else in
this report is one prompt).  Methodologically it is the right thing to do: the
directions inside one run are not independent samples of the *orbit*, so the
direction-IQR underestimates the uncertainty of L_inf; the prompt spread does not.
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE
from style import plt, save

PMAX = 4096


def sat(t, L, tau):
    return L * (1 - np.exp(-t / tau))


def pairs_for(tag, prompt, T="T1.0"):
    """pairs in results/longloop_v3b_prompt/<tag>/p<p>/<T>/ or longloop_prompt/"""
    out = []
    for root in ("results/longloop_v3b_prompt", "results/longloop_prompt"):
        base = os.path.join(HERE, root, tag, prompt, T)
        if not os.path.isdir(base):
            continue
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            pl = os.path.join(d, "loop", "softloop.npz")
            pr = os.path.join(d, "replay", "softloop.npz")
            if os.path.exists(pl) and os.path.exists(pr):
                out.append((pl, pr))
        pl = os.path.join(base, "loop", "softloop.npz")
        pr = os.path.join(base, "replay", "softloop.npz")
        if os.path.exists(pl) and os.path.exists(pr):
            out.append((pl, pr))
    if not out and prompt == "p1":
        d = os.path.join(HERE, "results/longloop", tag, T)
        pl = os.path.join(d, "loop", "softloop.npz"); pr = os.path.join(d, "replay", "softloop.npz")
        if os.path.exists(pl) and os.path.exists(pr):
            out.append((pl, pr))
    return out


def curve(pl, pr):
    zc = np.load(pl, allow_pickle=True); zr = np.load(pr, allow_pickle=True)
    ac = zc["dh"][:, -1, :].astype(float); ar = zr["dh"][:, -1, :].astype(float)
    n = min(ac.shape[1], ar.shape[1])
    ac = ac[:, :n] / np.maximum(ac[:, [1]], 1e-30)
    ar = ar[:, :n] / np.maximum(ar[:, [1]], 1e-30)
    return np.log(np.maximum(ac, 1e-30)) - np.log(np.maximum(ar, 1e-30))


MODELS = [("DeepSeek-1.5B", "ds15"), ("Qwen3-4B", "qwen34"),
          ("Qwen3-14B", "qwen314"), ("DeepSeek-14B", "ds14")]
PROMPTS = ["p1", "p2", "p3"]

fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.6))
rows = {}
for ax, (name, tag) in zip(axes, MODELS):
    Lends = []
    for p in PROMPTS:
        ps = pairs_for(tag, p)
        if not ps:
            continue
        Ld = np.concatenate([curve(pl, pr) for pl, pr in ps], axis=0)
        m = np.nanmedian(Ld, axis=0)
        t = np.arange(len(m))
        ax.plot(t[1:], m[1:], lw=1.4, color=COL.get(name, "gray"), alpha=0.9 if p == "p1" else 0.55,
                ls="-" if p == "p1" else ("--" if p == "p2" else ":"),
                label=f"{p} ({Ld.shape[0]} dirs)")
        k0 = int(0.8 * len(t))
        Lends.append(float(np.nanmedian(Ld[:, k0:])))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$L(t)=\log(a_{\rm closed}/a_{\rm replay})$")
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=6.2, title="prompt", title_fontsize=6.2)
    ax.grid(alpha=0.3)
    rows[name] = Lends
fig.suptitle("The same measurement on different prompts (each prompt = an independent orbit)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "pack_loop_prompts.png")

fig, ax = plt.subplots(figsize=(10.4, 4.4))
names = [n for n, v in rows.items() if v]
for i, n in enumerate(names):
    v = np.array(rows[n])
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v,
               color=COL[n], s=55, zorder=3)
    ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color="k", lw=2)
    ax.text(i + 0.26, np.median(v), f"{np.median(v):.1f} nats", va="center", fontsize=7)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=7.6)
ax.set_ylabel(r"$L_{\rm end}$ per prompt (nats)")
ax.set_title("Prompt-to-prompt variation of the loop gain (black = median over prompts)", loc="left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_loop_prompts_bars.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_loop_prompts.json"), "w"), indent=1)
print({k: [round(x, 2) for x in v] for k, v in rows.items()})
