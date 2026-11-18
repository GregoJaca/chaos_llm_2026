"""Clean figures for the notebook: two curves, explicit prompt region.

Both figures use the *same* prompt and the *same* generated tokens; only the
attention window differs.  r = 0.1 prompt perturbation, last layer.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_fixed"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
R = 0.1


def load(w):
    d = np.load(os.path.join(BASE, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27)
    ri = int(np.argmin(np.abs(d["radii"] - R)))
    return d, np.median(d["dh_norm"][ri, :, li, :], axis=0)


d100, a100 = load(100)
d1000, a1000 = load(1000)
P = int(d100["prompt_ids"].shape[0])
T = len(a100)
t = np.arange(T)

# ---------------- Figure: response vs token, prompt included ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

ax = axs[0]
ax.axvspan(-P, 0, color="0.92", zorder=0)
ax.text(-P / 2, 1.3, "prompt\n(65 tokens)", ha="center", fontsize=8, color="0.35")
ax.semilogy(t, a100, lw=1.3, color="tab:blue", label=r"$W=100$")
ax.semilogy(t, a1000, lw=1.3, color="black", label=r"$W=1000$ (no clipping)")
ax.axvline(0, color="0.4", lw=0.8)
ax.axvline(35, color="tab:red", ls="--", lw=1)
ax.annotate("at $t=35$ the context has 100 tokens:\nthis is where $W=100$ starts to clip",
            xy=(35, 0.3), xytext=(90, 0.55), fontsize=8, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax.set_xlabel(r"generated token index $t$ (0 = first generated token)")
ax.set_ylabel(r"$\|\delta h_{t,27}\|$  (response magnitude, layer 27)")
ax.set_title(r"(a) Same prompt, same tokens, $r=0.1$: two windows")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.25)

ax = axs[1]
lo, hi = 0, 60
ax.semilogy(t[lo:hi], a100[lo:hi], "o-", ms=3, lw=1.2, color="tab:blue", label=r"$W=100$")
ax.semilogy(t[lo:hi], a1000[lo:hi], "s--", ms=3, lw=1.2, color="black", label=r"$W=1000$")
ax.axvline(35, color="tab:red", ls="--", lw=1)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\|\delta h_{t,27}\|$")
ax.set_title(r"(b) $t\in[0,60]$: identical up to 34, separate after", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_response_two_curves.png"), dpi=190)
plt.close(fig)

# ---------------- Figure: identical tokens, prompt included ----------------
# Here the two runs generated their own text, so we can compare the text itself.
B2 = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
g100 = np.load(os.path.join(B2, "w100_layers", "a1.npz"), allow_pickle=True)["gen_ids"]
g1000 = np.load(os.path.join(B2, "w200", "a1.npz"), allow_pickle=True)["gen_ids"]
n = min(len(g100), len(g1000))
same = (g100[:n] == g1000[:n]).astype(float)
frac = np.cumsum(same) / np.arange(1, n + 1)
mm = np.where(g100[:n] != g1000[:n])[0]
first = int(mm[0]) if len(mm) else -1

fig, ax = plt.subplots(figsize=(11.8, 4.4))
ax.axvspan(-P, 0, color="0.92", zorder=0)
ax.text(-P / 2, 1.02, "prompt", ha="center", fontsize=8, color="0.35")
xx = np.concatenate([np.arange(-P, 0), np.arange(0, n)])
yy = np.concatenate([np.ones(P), frac])
ax.plot(xx, yy, lw=1.6, color="tab:blue")
ax.axvline(0, color="0.4", lw=0.8)
ax.axvline(35, color="tab:red", ls="--", lw=1)
ax.axvline(first, color="tab:orange", ls=":", lw=1.2)
ax.text(35, 0.86, "$t=35$: context reaches 100 tokens\n(response departs here)", color="tab:red", fontsize=8)
ax.text(first + 3, 0.62, f"$t={first}$: first different generated token", color="tab:orange", fontsize=8)
ax.set_xlabel(r"token index $t$ (negative = prompt, 0 = first generated token)")
ax.set_ylabel("fraction of tokens identical\nto the $W=200$ run, up to $t$")
ax.set_title(r"$W=100$ vs $W=200$: identical text for 40 tokens", fontsize=11)
ax.set_ylim(0, 1.08)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_token_agreement.png"), dpi=190)
print("saved; prompt", P, "first token difference", first)
