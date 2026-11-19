"""Where are the near-ties in a 4000-token run?"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\a1_4k_test.npz")
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
m = d["margins"]
T = len(m)

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.4))
ax = axs[0]
ax.semilogy(np.arange(T), np.maximum(m, 1e-3), lw=0.5, color="0.3")
ax.axvline(300, color="tab:red", ls="--", lw=1)
ax.axhline(0.05, color="tab:orange", ls=":", lw=1)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"decision gap $m_t$ (log)")
ax.set_title("(a) All close calls are in the first 300 tokens\n(after that the model is confident, $m_t>2.6$)")
ax.grid(alpha=0.25)
ax.text(320, 0.06, "no gap < 0.05 after $t=300$", fontsize=8, color="tab:red")

ax = axs[1]
bins = [0, 300, 600, 1200, 2000, 3000, 4000]
mins = [m[a:b].min() for a, b in zip(bins[:-1], bins[1:])]
ax.bar(range(len(mins)), mins, color=["tab:red"] + ["0.7"] * (len(mins) - 1))
ax.set_yscale("log")
ax.set_xticks(range(len(mins)))
ax.set_xticklabels([f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])], fontsize=8)
ax.set_xlabel("token range")
ax.set_ylabel("smallest gap in the range")
ax.set_title("(b) Minimum gap per block: the 300-token horizon\ncaptures every decision boundary")
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(OUT + r"\long_margins.png", dpi=190)
print("saved; min gap after 300:", float(m[300:].min()))
