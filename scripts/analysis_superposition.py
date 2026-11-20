"""Superposition test: does the sum of single-position responses equal the all-position response?"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "superposition")
OUT = os.path.join(ROOT, "notebook", "figures", "transport")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

rows = {}
for d in sorted(glob.glob(os.path.join(B, "*"))):
    f = os.path.join(d, "superposition.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    r = float(z["r"])
    name = os.path.basename(d)
    if not name.startswith("deepseek"):
        continue
    key = [k for k in z.files if k.startswith("err_L")][-1]
    err = z[key]
    rows[r] = dict(err_med=float(np.median(err)), err_q90=float(np.percentile(err, 90)),
                   err_by_t=np.median(err, axis=0))
print({k: round(v["err_med"], 4) for k, v in sorted(rows.items())})

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axs[0]
rs = sorted(rows)
ax.loglog(rs, [rows[r]["err_med"] for r in rs], "o-", color="tab:blue", label="median over tokens/trials")
ax.loglog(rs, [rows[r]["err_q90"] for r in rs], "s--", color="tab:orange", label="90th percentile")
ax.set_xlabel("perturbation radius per position $r$")
ax.set_ylabel(r"$\|A-S\|/\|A\|$  (0 = perfect superposition)")
ax.set_title("(a) Superposition error vs amplitude, DeepSeek")
ax.legend(); ax.grid(alpha=0.25, which="both")
ax2 = ax.twiny()
ax2.set_xscale("log")
lo, hi = ax.get_xlim()
ax2.set_xlim(lo, hi)
ax2.set_xlabel(r"total norm added to the prompt $r\sqrt{P}$  ($P=65$: $r=0.03\to0.24$)")

ax = axs[1]
for r in rs:
    ax.semilogy(np.arange(len(rows[r]["err_by_t"])), np.maximum(rows[r]["err_by_t"], 1e-4),
                lw=1.0, label=f"$r={r}$")
ax.set_xlabel("generated token $t$"); ax.set_ylabel(r"$\|A-S\|/\|A\|$ (median over trials)")
ax.set_title("(b) Error along the continuation")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "superposition.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "err_by_t"} for k, v in rows.items()},
          open(os.path.join(ROOT, "results", "superposition.json"), "w"), indent=1)
print("saved")
