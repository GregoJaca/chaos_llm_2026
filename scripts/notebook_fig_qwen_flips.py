"""Cross-model stable-chaos: flip fraction vs r (from free-run obs)."""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\models"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def collect(folder):
    out = {}
    for f in glob.glob(os.path.join(B, folder, "chaos", "*_r*", "free_run.npz")):
        name = os.path.basename(os.path.dirname(f))
        prompt, r = name.split("_r"); r = float(r)
        d = np.load(f, allow_pickle=True)
        out.setdefault(prompt, {})[r] = d["obs"]
    return out


qwen = collect("qwen3_4b")
print("qwen prompts", list(qwen), "radii", sorted(next(iter(qwen.values())).keys()) if qwen else None)

fig, ax = plt.subplots(figsize=(8.6, 4.6))
for p, d in qwen.items():
    rs = sorted(d)
    fr = [np.mean(np.asarray(d[r]) >= 0) for r in rs]
    ax.semilogx(rs, fr, "o-", label=f"Qwen3-4B {p}")
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"perturbation radius $r$")
ax.set_ylabel("fraction of directions that flip")
ax.set_title("Qwen3-4B: flip fraction vs radius (20 directions, 300 tokens)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "qwen_flip_fraction.png"), dpi=190)
print("saved")
for p, d in qwen.items():
    print(p, [(r, round(float(np.mean(np.asarray(d[r]) >= 0)), 2)) for r in sorted(d)])
