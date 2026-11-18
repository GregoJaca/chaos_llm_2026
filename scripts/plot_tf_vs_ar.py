"""Teacher forcing vs autoregressive generation, explicitly.

For each perturbation we have:
  pred : tokens from one teacher-forced pass along the UNPERTURBED token sequence
  ar   : tokens from the autoregressive (free-running) perturbed generation
  ref  : unperturbed tokens
Question: for how long does the teacher-forced prediction agree with what the
model actually generates, and does it get the first flipped token right?
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\tf_pred"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

sets = {}
for name in ["mid", "large"]:
    d = np.load(os.path.join(BASE, f"{name}.npz"), allow_pickle=True)
    sets[name] = d

rows = []
for name, d in sets.items():
    pred, ar, ref = d["pred"], d["ar"], d["ref"]
    radii = d["radii"]
    for ri, r in enumerate(radii):
        for di in range(pred.shape[1]):
            a, p, f = ar[ri, di], pred[ri, di], ref
            valid = a >= 0
            diff = np.where((a != f) & valid)[0]
            Tflip = int(diff[0]) if len(diff) else -1
            rows.append(dict(r=float(r), di=di, T=Tflip, a=a, p=p, f=f,
                             agree_all=float(np.mean(p[valid] == a[valid]))))

print("cases", len(rows), "with a flip", sum(r["T"] >= 0 for r in rows))

# agreement as a function of distance from the flip
maxlead = 60
by_lead = {}
for x in rows:
    if x["T"] < 0:
        continue
    for k in range(0, maxlead):
        t = x["T"] + k
        if t >= len(x["a"]) or x["a"][t] < 0:
            continue
        by_lead.setdefault(k, []).append(1.0 if x["p"][t] == x["a"][t] else 0.0)

leads = np.array(sorted(by_lead))
acc = np.array([np.mean(by_lead[k]) for k in leads])
n = np.array([len(by_lead[k]) for k in leads])

# agreement as a function of radius, split before/after flip
rad_vals = sorted(set(x["r"] for x in rows))
pre = []
post = []
for r in rad_vals:
    sel = [x for x in rows if x["r"] == r and x["T"] >= 0]
    pp, po = [], []
    for x in sel:
        T = x["T"]
        valid = x["a"] >= 0
        pp.append(np.mean(x["p"][:T][valid[:T]] == x["a"][:T][valid[:T]]))
        m = min(len(x["a"]) - T, 30)
        po.append(np.mean(x["p"][T:T + m][valid[T:T + m]] == x["a"][T:T + m][valid[T:T + m]]))
    pre.append(np.mean(pp) if pp else np.nan)
    post.append(np.mean(po) if po else np.nan)

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax = axs[0]
ax.plot(leads, acc, "o-", ms=3, lw=1.2, color="tab:blue")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("tokens after the first flip")
ax.set_ylabel("fraction of tokens where the teacher-forced\nprediction equals the actual free-run token")
ax.set_title("(a) One teacher-forced pass agrees exactly at the flip,\nthen loses the thread within a few tokens")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)

ax = axs[1]
x = np.arange(len(rad_vals))
ax.bar(x - 0.18, pre, width=0.36, label="before the flip", color="tab:green")
ax.bar(x + 0.18, post, width=0.36, label="30 tokens after the flip", color="tab:red")
ax.set_xticks(x); ax.set_xticklabels([f"{r:g}" for r in rad_vals])
ax.set_xlabel("perturbation size r")
ax.set_ylabel("fraction of tokens predicted correctly")
ax.set_title("(b) Accuracy before and after the first flip")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")

ax = axs[2]
for r, c in zip(rad_vals, plt.cm.viridis(np.linspace(0, 0.9, len(rad_vals)))):
    sel = [x["agree_all"] for x in rows if x["r"] == r]
    ax.scatter([r] * len(sel), sel, s=22, color=c, alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("perturbation size r")
ax.set_ylabel("fraction of all 300 tokens predicted correctly")
ax.set_title("(c) Teacher forcing is exact for small r\nand degrades only where flips are frequent")
ax.grid(alpha=0.25)

fig.suptitle("Teacher forcing vs autoregressive generation (window 100, 20 directions)", fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "teacher_forced_vs_autoregressive.png"), dpi=190)
print("saved")
print("agreement at lead 0,1,2,3,5,10,20,30:",
      [round(float(acc[np.where(leads == k)[0][0]]), 3) if (leads == k).any() else None for k in [0, 1, 2, 3, 5, 10, 20, 30]])
for r, a, b in zip(rad_vals, pre, post):
    print(f"r={r:g} before={a:.3f} after30={b:.3f}")
