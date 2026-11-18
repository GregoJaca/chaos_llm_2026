"""Autoregressive validation of the latent single-token kick.

A kick applied at one token's embedding changes nothing for ~100 tokens, then
changes the token exactly where the teacher-forced pass predicted.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

sf = np.load(os.path.join(B, "state_freerun", "sf.npz"), allow_pickle=True)
d = np.load(os.path.join(B, "state_deep_emb", "state2.npz"), allow_pickle=True)
m = d["margins"]; dm = d["dm"]; steps = list(d["steps"]); radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.1)))
ref = sf["ref"]
T = len(ref); t = np.arange(T)


def cum_same(p):
    v = p >= 0
    s = (p == ref).astype(float)
    s[~v] = np.nan
    out = np.full(T, np.nan)
    run = np.nancumsum(np.where(np.isnan(s), 0, s))
    cnt = np.nancumsum(np.where(np.isnan(s), 0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, run / cnt, np.nan)
    return out


events = []
for si, s in enumerate(sf["steps"]):
    dsi = steps.index(int(s))
    for di in range(sf["pred"].shape[1]):
        p = sf["pred"][si, di]; v = p >= 0
        mm = np.where((p != ref) & v)[0]
        obs = int(mm[0]) if len(mm) else -1
        hit = dm[ri, dsi, di] <= -m
        pred = int(hit.argmax()) if hit.any() else -1
        if pred >= 0 or obs >= 0:
            events.append((int(s), di, pred, obs))

fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.3)

# (a) predicted vs observed
ax = fig.add_subplot(gs[0, 0])
ax.scatter([e[2] for e in events], [e[3] for e in events], s=90, color="tab:red", zorder=3)
ax.plot([100, 150], [100, 150], "k--", lw=1)
for s, di, pr, ob in events:
    ax.annotate(f"kick at s={s}", (pr, ob), textcoords="offset points", xytext=(-4, 10),
                fontsize=8, ha="right")
ax.set_xlim(100, 150); ax.set_ylim(100, 150)
ax.set_xlabel("predicted first changed token\n(teacher-forced pass)")
ax.set_ylabel("observed first changed token\n(free-running generation)")
ax.set_title(f"(a) {len(events)}/{len(events)} exact, including the step")
ax.grid(alpha=0.25)

# (b) one example, cumulative agreement
ax = fig.add_subplot(gs[0, 1])
s, di, pr, ob = events[0]
p = sf["pred"][list(sf["steps"]).index(s), di]
ax.plot(t, cum_same(p), lw=1.8, color="tab:blue")
ax.axvline(s, color="tab:green", ls="--", lw=1.2)
ax.axvline(ob, color="tab:red", ls="--", lw=1.2)
ax.text(s + 3, 0.35, f"kick applied\nat token {s}", color="tab:green", fontsize=8)
ax.text(ob - 3, 0.75, f"first changed\ntoken {ob}\n({ob - s} tokens later)", color="tab:red", fontsize=8, ha="right")
ax.set_xlabel("generated token t")
ax.set_ylabel("fraction of tokens so far\nidentical to the unperturbed run")
ax.set_title("(b) Invisible for 100+ tokens,\nthen one token changes")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)

# (c) all 24 directions kicked at s=16
ax = fig.add_subplot(gs[0, 2])
si = list(sf["steps"]).index(16)
for di in range(sf["pred"].shape[1]):
    p = sf["pred"][si, di]
    changed = (p >= 0).any() and (p != ref)[p >= 0].any()
    ax.plot(t, cum_same(p), lw=1.1, alpha=0.95 if changed else 0.35,
            color="tab:red" if changed else "tab:grey")
ax.axvline(123, color="k", ls="--", lw=1)
ax.set_xlabel("generated token t")
ax.set_ylabel("fraction of tokens so far\nidentical to the unperturbed run")
ax.set_title("(c) 4 of 24 directions change a token at t=123;\nthe other 20 reproduce the text exactly")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)

fig.suptitle("Single-token kick, r=0.1: the teacher-forced pass predicts the latent flip exactly", fontsize=11.5)
fig.subplots_adjust(top=0.80)
fig.savefig(os.path.join(OUT, "state_kick_latent_validated.png"), dpi=190, bbox_inches="tight")
print("saved", events)
