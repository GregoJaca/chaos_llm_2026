"""TEST 7: current-state injection (state.npz, layer 27, r=1e-3).

32 Gaussian directions injected into the CURRENT hidden state at step s;
dh[s, d, t] = 0 for t < s. Same tests:
- gain per direction per s; is a direction strong at s=16 strong at s=150?
  (Spearman rank corr across s -- the number the user asked for)
- shape reliability across s (direction-specific token profiles that persist
  as the injection point moves)
- decay of response with later injection; is the decay direction-specific?
- same for the layer-14 injection (state_mid.npz, 8 directions).
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (annot_heat, load_state, load_state_mid, load_w100,
                    logd, pearson, save_json, savefig, spear)

st = load_state()
dh_s, dm_s, steps = st["dh"], st["dm"], st["steps"]
S, D, T = dh_s.shape
EX = None
cols = ["#d62728", "#1f77b4", "#2ca02c"]
N = {}

# per-(s, d) gains over the responsive range t >= s
g_s = np.zeros((S, D))
for i, s in enumerate(steps):
    g_s[i] = np.median(dh_s[i][:, s:], axis=-1)
exv = g_s[0]
hi, mid, lo = int(np.argmax(exv)), int(np.argsort(exv)[D // 2]), int(np.argmin(exv))
EX = [hi, mid, lo]
N["example_dirs_state"] = EX
N["gain_spread_per_s"] = {f"s={s}": float(g_s[i].max() / g_s[i].min())
                          for i, s in enumerate(steps)}

# rank correlation across s (the requested number)
M = np.eye(S)
for i in range(S):
    for j in range(S):
        if i < j:
            M[i, j] = M[j, i] = spear(g_s[i], g_s[j])
off = M[np.triu_indices(S, 1)]
N["rankcorr_across_s"] = {"min": float(off.min()), "median": float(np.median(off)),
                          "matrix": M.tolist()}
# same for |dm| gain
g_dm = np.zeros((S, D))
for i in range(S):
    g_dm[i] = np.median(np.abs(dm_s[i][:, steps[i]:]), axis=-1)
Mdm = np.eye(S)
for i in range(S):
    for j in range(S):
        if i < j:
            Mdm[i, j] = Mdm[j, i] = spear(g_dm[i], g_dm[j])
N["rankcorr_across_s_dmgain"] = {"min": float(Mdm[np.triu_indices(S,1)].min()),
                                  "median": float(np.median(Mdm[np.triu_indices(S,1)]))}

# shape reliability across s: direction deviations from the per-s common shape
def dev_shape(i):
    s = int(steps[i])
    X = logd(dh_s[i][:, s:])                      # [D, T-s]
    X = X - X.mean(axis=1, keepdims=True)          # remove own gain
    common = X.mean(axis=0)
    return X - common                              # [D, T-s]

devs = [dev_shape(i) for i in range(S)]
Mr = np.eye(S)
for i in range(S):
    for j in range(S):
        if i < j:
            # compare on the shared token range (both start at the later s)
            s0 = int(max(steps[i], steps[j]))
            a = devs[i][:, s0 - steps[i]:].ravel()
            b = devs[j][:, s0 - steps[j]:].ravel()
            Mr[i, j] = Mr[j, i] = pearson(a, b)
N["shape_dev_cross_s_pearson"] = {"min": float(Mr[np.triu_indices(S,1)].min()),
                                  "median": float(np.median(Mr[np.triu_indices(S,1)]))}

# decay with s: direction-specific?
dec = np.log(g_s[-1]) - np.log(g_s[0])             # log gain drop s=16 -> s=150
N["decay"] = {"median_logdrop": float(np.median(dec)), "std_logdrop": float(np.std(dec)),
              "min_dir": int(np.argmin(dec)), "max_dir": int(np.argmax(dec))}
N["spearman_gain16_vs_decay"] = spear(np.log(g_s[0]), dec)

# state_mid (layer 14 injection, 8 dirs)
stm = load_state_mid()
dh_m, steps_m = stm["dh"], stm["steps"]
Dm_ = dh_m.shape[1]
g_m = np.zeros((len(steps_m), Dm_))
for i, s in enumerate(steps_m):
    g_m[i] = np.median(dh_m[i][:, s:], axis=-1)
Mm = np.eye(len(steps_m))
for i in range(len(steps_m)):
    for j in range(len(steps_m)):
        if i < j:
            Mm[i, j] = Mm[j, i] = spear(g_m[i], g_m[j])
N["state_mid_rankcorr"] = {"matrix": Mm.tolist(),
                           "min": float(Mm[np.triu_indices(len(steps_m),1)].min()),
                           "median": float(np.median(Mm[np.triu_indices(len(steps_m),1)]))}
# state vs prompt-perturbation magnitude comparison
zw = load_w100()
g_prompt = np.median(zw["dh_norm"][0, :, -1], axis=-1)   # r=1e-3, layer 27
N["magnitude_ratio_state_over_prompt_r1e-3"] = float(np.median(g_s[0]) / np.median(g_prompt))

print("=== TEST 7 ===")
print("gain spread per s:", N["gain_spread_per_s"])
print("rank corr across s: min %.3f median %.3f" % (off.min(), np.median(off)))
print("matrix:\n", np.array2string(M, precision=3))
print("dm-gain rank corr across s: min %.3f median %.3f" %
      (N["rankcorr_across_s_dmgain"]["min"], N["rankcorr_across_s_dmgain"]["median"]))
print("shape-dev cross-s pearson: min %.3f median %.3f" %
      (N["shape_dev_cross_s_pearson"]["min"], N["shape_dev_cross_s_pearson"]["median"]))
print("decay log-drop: median %.3f std %.3f; spearman(gain16, decay) %.3f" %
      (np.median(dec), np.std(dec), N["spearman_gain16_vs_decay"]))
print("state_mid rank corr matrix:\n", np.array2string(Mm, precision=3))
print("state-vs-prompt magnitude ratio (r=1e-3, layer 27):",
      round(N["magnitude_ratio_state_over_prompt_r1e-3"], 2))

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 3, figsize=(15, 9))

ax = axs[0, 0]
for d in range(D):
    ax.plot(steps, g_s[:, d], color="gray", alpha=0.35, lw=0.8)
lbl = ["highest gain", "median gain", "lowest gain"]
for k, d in enumerate(EX):
    ax.plot(steps, g_s[:, d], "o-", color=cols[k], lw=2, label=f"dir {d} ({lbl[k]})")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("injection step s (state perturbed at token s, r=1e-3, layer 27)")
ax.set_ylabel("gain = median_{t>=s} ||dh_t|| (log scale)")
ax.set_title(f"(a) 32 Gaussian directions, current-state injection\n"
             f"gain spread per s: {N['gain_spread_per_s']['s=16']:.1f}x (s=16) -> "
             f"{N['gain_spread_per_s']['s=150']:.1f}x (s=150)")

ax = axs[0, 1]
annot_heat(ax, M, [f"s={s}" for s in steps], [f"s={s}" for s in steps],
           f"(b) Is a strong direction at one s strong at another?\n"
           f"Spearman rank corr of the 32 gains; min = {off.min():.2f}, "
           f"median = {np.median(off):.2f}",
           vmin=0, vmax=1)

ax = axs[0, 2]
annot_heat(ax, Mr, [f"s={s}" for s in steps], [f"s={s}" for s in steps],
           f"(c) Direction-specific SHAPE reliability across s\n"
           f"Pearson corr of gain-removed log-profiles (shared token range); "
           f"median = {np.median(Mr[np.triu_indices(S,1)]):.2f}",
           vmin=-0.2, vmax=1, cmap="RdYlGn")

ax = axs[1, 0]
i16, i150 = 0, 3
tt = np.arange(T)
for k, d in enumerate(EX):
    ax.plot(tt[16:], logd(dh_s[i16][d, 16:]) - logd(dh_s[i16][d, 16:]).mean(),
            color=cols[k], lw=0.8, alpha=0.6)
    ax.plot(tt[150:], logd(dh_s[i150][d, 150:]) - logd(dh_s[i150][d, 150:]).mean(),
            color=cols[k], lw=1.8, label=f"dir {d}: s=16 (thin) vs s=150 (thick)")
ax.set_xlabel("token position t")
ax.set_ylabel("gain-removed log||dh_t|| (log-units)")
ax.set_title(f"(d) 3 examples: same direction, injection at s=16 vs s=150\n"
             f"curves shifted vertically = direction-specific shape, "
             f"not a shared token profile")

ax = axs[1, 1]
o = np.argsort(dec)
ax.bar(range(D), dec[o], color="gray")
for k, d in enumerate(EX):
    ax.bar(np.where(o == d)[0], dec[d], color=cols[k], label=f"dir {d}")
ax.set_xticks(range(D)); ax.set_xticklabels([f"{d}" for d in o], fontsize=5)
ax.set_xlabel("direction (sorted by decay)")
ax.set_ylabel("log gain drop from s=16 to s=150\n(negative = response fades with later injection)")
ax.set_title(f"(e) Decay is direction-specific: log-drop median {np.median(dec):.2f}, "
             f"spread (std) {np.std(dec):.2f}\n"
             f"spearman(gain at s=16, decay) = {N['spearman_gain16_vs_decay']:.2f}")
ax.legend(fontsize=7)

ax = axs[1, 2]
annot_heat(ax, Mm, [f"s={s}" for s in steps_m], [f"s={s}" for s in steps_m],
           f"(f) Layer-14 injection (state_mid, 8 directions)\n"
           f"rank corr of gains across s; min = {N['state_mid_rankcorr']['min']:.2f}, "
           f"median = {np.median(Mm[np.triu_indices(len(steps_m),1)]):.2f}",
           vmin=0, vmax=1)

fig.suptitle("TEST 7: current-state directions - gain rank across injection steps "
             "(the user-requested number in panel b)", y=1.005)
savefig(fig, "fig7_state.png")
save_json("numbers_t7.json", N)
