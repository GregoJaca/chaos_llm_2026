import numpy as np, os, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_forms"

d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"]; nh0 = d["norm_h0"]
print("radii:", radii)
print("dh at t=0 per radius (median over dirs, layer27):", np.median(dh[:, :, -1, 0], axis=1))
print("dh at t=1:", np.median(dh[:, :, -1, 1], axis=1))
print("norm_h0 layer27 head:", nh0[-1, :8])
print("margins at 16,123,143:", d["margins"][[16, 123, 143]])

# median curves at a few radii, layer 27 and layer 7
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
t = np.arange(dh.shape[-1])
for r in [1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0]:
    i = int(np.argmin(np.abs(radii - r)))
    for ax, li in zip(axs, [-1, -4]):
        y = np.median(dh[i, :, li, :], axis=0)
        ax.semilogy(t, np.maximum(y, 1e-12), label=f"r={r:g}", lw=1)
axs[0].set_title("layer 27, median over 20 dirs"); axs[1].set_title("layer 7")
for ax in axs: ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_xlabel("t")
fig.suptitle("w100_layers median response curves")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "explore_median.png"), dpi=130); plt.close(fig)

# single-direction curves (noise level) at r=0.1
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
i = int(np.argmin(np.abs(radii - 0.1)))
for di in range(6):
    axs[0].semilogy(t, np.maximum(dh[i, di, -1, :], 1e-12), lw=.8)
    axs[1].semilogy(t, np.maximum(dh[i, di, 1, :], 1e-12), lw=.8)  # layer 1
axs[0].set_title("r=0.1 layer27, 6 directions"); axs[1].set_title("r=0.1 layer1")
for ax in axs: ax.grid(alpha=.3); ax.set_xlabel("t")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "explore_dirs.png"), dpi=130); plt.close(fig)

# fraction of zeros/negatives
for r in [1e-3, 1e-2, 0.1, 0.3]:
    i = int(np.argmin(np.abs(radii - r)))
    print(f"r={r}: min dh {dh[i].min():.3g}, zeros {np.mean(dh[i]<=0)*100:.2f}%")

# window_fixed check
for w in [100, 150, 200, 300, 1000]:
    dz = np.load(os.path.join(BASE, "window_fixed", f"w{w}", "a1.npz"))
    g = dz["gen_ids"]
    print(f"w{w}: gen_ids[:8] = {g[:8]}, T={len(g)}, radii={dz['radii']}, layers={dz['layers']}")
g100 = np.load(os.path.join(BASE, "window_fixed", "w100", "a1.npz"))["gen_ids"]
g200 = np.load(os.path.join(BASE, "window_fixed", "w200", "a1.npz"))["gen_ids"]
g1000 = np.load(os.path.join(BASE, "window_fixed", "w1000", "a1.npz"))["gen_ids"]
print("agree w100 vs w200:", np.mean(g100 == g200), " w100 vs w1000:", np.mean(g100 == g1000))
print("w100_layers gen vs window_fixed w100 gen:",
      np.mean(d["gen_ids"][:300] == g100[:300]))

# fr files
for f in ["fr_w100_large.npz", "fr_w100_mid.npz", "fr_childhood.npz", "fr_cyber.npz"]:
    z = np.load(os.path.join(BASE, f), allow_pickle=True)
    print("===", f, {k: (z[k].shape, z[k].dtype) for k in z.files})
    if "radii" in z.files: print("   radii:", z["radii"])
    if "obs" in z.files:
        obs = z["obs"]
        for ri in range(obs.shape[0]):
            o = obs[ri]; print(f"   obs r={z['radii'][ri]}: flips={np.sum(o>=0)}/{len(o)} median={np.median(o[o>=0]) if np.any(o>=0) else 'NA'}")
    gi = z["gen_ids"] if "gen_ids" in z.files else None
    if gi is not None:
        g0 = np.asarray(gi[0]); ref0 = np.asarray(z["ref_ids"][0])
        print("   gen_ids[0] shape:", g0.shape, " ref0 shape:", ref0.shape)
        print("   gen row0 == ref0 ?", np.array_equal(g0[0], ref0), " row1==ref0?", np.array_equal(g0[1], ref0))
