"""Control study: core comparison figures, Mamba-130m (negative control) vs
DeepSeek-R1-Distill-Qwen-1.5B (transformer), same prompts, same pipeline.

All figures go to notebook/figures_controls/. Every figure states its
conclusion in the title and defines its axes.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = os.path.join(ROOT, "results", "ctrl_local", "mamba130")
DS = os.path.join(ROOT, "results", "local_data_ds", "w100_layers", "a1.npz")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)

C_MAM = "#0072B2"
C_MAM2 = "#56B4E9"
C_DS = "#D55E00"

mam = {}
for tag in ["txt", "childhood", "cyber"]:
    mam[tag] = np.load(os.path.join(CTRL, f"a1i_{tag}.npz"))
ds = np.load(DS)
print("mamba radii:", mam["txt"]["radii"])
print("ds radii:", ds["radii"])


def band(ax, x, ymed, y10, y90, color, label, lw=1.4):
    ax.fill_between(x, y10, y90, color=color, alpha=0.15)
    ax.plot(x, ymed, color=color, lw=lw, label=label)


def last_layer(z):
    return z["dh_norm"][:, :, -1, :], z["layers"][-1]


# ---------------- c2: teacher-forced response ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
z = mam["txt"]; radii = z["radii"]
for r, c in [(0.1, C_MAM), (1.0, C_MAM2)]:
    ri = int(np.argmin(np.abs(radii - r)))
    a = z["dh_norm"][ri, :, -1, :] / radii[ri]  # [K,T]
    x = np.arange(a.shape[1])
    band(ax, x, np.median(a, 0), np.quantile(a, .1, 0), np.quantile(a, .9, 0),
         c, f"Mamba-130m r={r:g}")
radii_ds = ds["radii"]; ri = int(np.argmin(np.abs(radii_ds - 0.1)))
a = ds["dh_norm"][ri, :, -1] / radii_ds[ri]
x = np.arange(a.shape[1])
band(ax, x, np.median(a, 0), np.quantile(a, .1, 0), np.quantile(a, .9, 0),
      C_DS, "DeepSeek-1.5B r=0.1 (W=100)")
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t)/r = \|\delta h_{L}(t)\|/r$   (last layer)")
ax.set_title("(a) Prompt-perturbation response at the last layer: both models carry\nthe kick; the transformer's response is ~3x larger and holds")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

ax = axs[1]
for z, c, lab in [(mam["txt"], C_MAM, "Mamba-130m r=0.1"), (None, C_DS, None)]:
    if z is None:
        radii_ds = ds["radii"]; ri = int(np.argmin(np.abs(radii_ds - 0.1)))
        a = ds["dh_norm"][ri, :, -1] / radii_ds[ri]
    else:
        ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        a = z["dh_norm"][ri, :, -1, :] / z["radii"][ri]
    a1 = np.median(a, 0)
    ax.plot(np.arange(len(a1)), a1 / a1[1], color=c, lw=1.4, label=lab)
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t)/a(1)$ (median over 20 directions)")
ax.set_title("(b) Normalised to $a(1)$: neither model's response decays to zero;\nSSM slow channels and the KV cache both carry the kick")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2_resp_time.png"), dpi=190); plt.close(fig)

# depth profile
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, tsel, ttl in [(axs[0], 10, "t=10"), (axs[1], 100, "t=100")]:
    z = mam["txt"]; ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    layers = z["layers"]; nl = len(layers)
    g = np.median(z["dh_norm"][ri, :, :, tsel] / z["radii"][ri], axis=0)
    ax.plot(np.asarray(layers) / max(layers), g, "o-", color=C_MAM, label="Mamba-130m (24 layers)")
    ri = int(np.argmin(np.abs(ds["radii"] - 0.1)))
    layers_ds = ds["layers"]
    g = np.median(ds["dh_norm"][ri, :, :, tsel] / ds["radii"][ri], axis=0)
    ax.plot(np.asarray(layers_ds) / max(layers_ds), g, "o-", color=C_DS, label="DeepSeek-1.5B (28 layers)")
    ax.set_yscale("log")
    ax.set_xlabel(r"layer $\ell$ / number of layers")
    ax.set_ylabel(r"$\|\delta h_\ell(t)\|/r$ (median over dirs)")
    ax.set_title("(%s) Depth profile at token t=%s: both grow with depth;\ndepth growth alone is not evidence for chaos" % ("ab"[tsel == 100], tsel))
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2_layers.png"), dpi=190); plt.close(fig)

# lambda(t)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
k = 10
for tag, c, lab in [("txt", C_MAM, "Mamba-130m txt"), ("childhood", C_MAM2, "Mamba-130m childhood"),
                     ("cyber", "#009E73", "Mamba-130m cyber")]:
    z = mam[tag]; ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    ax.plot(t, lam, lw=1.0, color=c, label=lab)
ri = int(np.argmin(np.abs(ds["radii"] - 0.1)))
a = np.median(ds["dh_norm"][ri, :, -1] / ds["radii"][ri], axis=0)
t = np.arange(k, len(a) - k)
lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
ax.plot(t, lam, lw=1.4, color=C_DS, label="DeepSeek-1.5B txt (W=100)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$\lambda_{10}(t)=\log[a(t+10)/a(t-10)]/20$  (nats/token)")
ax.set_title("Local stretching rate $\\lambda_{10}(t)$: transient positive excursions occur in the\nprovably-contracting control too; no sustained positive rate in either")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2_lambda.png"), dpi=190); plt.close(fig)

# linearity collapse
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
z = mam["txt"]; radii = z["radii"]
for ri in range(len(radii)):
    if radii[ri] < 1e-3 or radii[ri] > 5.1:
        continue
    a = np.median(z["dh_norm"][ri, :, -1] / radii[ri], axis=0)
    ax.plot(np.arange(len(a)), a, lw=0.8, alpha=0.8, label=f"r={radii[ri]:g}")
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t)/r$")
ax.set_title("(a) Mamba: response/r collapses for $10^{-3}\\leq r\\leq 5$ (linear regime)")
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3, which="both")
ax = axs[1]
rshow = [0.001, 0.01, 0.1, 0.3, 1, 3]
for z, c, name in [(mam["txt"], C_MAM, "Mamba"), (ds, C_DS, "DeepSeek")]:
    xs, ys = [], []
    for r in rshow:
        ri = int(np.argmin(np.abs(z["radii"] - r)))
        a = np.median(z["dh_norm"][ri, :, -1, 50] / z["radii"][ri])
        xs.append(z["radii"][ri]); ys.append(a)
    ax.loglog(xs, ys, "o-", color=c, label=name)
    s = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    print(name, "alpha(t=50):", round(float(s), 3))
ax.set_xlabel(r"perturbation radius $r$")
ax.set_ylabel(r"$a(t{=}50)/r$ at the last layer")
ax.set_title("(b) Linearity in $r$ at $t=50$: slope $\\alpha\\approx 1$ for both\n(below the fp32 floor, above: tie-dependent)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2_linearity.png"), dpi=190); plt.close(fig)

# long horizon
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
z = np.load(os.path.join(CTRL, "a1i_txt_T1000.npz"))
ax = axs[0]
for r, c in [(0.001, C_MAM2), (0.1, C_MAM), (1.0, "#009E73")]:
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    ax.plot(np.arange(len(a)), a, lw=1.0, color=c, label=f"r={z['radii'][ri]:g}")
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$ (teacher-forced, 1000 tokens)")
ax.set_ylabel(r"$a(t)/r$ (median)")
ax.set_title("(a) Long horizon: no growth over 1000 teacher-forced tokens;\nthe control's response decays slowly (halves in ~500 tokens)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax = axs[1]
for r, c in [(0.1, C_MAM), (1.0, "#009E73")]:
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    ax.plot(t, lam, lw=1.0, color=c, label=f"r={z['radii'][ri]:g}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$\lambda_{10}(t)$ (nats/token)")
ax.set_title("(b) No sustained positive stretching rate over 1000 tokens")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c2_T1000.png"), dpi=190); plt.close(fig)

# ---------------- c3: margins, thresholds, flips ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
ax.plot(np.arange(300), mam["txt"]["margins"], lw=0.9, color=C_MAM, label="Mamba-130m")
ax.plot(np.arange(300), ds["margins"], lw=0.9, color=C_DS, label="DeepSeek-1.5B (W=100)")
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"margin $m_t = z_{(1)}-z_{(2)}$ (top1 - top2 logit gap)")
ax.set_title("(a) Margins along the greedy trajectory, same prompt:\nnear-ties exist in both models")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax = axs[1]
for z, c, name in [(mam["txt"], C_MAM, "Mamba-130m"), (ds, C_DS, "DeepSeek")]:
    m = np.sort(z["margins"])
    ax.semilogy(np.arange(1, len(m) + 1) / len(m), m, color=c, label=name)
ax.set_xlabel(r"empirical CDF of $m_t$")
ax.set_ylabel(r"$m_t$")
ax.set_title("(b) Margin CDFs: quantitatively similar landscapes\n(min 0.055 vs 0.020, median 6.5 vs 4.8)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_margins.png"), dpi=190); plt.close(fig)

# danger ladder: per-direction r* from the SIGNED threshold rule,
# measured at a sub-flip radius. Two panels: txt and cyber prompts.
def signed_rstar(z, r):
    """r*(u) = min_t m_t*r/max(-dm_r(t,u),0) and the full signed r_c."""
    ri = int(np.argmin(np.abs(z["radii"] - r)))
    m = z["margins"][None, :]  # [1,T]
    neg = np.maximum(-z["dm"][ri], 0.0)  # [K,T]
    rc = np.where(neg > 0, m * float(z["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc.min(axis=1), rc, float(z["radii"][ri])

ds_cyber = np.load(os.path.join(ROOT, "results", "local_data_ds", "a1_cyber_w100", "a1.npz"))

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
for ax, mz, dz, ptitle in [
        (axs[0], mam["txt"], ds, "txt (interstellar)"),
        (axs[1], mam["cyber"], ds_cyber, "cyber")]:
    rs_m, _, rmeas = signed_rstar(mz, 1.0)
    rs_m = np.sort(rs_m)
    ax.step(np.arange(1, len(rs_m) + 1), rs_m, where="post", color=C_MAM,
            label=f"Mamba-130m (measured at r={rmeas:g})")
    rs_d, _, rmeas_d = signed_rstar(dz, 0.1)
    rs_d = np.sort(rs_d)
    ax.step(np.arange(1, len(rs_d) + 1), rs_d, where="post", color=C_DS,
            label=f"DeepSeek-1.5B (measured at r={rmeas_d:g})")
    ax.axhline(rs_m[0], color=C_MAM, ls=":", lw=1)
    ax.axhline(rs_d[0], color=C_DS, ls=":", lw=1)
    ax.annotate(f"$r^*$={rs_m[0]:.3g}", (1, rs_m[0]), color=C_MAM, fontsize=9, va="bottom")
    ax.annotate(f"$r^*$={rs_d[0]:.3g}", (1, rs_d[0]), color=C_DS, fontsize=9, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("direction rank (sorted)")
    ax.set_ylabel(r"$r^*(u)=\min_t r_c^-(t,u)$  (signed threshold)")
    ax.set_title(f"{ptitle}: the signed rule transfers; $r^*$ is governed\nby the margin near-ties, not by the architecture")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_ladder.png"), dpi=190); plt.close(fig)

# obs vs pred across all free runs (pred = SIGNED threshold rule, first crossing)
rows = []
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    name = os.path.basename(f)[:-4]  # fr_<tag>_r<r>
    tag = name.split("_")[1]
    r = float(name.split("_r")[1])
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    a1 = mam.get(tag)
    if a1 is None:
        continue
    _, rc, r_meas = signed_rstar(a1, r)
    hit = rc <= r
    pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    K = len(obs)
    n_obs = int((obs >= 0).sum()); n_pred = int((pred_T >= 0).sum())
    both = (obs >= 0) & (pred_T >= 0)
    exact = int((obs[both] == pred_T[both]).sum())
    fa = int(((pred_T >= 0) & (obs < 0)).sum())
    miss = int(((pred_T < 0) & (obs >= 0)).sum())
    rows.append(dict(name=name, r=r, K=K, n_obs=n_obs, n_pred=n_pred, exact=exact,
                     fa=fa, miss=miss))
print("\nobs/pred (signed threshold rule, first crossing):")
print("%-24s %4s %5s %5s %5s %4s %4s" % ("run", "r", "obs", "pred", "exact", "FA", "miss"))
for x in rows:
    print("%-24s %4g %5d %5d %5d %4d %4d" % (x["name"], x["r"], x["n_obs"], x["n_pred"],
                                             x["exact"], x["fa"], x["miss"]))
json.dump(rows, open(os.path.join(OUT, "..", "..", "results", "ctrl_local", "obspred_mamba130.json"), "w"), indent=1)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
xs = np.arange(len(rows))
w = 0.38
ax.bar(xs - w / 2, [x["n_obs"] for x in rows], w, color=C_MAM, label="observed flips (free run)")
ax.bar(xs + w / 2, [x["n_pred"] for x in rows], w, color="#999999", label="predicted flips (threshold rule)")
ax.set_xticks(xs)
ax.set_xticklabels([f"{x['name'].split('_')[1]}\n{x['r']:g}" for x in rows], fontsize=6, rotation=0)
ax.set_xlabel("prompt / free-run radius")
ax.set_ylabel("flipping directions (out of 20)")
ax.set_title("Observed vs predicted flips, Mamba-130m, 18 free runs:\nthe same signed-threshold rule predicts the control's flips")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_obspred.png"), dpi=190); plt.close(fig)

# survival: token agreement after the first flip
fig, ax = plt.subplots(figsize=(8.6, 4.6))
agrees = []
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    gen_rows = list(z["gen_ids"])
    nb = len(gen_rows)
    K = len(obs)
    nchunk = int(np.ceil(K / 6))
    for ri in range(len(z["radii"])):
        for b in range(nb):
            for j in range(6):
                di = b * 6 + j
                if di >= K or obs[di] < 0:
                    continue
                ref = gen_rows[ri * nb + b][0]
                row = gen_rows[ri * nb + b][j + 1]
                T0 = int(obs[di])
                n = min(len(ref) - 1, len(row) - 1)
                # agreement in windows after the flip
                for w0 in range(T0 + 1, min(n, T0 + 60), 5):
                    seg_ref = ref[w0 + 1:w0 + 6]
                    seg_row = row[w0 + 1:w0 + 6]
                    if len(seg_ref) == 0:
                        continue
                    agrees.append((w0 - T0, float((seg_ref == seg_row).mean())))
if agrees:
    arr = np.array(agrees)
    for lag in sorted(set(arr[:, 0].astype(int))):
        sel = arr[arr[:, 0] == lag, 1]
        ax.plot(lag, np.median(sel), "o", ms=4, color=C_MAM)
        ax.errorbar(lag, np.median(sel), yerr=[[np.median(sel) - np.quantile(sel, .1)],
                    [np.quantile(sel, .9) - np.median(sel)]], color=C_MAM, alpha=0.5, lw=1)
    # re-sync: full remainder equality after the flip
    resync = 0
    nflip = 0
    for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
        z = np.load(f, allow_pickle=True)
        obs = z["obs"][0]; gen_rows = list(z["gen_ids"]); nb = len(gen_rows); K = len(obs)
        for b in range(nb):
            for j in range(6):
                di = b * 6 + j
                if di >= K or obs[di] < 0:
                    continue
                nflip += 1
                ref = gen_rows[b][0]; row = gen_rows[b][j + 1]
                T0 = int(obs[di])
                if np.array_equal(row[T0 + 1:], ref[T0 + 1:]):
                    resync += 1
    ax.set_xlabel(r"tokens after the first flip")
    ax.set_ylabel("token agreement with reference")
    ax.set_title("After the first flip the control also never re-synchronises\n(%d/%d flipped runs re-sync)" % (resync, nflip))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_survival.png"), dpi=190); plt.close(fig)

# flip structure: are flips at near-ties? response level at the flip step?
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
flip_m = []
allflips = 0
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    name = os.path.basename(f)[:-4]
    tag = name.split("_")[1]
    if tag not in mam:
        continue
    r = float(name.split("_r")[1])
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    a1 = mam[tag]
    ri = int(np.argmin(np.abs(a1["radii"] - r)))
    for di in range(len(obs)):
        if obs[di] >= 0:
            flip_m.append((int(obs[di]), float(a1["margins"][int(obs[di])]), r))
            allflips += 1
m = np.sort([x[1] for x in flip_m])
ax.semilogy((np.arange(len(m)) + 1) / (len(m) + 1), m, color=C_MAM, label="margins at flip steps")
mtxt = np.sort(mam["txt"]["margins"])
ax.semilogy((np.arange(len(mtxt)) + 1) / (len(mtxt) + 1), mtxt, color="#888", lw=0.8,
            label="all margins (txt prompt)")
ax.set_xlabel("empirical CDF")
ax.set_ylabel(r"$m_t$ at the flip step")
ax.set_title("(a) %d observed flips: they sit on small margins\n(flip = near-tie crossing, same as the transformer)" % allflips)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax = axs[1]
# response level at flip: a(t)/a_peak
levs = []
for f in sorted(glob.glob(os.path.join(CTRL, "fr_*.npz"))):
    name = os.path.basename(f)[:-4]
    tag = name.split("_")[1]
    if tag not in mam:
        continue
    r = float(name.split("_r")[1])
    z = np.load(f, allow_pickle=True)
    obs = z["obs"][0]
    a1 = mam[tag]
    ri = int(np.argmin(np.abs(a1["radii"] - r)))
    a = np.median(a1["dh_norm"][ri, :, -1] / a1["radii"][ri], axis=0)
    for di in range(len(obs)):
        if obs[di] >= 0:
            t = int(obs[di])
            levs.append(a[t] / (a[:t + 1].max() if t > 0 else a[0]))
ax.hist(levs, bins=20, color=C_MAM, alpha=0.8)
ax.set_xlabel(r"$a(t_{\rm flip})/\max_{s\leq t} a(s)$ (response level at the flip)")
ax.set_ylabel("flips")
ax.set_title("(b) Response level at the flip step: immediate flips (level ~1)\nvs delayed flips at later close calls (level < 0.3)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c3_flipstruct.png"), dpi=190); plt.close(fig)

print("wrote figures to", OUT)
