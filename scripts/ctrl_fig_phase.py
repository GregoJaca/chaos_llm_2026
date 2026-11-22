"""Part 6: random-init Qwen2 (DeepSeek-1.5B architecture), order-to-chaos
phase sweep over the residual-update init gain g.

Reads results/ctrl/randqwen_g{G}/ (a1i_txt + fr_r* + state) and produces
the phase figures. Safe to re-run at any time; skips missing gains.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTRL = os.path.join(ROOT, "results", "ctrl_local", "randqwen")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
os.makedirs(OUT, exist_ok=True)


def signed_rc(zz, r):
    ri = int(np.argmin(np.abs(zz["radii"] - r)))
    m = zz["margins"][None, :]
    neg = np.maximum(-zz["dm"][ri], 0.0)
    rc = np.where(neg > 0, m * float(zz["radii"][ri]) / np.maximum(neg, 1e-30), np.inf)
    return rc, float(zz["radii"][ri])


gains = sorted(
    [float(os.path.basename(d).split("_g")[1]) for d in glob.glob(os.path.join(CTRL, "randqwen_g*"))])
print("gains present:", gains)

rows = []
for g in gains:
    d = os.path.join(CTRL, f"randqwen_g{g}")
    a1 = os.path.join(d, "a1i_txt", "a1.npz")
    if not os.path.exists(a1):
        continue
    z = np.load(a1, allow_pickle=True)
    row = dict(gain=g)
    # response level + lambda at r=0.1 (or the closest available)
    ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    row["a1"] = float(a[1]); row["a100"] = float(a[100]); row["a299"] = float(a[299])
    row["lam_mean"] = float(lam.mean()); row["lam_max"] = float(lam.max())
    rc, rmeas = signed_rc(z, 0.1)
    row["rstar"] = float(np.min(rc))
    row["margin_min"] = float(z["margins"].min())
    # depth profile from the state2 probe
    st = os.path.join(d, "state", "state2.npz")
    if os.path.exists(st):
        s = np.load(st, allow_pickle=True)
        dh = s["dh"]  # [nr, ns, K, nl, T]
        smeta = os.path.join(d, "state", "meta.json")
        if os.path.exists(smeta):
            layers = json.load(open(smeta))["layers"]
        else:
            layers = list(range(0, dh.shape[3] * 2 - 1, 2))
        # depth growth at s=20, r=0.05, t=10: log slope over layers 1..27
        prof = np.median(dh[0, 0, :, :, 10], axis=0)  # [nl]
        lo = prof[1:-1].min() + 1e-30
        xs = np.log(np.maximum(np.asarray(layers[1:-1], float) + 1, 1.0))
        ys = np.log(np.maximum(prof[1:-1], lo))
        row["lam_depth"] = float(np.polyfit(xs, ys, 1)[0])
        row["depth_profile"] = prof.tolist()
        row["depth_layers"] = layers
    # free-run scorecards
    sc = []
    for f in sorted(glob.glob(os.path.join(d, "fr_r*", "free_run.npz"))):
        fr = np.load(f, allow_pickle=True)
        rr = float(np.atleast_1d(fr["radii"])[0])
        obs = fr["obs"][0]
        rc, _ = signed_rc(z, rr)
        hit = rc <= rr
        pred_T = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        both = (obs >= 0) & (pred_T >= 0)
        sc.append(dict(r=rr, obs=int((obs >= 0).sum()),
                       exact=int((obs[both] == pred_T[both]).sum()),
                       fa=int(((pred_T >= 0) & (obs < 0)).sum()),
                       miss=int(((pred_T < 0) & (obs >= 0)).sum())))
    row["score"] = sc
    rows.append(row)
    print(f"g={g:g}: a/r {row['a1']:.3g}->{row['a299']:.3g}, lam {row['lam_mean']:+.4f}/{row['lam_max']:+.4f}, "
          f"r* {row['rstar']:.3g}, margin_min {row['margin_min']:.3g}, "
          f"{'lam_depth %+.3f' % row['lam_depth'] if 'lam_depth' in row else 'no state2'}, fr={sc}")

json.dump(rows, open(os.path.join(CTRL, "phase_rows.json"), "w"), indent=1, default=float)

if not rows:
    raise SystemExit("no data yet")

# ---- phase figure: lambda depth + lambda time vs gain ----
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
have = [r for r in rows if "lam_depth" in r]
if have:
    ax.plot([r["gain"] for r in have], [r["lam_depth"] for r in have], "o-", color="#333")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"residual-update init gain $g$ (std $=0.02\,g$)")
    ax.set_ylabel(r"$\lambda_{\rm depth} = d\log\|\delta h_\ell\|/d\ell$ (t=10, kick at s=20)")
    ax.set_title("(a) Depth-wise growth rate vs gain: the mean-field\norder parameter of the random transformer")
    ax.grid(alpha=0.3, which="both")
ax = axs[1]
for r in rows:
    ax.plot(r["gain"], r["lam_mean"], "o", color="#0072B2")
    ax.plot(r["gain"], r["lam_max"], "s", color="#D55E00", alpha=0.7)
ax.axhline(0, color="k", lw=1)
ax.set_xscale("log")
ax.set_xlabel(r"residual-update init gain $g$")
ax.set_ylabel(r"$\lambda_{10}(t)$ along tokens: mean (blue), max (red)")
ax.set_title("(b) Along-token stretching rate vs gain:\nmaxima grow with $g$ even while the mean stays ~0")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c6_phase.png"), dpi=190); plt.close(fig)

# ---- response curves per gain ----
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for r in rows:
    d = os.path.join(CTRL, f"randqwen_g{r['gain']}", "a1i_txt", "a1.npz")
    z = np.load(d, allow_pickle=True)
    ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    a = np.median(z["dh_norm"][ri, :, -1] / z["radii"][ri], axis=0)
    ax.plot(np.arange(len(a)), a, lw=1.0, label=f"g={r['gain']:g}")
ax.set_yscale("log")
ax.set_xlabel(r"generated token $t$ (teacher-forced, $r=0.1$)")
ax.set_ylabel(r"$a(t)/r$ at the last layer")
ax.set_title("Random-init transformer, sweep of the update gain $g$:\nteacher-forced response goes from decaying to growing")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c6_response.png"), dpi=190); plt.close(fig)

# ---- obs/pred scorecard + r* ----
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs[0]
labels, exacts, fas, misss, obss = [], [], [], [], []
for r in rows:
    for s in r["score"]:
        labels.append(f"g{r['gain']:g}\n{s['r']:g}")
        exacts.append(s["exact"]); fas.append(s["fa"]); misss.append(s["miss"]); obss.append(s["obs"])
xs = np.arange(len(labels))
ax.bar(xs, obss, 0.6, color="#bbb", label="observed flips")
ax.bar(xs, exacts, 0.6, color="#009E73", label="predicted exactly")
ax.bar(xs, misss, 0.6, bottom=exacts, color="#f4a582", label="misses")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
ax.set_xlabel("gain / free-run radius")
ax.set_ylabel("direction-runs (of 20)")
ax.set_title("Greedy argmax on top of a random transformer: flips occur in the\nORDERED phase too - the threshold alone suffices (0 false alarms)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax = axs[1]
ax.plot([r["gain"] for r in rows], [r["rstar"] for r in rows], "o-", color="#333")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"gain $g$")
ax.set_ylabel(r"signed flip threshold $r^*$")
ax.set_title("Per-gain flip threshold: set by the random margin landscape,\nroughly independent of the phase")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c6_obspred.png"), dpi=190); plt.close(fig)
print("wrote", OUT)
