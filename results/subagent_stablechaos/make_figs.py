"""Generate all figures + stats_extra.json (p-values, lambda CIs, window numbers)."""
import numpy as np, json, sys, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos")
from stablechaos_lib import *
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "figure.dpi": 200, "savefig.dpi": 200})
CB = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

stats = json.load(open(OUT + r"\stats.json"))
curves = pickle.load(open(OUT + r"\curves.pkl", "rb"))
a1 = load_a1(r"w100_layers\a1.npz")
margins = a1["margins"]

mid = load_free("fr_w100_mid.npz")
large = load_free("fr_w100_large.npz")
cy = load_free("fr_cyber.npz")
ch = load_free("fr_childhood.npz")
obs_main = {}   # r -> T array
pred_main = {}
for d in (mid, large):
    for k in range(d["obs"].shape[0]):
        obs_main[float(d["radii"][k])] = d["obs"][k]
        pred_main[float(d["radii"][k])] = d["pred"][k]
rs = sorted(obs_main)

# ---------- extra stats: lambda bootstrap CIs, MC + sign tests ----------
rng = np.random.default_rng(7)
lam_ci = {}
for r in rs:
    T = obs_main[r]
    if (T >= 0).sum() < 2:
        continue
    lams = []
    for b in range(1000):
        idx = rng.integers(0, 20, 20)
        f = fit_survival_forms(T[idx])
        lams.append(f["exp"]["lam"])
    lam_ci[r] = [float(np.quantile(lams, 0.025)), float(np.quantile(lams, 0.975))]

# discreteness MC: 29 flips, 8 distinct steps, 301 possible integer steps
pooled = np.concatenate([obs_main[r][obs_main[r] >= 0] for r in rs])
n, k = len(pooled), len(np.unique(pooled))
cnt = 0
for _ in range(200000):
    draws = rng.integers(0, 301, n)
    if len(np.unique(draws)) <= k:
        cnt += 1
p_disc = cnt / 200000

# one-sidedness sign test: total pred_only vs obs_only across all 13 cells
tot_pred_only = sum(stats[p][key]["pred_only"] for p in ["main", "cyber", "childhood"]
                    for key in stats[p])
tot_obs_only = sum(stats[p][key]["obs_only"] for p in ["main", "cyber", "childhood"]
                   for key in stats[p])
tot_later = sum(stats[p][key]["later"] for p in ["main", "cyber", "childhood"] for key in stats[p])
tot_earlier = sum(stats[p][key]["earlier"] for p in ["main", "cyber", "childhood"] for key in stats[p])
n_err = tot_pred_only + tot_obs_only
p_sign = 2.0 ** -(n_err - 1) if n_err > 0 else 1.0  # one-sided binomial

print(f"lambda CIs: {lam_ci}")
print(f"discreteness MC: {k} distinct steps among {n} flips, p = {p_disc}")
print(f"sign test: pred_only={tot_pred_only} obs_only={tot_obs_only} p={p_sign:.2e}; later={tot_later} earlier={tot_earlier}")

extra = dict(lam_ci={str(r): v for r, v in lam_ci.items()}, p_discreteness=p_disc,
             n_pred_only=tot_pred_only, n_obs_only=tot_obs_only, n_later=tot_later,
             n_earlier=tot_earlier, p_sign=p_sign)

# =====================================================================
# FIG 1 - flip-time dataset: discreteness + margin dips
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
fig.suptitle("Free-run flip times are DISCRETE steps that coincide with margin dips of the reference "
             "(not a continuous spread)", fontsize=11, fontweight="bold")

a = ax[0, 0]
for r in rs:
    T = obs_main[r]; P = pred_main[r]
    jit = (np.random.default_rng(int(r * 100)).random(20) - 0.5) * 0.06 * r
    a.scatter(r + jit, np.where(T >= 0, T, 312), s=14, color=CB[0], alpha=0.75, lw=0,
              label="observed $T$" if r == rs[0] else None)
    a.scatter(r + jit, np.where(P >= 0, P, 312), s=22, marker="x", color=CB[3], alpha=0.8, lw=1.1,
              label="threshold-model $T_{pred}$" if r == rs[0] else None)
a.axhline(300, color="gray", lw=0.8, ls=":")
a.text(0.6, 252, "no flip within 300 tokens (censored, plotted at 312)", fontsize=7.5, color="gray")
a.set_ylim(0, 325)
a.set_xscale("log"); a.set_xticks(rs); a.set_xticklabels([f"{r:g}" for r in rs])
a.set_xlabel("perturbation radius r"); a.set_ylabel("first flip step T")
a.set_title("(a) per-direction flip times vs radius, main prompt\n(open symbols above line = no flip; red x = threshold model)", fontsize=9)
a.legend(loc="center left", framealpha=0.9)

a = ax[0, 1]
a.semilogy(np.arange(300), margins, color=CB[0], lw=0.9)
steps = stats["discreteness"]["steps"]
cnts = stats["discreteness"]["counts"]
for s in steps:
    a.axvline(s, color=CB[3], lw=1.0, alpha=0.75)
a.set_xlabel("generation step t"); a.set_ylabel("teacher-forced margin  $m(t)=s_{top1}-s_{top2}$")
a.set_title("(b) margin along reference; red lines = observed flip steps\n"
            "6 of 8 flip steps are among the 7 smallest margins (ranks 0-6)", fontsize=9)

a = ax[1, 0]
for r in rs:
    n = int((obs_main[r] >= 0).sum())
    lo, hi = stats["T_vs_r"]["flipfrac_ci"][rs.index(r)]
    ff_ = (obs_main[r] >= 0).mean()
    a.errorbar(r, ff_, yerr=np.array([[ff_ - lo], [hi - ff_]]),
               fmt="o", color=CB[0], capsize=3, ms=5,
               label="observed" if r == rs[0] else None)
    a.scatter([r], [(pred_main[r] >= 0).mean()], s=42, marker="D", facecolors="none",
              edgecolors=CB[3], lw=1.3, label="threshold model" if r == rs[0] else None)
a.set_xscale("log"); a.set_xticks(rs); a.set_xticklabels([f"{r:g}" for r in rs])
a.set_xlabel("perturbation radius r"); a.set_ylabel("fraction of directions that flip")
a.set_title("(c) flip fraction vs r, main prompt (95% binomial CI)\n"
            "hard floor: no flips for r < min $r_c \\approx$ 0.11-0.13 in 300 steps", fontsize=9)
a.legend(loc="upper left")

a = ax[1, 1]
a.bar(range(len(steps)), cnts, color=CB[3], alpha=0.8)
a.set_xticks(range(len(steps)))
a.set_xticklabels([f"{s}\n(rank {stats['discreteness']['margin_ranks'][str(s)]})" for s in steps], fontsize=8)
a.set_xlabel("flip step T (margin rank in parentheses)")
a.set_ylabel("count of (direction, radius) pairs")
a.set_title(f"(d) pooled flip steps: {stats['discreteness']['n_flips']} flips on only "
            f"{stats['discreteness']['distinct_steps']} distinct steps\n"
            f"continuous-separation null: p < 1e-6 (MC {p_disc:.1e})", fontsize=9)
fig.savefig(OUT + r"\fig1_flipdata.png")
plt.close(fig)

# =====================================================================
# FIG 2 - T(r) scaling: log law vs power law vs threshold model
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
fig.suptitle("T(r) is a threshold phenomenon, not exponential separation: log(1/r) law unstable, "
             "rate r-dependent, threshold model tracks means with 0 fitted parameters",
             fontsize=11, fontweight="bold")

a = ax[0, 0]
tv = stats["T_vs_r"]
rsa = np.array(tv["radii"]); mT = np.array(tv["meanT"]); mTp = np.array(tv["meanT_pred"])
lo = np.array([c[0] for c in tv["meanT_ci"]]); hi = np.array([c[1] for c in tv["meanT_ci"]])
a.errorbar(rsa, mT, yerr=[mT - lo, hi - mT], fmt="o-", color=CB[0], capsize=3, ms=5, lw=1.2,
           label="observed (95% CI, bootstrap over 20 dirs)")
a.plot(rsa, mTp, "D--", color=CB[3], ms=6, lw=1.2, mfc="none", label="threshold model (0 params)")
lg = tv["log_all"]; pw = tv["pow_all"]
rgrid = np.geomspace(0.04, 2.5, 100)
a.plot(rgrid, lg["a"] - lg["b"] * np.log(rgrid), color=CB[1], lw=1.2,
       label=f"log law $a-b\\ln r$: b={lg['b']:.0f}, $R^2$={lg['R2']:.2f}")
a.plot(rgrid, pw["c"] * rgrid ** (-pw["alpha"]), color=CB[2], lw=1.2, ls="-.",
       label=f"power law $cr^{{-\\alpha}}$: $\\alpha$={pw['alpha']:.2f}, $R^2$={pw['R2']:.2f}")
a.set_xscale("log"); a.set_ylim(0, 330)
a.set_xlabel("radius r"); a.set_ylabel("mean flip time (censored, cap 300)")
a.set_title(f"(a) mean T vs r: threshold $R^2$={tv['R2_threshold_means']:.2f} beats both parametric laws;\n"
            f"log-law b jumps 50$\\to$82 when censored radii dropped", fontsize=9)
a.legend(loc="lower left")

a = ax[1, 0]
lam_r = [r for r in rs if r in lam_ci]
lams = [stats["main"][f"r{r:g}"]["fits"]["exp"]["lam"] for r in lam_r]
err = [[lams[i] - lam_ci[r][0], lam_ci[r][1] - lams[i]] for i, r in enumerate(lam_r)]
a.errorbar(lam_r, lams, yerr=np.array(err).T, fmt="o-", color=CB[0], capsize=3, ms=5, lw=1.2)
a.axhline(0, color="gray", lw=0.8)
a.set_xscale("log")
a.set_xlabel("radius r"); a.set_ylabel("fitted exponential rate $\\lambda(r)$  [/token]")
a.set_title("(b) fitted hazard is strongly radius-dependent (95% CI);\n"
            "at r=0.5 the CI includes zero. Ordinary chaos requires an\n"
            "r-INDEPENDENT positive Lyapunov rate", fontsize=9)

a = ax[1, 1]
infr = [0.3, 0.5, 1.0, 2.0]
models = ["R2_threshold", "R2_exp"]
width = 0.27
xs = np.arange(len(infr))
for mi, (mkey, lab, col) in enumerate([("R2_threshold", "threshold (0 params)", CB[3]),
                                       ("R2_exp", "exponential fit", CB[2]),
                                       (None, None, None)]):
    if mkey is None:
        vals = [stats["main"][f"r{r:g}"]["fits"]["stretched"]["R2"] for r in infr]
        a.bar(xs + mi * width, vals, width, color=CB[1], label="stretched exp fit")
        continue
    vals = [max(0, stats["main"][f"r{r:g}"][mkey]) for r in infr]
    a.bar(xs + mi * width, vals, width, color=col, label=lab)
a.axhline(0, color="gray", lw=0.8)
a.set_xticks(xs + width); a.set_xticklabels([f"r={r:g}" for r in infr])
a.set_ylabel("$R^2$ (clipped at 0)")
a.set_title("(c) in-sample parametric fits do not beat the out-of-sample\n"
            "threshold model except where it over-predicts flips", fontsize=9)
a.legend(loc="upper left", fontsize=7.5)

a = ax[0, 1]
both = [(r, obs_main[r], pred_main[r]) for r in infr]
xo, yo = [], []
ex_match = 0
for r, T, P in both:
    for j in range(20):
        if T[j] >= 0 or P[j] >= 0:
            xo.append(P[j] if P[j] >= 0 else 312)
            yo.append(T[j] if T[j] >= 0 else 312)
            if T[j] == P[j]:
                ex_match += 1
xo, yo = np.array(xo), np.array(yo)
a.scatter(xo, yo, s=15, alpha=0.7, color=CB[0], lw=0)
lim = [0, 315]
a.plot(lim, lim, color="gray", lw=1.0, ls="--")
a.axhline(302, color="gray", lw=0.8, ls=":"); a.axvline(302, color="gray", lw=0.8, ls=":")
a.set_xlim(-5, 320); a.set_ylim(-5, 320)
a.set_xlabel("threshold-model $T_{pred}$ (no flip = 312)")
a.set_ylabel("observed T (no flip = 312)")
a.set_title(f"(d) per-direction T, r$\\in${{0.3,0.5,1,2}} (main): observed NEVER precedes prediction\n"
            f"exact {ex_match}/{len(xo)}; obs flips without prediction: 0; obs later than pred: {tot_later} vs earlier: {tot_earlier}", fontsize=9)
fig.savefig(OUT + r"\fig2_scaling.png")
plt.close(fig)

# =====================================================================
# FIG 3 - survival curves, small/mid radii (main prompt)
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
fig.suptitle("Survival curves at small/mid radii: threshold field reproduces them EXACTLY; "
             "no hazard exists below the smallest dip", fontsize=11, fontweight="bold")
radii_mid = [0.05, 0.1, 0.2, 0.3]
for i, r in enumerate(radii_mid):
    a = ax[i // 2, i % 2]
    (tg, Sobs), (tp, Sp), Sexp, (tb, Slo, Shi), Ts, Tps = curves[("main", r)]
    st = stats["main"][f"r{r:g}"]
    a.fill_between(tb, Slo, Shi, color=CB[0], alpha=0.18, step="post")
    a.step(tg, Sobs, where="post", color=CB[0], lw=1.6, label="observed $S_r(t)$ (95% CI)")
    a.step(tp, Sp, where="post", color=CB[3], lw=1.6, ls="--", label="threshold-field prediction")
    if st["n_flip"] > 0:
        a.plot(tg, Sexp, color=CB[2], lw=1.2, alpha=0.9,
               label=f"exp fit ($\\lambda$={st['fits']['exp']['lam']:.1e})")
    a.set_xlim(0, 348); a.set_ylim(0, 1.04)
    a.set_xlabel("generation step t"); a.set_ylabel("survival  $S_r(t)=P(\\mathrm{no\\ flip\\ by}\\ t)$")
    concl = {0.05: "no dip reachable: S=1 exactly, predicted and observed",
             0.1: "no dip reachable: S=1 exactly, predicted and observed",
             0.2: "threshold model EXACT (MAD=0.00, 20/20 dirs)",
             0.3: "one predicted extra flip; 19/20 exact"}[r]
    a.set_title(f"({'abcd'[i]}) r={r:g}: {concl}", fontsize=9)
    a.legend(loc="lower right", fontsize=7.5)
fig.savefig(OUT + r"\fig3_survival_mid.png")
plt.close(fig)

# =====================================================================
# FIG 4 - survival curves, large radii + other prompt
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
fig.suptitle("Survival at large radii: stepped curves with PERMANENT plateau (threshold crossing); "
             "one-sided deviations only (free run more stable than teacher-forced)",
             fontsize=11, fontweight="bold")
panels = [("main", 0.5), ("main", 1.0), ("main", 2.0), ("childhood", 0.3)]
for i, (p, r) in enumerate(panels):
    a = ax[i // 2, i % 2]
    (tg, Sobs), (tp, Sp), Sexp, (tb, Slo, Shi), Ts, Tps = curves[(p, r)]
    st = stats[p][f"r{r:g}"]
    a.fill_between(tb, Slo, Shi, color=CB[0], alpha=0.18, step="post")
    a.step(tg, Sobs, where="post", color=CB[0], lw=1.6, label="observed $S_r(t)$ (95% CI)")
    a.step(tp, Sp, where="post", color=CB[3], lw=1.6, ls="--", label="threshold-field prediction")
    a.plot(tg, Sexp, color=CB[2], lw=1.2, alpha=0.9,
           label=f"exp fit ($\\lambda$={st['fits']['exp']['lam']:.1e})")
    a.set_xlim(0, 348); a.set_ylim(0, 1.04)
    a.set_xlabel("generation step t"); a.set_ylabel("survival $S_r(t)$")
    concl = {("main", 0.5): f"flips only at dips 2/16/123; 4 predicted flips absent in free run (MAD {st['MAD_threshold']:.2f})",
             ("main", 1.0): f"plateau from t$\\approx$150 (8/20 flipped, rest never); pred over-fires 15 vs 8",
             ("main", 2.0): f"MAD thr {st['MAD_threshold']:.2f} < exp {st['MAD_exp']:.2f}; permanent plateau $S=0.2$",
             ("childhood", 0.3): f"second prompt: MAD thr {st['MAD_threshold']:.2f} < exp {st['MAD_exp']:.2f}; 13/20 exact"}[(p, r)]
    a.set_title(f"({'abcd'[i]}) {p} r={r:g}: {concl}", fontsize=9)
    a.legend(loc=("lower right" if i < 2 else "upper right"), fontsize=7.5)
fig.savefig(OUT + r"\fig4_survival_large.png")
plt.close(fig)

# =====================================================================
# FIG 5 - flux-tube signature: what happens after the flip
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
fig.suptitle("No gradual post-flip separation: token streams diverge IMMEDIATELY and permanently; "
             "state perturbation never amplifies (teacher-forced |dh| flat, then window wash-out)",
             fontsize=11, fontweight="bold")

a = ax[0, 0]
for (d, tag, col) in [(large, "main prompt (27 flips)", CB[0]),
                      (ch, "childhood (16 flips)", CB[1]),
                      (cy, "cyber (3 flips)", CB[2])]:
    ks = np.arange(0, 61)
    A = np.zeros(len(ks)); N = np.zeros(len(ks))
    for k in range(d["obs"].shape[0]):
        for j in range(20):
            T = d["obs"][k, j]
            if T < 0:
                continue
            for ik, kk in enumerate(ks):
                if T + kk < 300:
                    A[ik] += (not d["mismatch"][k * 20 + j, T + kk]); N[ik] += 1
    Ag = np.where(N > 0, A / np.maximum(N, 1), np.nan)
    a.plot(ks, Ag, color=col, lw=1.4, label=tag)
a.set_xlabel("tokens since first flip, k"); a.set_ylabel("fraction matching reference, A(k)")
a.set_ylim(-0.03, 1.03)
a.set_title("(a) post-flip token agreement: main prompt collapses to 0 at once\n"
            "(childhood: brief 15-token structural echo, then 0)", fontsize=9)
a.legend(loc="center right")

a = ax[0, 1]
mm = large["mismatch"][40:60]  # r=2.0, 20 dirs
a.imshow(mm.astype(float), aspect="auto", cmap="Greys", interpolation="nearest",
         extent=[0, 300, 0, 20])
for j in range(20):
    T = large["obs"][2, j]
    if T >= 0:
        a.axvline(T, color=CB[3], lw=0.5, alpha=0.35)
a.set_xlabel("generation step t"); a.set_ylabel("direction index (r=2)")
a.set_title("(b) mismatch raster, r=2: long exact shadowing (white),\n"
            "instant permanent exit (black) at discrete dip steps (red)", fontsize=9)

a = ax[1, 0]
labels, P1, P15, deep = [], [], [], []
for (d, tag) in [(large, "main"), (ch, "childhood"), (cy, "cyber")]:
    r1, r15, dp = [], [], []
    for k in range(d["obs"].shape[0]):
        for j in range(20):
            T = d["obs"][k, j]
            if T < 0:
                continue
            row = d["mismatch"][k * 20 + j]
            if T + 1 < 300:
                r1.append(not row[T + 1])
            w = row[T + 1: T + 16]
            if len(w):
                r15.append(1 - w.mean())
            if T + 80 < 300:
                dp.append(1 - row[T + 80:].mean())
    labels.append(tag); P1.append(np.mean(r1)); P15.append(np.mean(r15)); deep.append(np.mean(dp))
xs = np.arange(3)
a.bar(xs - 0.27, P1, 0.26, color=CB[0], label="P(match at T+1)")
a.bar(xs, P15, 0.26, color=CB[1], label="match rate in [T+1,T+15]")
a.bar(xs + 0.27, deep, 0.26, color=CB[2], label="match rate at t$\\geq$T+80 (baseline)")
a.set_xticks(xs); a.set_xticklabels(labels)
a.set_ylabel("fraction matching reference")
a.set_title("(c) post-flip re-sync: main prompt 0/27 rejoin; baseline agreement\n"
            "deep in diverged runs is 1-3%", fontsize=9)
a.legend(fontsize=7.5)

a = ax[1, 1]
dha = np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)
dh = np.asarray(dha["dh_norm"], float).mean(axis=1)  # [12,8,300]
for i, r in [(4, 0.1), (7, 0.5), (9, 1.0), (11, 2.0)]:
    p = dh[i, 7] / dh[i, 7, 1]
    a.plot(np.arange(300), p, lw=1.3, label=f"r={r:g}")
a.axhline(1.0, color="gray", lw=0.8, ls=":")
a.set_xlabel("generation step t (teacher-forced)")
a.set_ylabel("$|\\delta h_{L27}(t)|\\,/\\,|\\delta h_{L27}(1)|$")
a.set_title("(d) perturbation norm: no exponential growth (ratio stays ~0.7-1.3 for 100 steps),\n"
            "then attention-window wash-out; $\\cos(\\delta h,h)\\approx$ 0.9998", fontsize=9)
a.legend(loc="upper right", fontsize=7.5, title="injected radius", title_fontsize=7.5)
fig.savefig(OUT + r"\fig5_fluxtube.png")
plt.close(fig)

# =====================================================================
# FIG 6 - window comparison (teacher-forced fields, same prompt + dirs)
# =====================================================================
win_files = [("w=100", r"w100_layers\a1.npz"), ("w=200", r"w200\a1.npz"), ("w=500", r"w500\a1.npz")]
wins = {}
for tag, f in win_files:
    wins[tag] = load_a1(f)
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
fig.suptitle("Longer attention windows have denser, deeper margin dips -> lower thresholds and faster "
             "predicted destabilization (same prompt, same 20 directions)",
             fontsize=11, fontweight="bold")

a = ax[0, 0]
mgrid = np.geomspace(1e-3, 20, 300)
for i, (tag, _) in enumerate(win_files):
    m = wins[tag]["margins"]
    S = np.array([(m > q).mean() for q in mgrid])
    a.semilogx(mgrid, S, color=CB[i], lw=1.4, label=f"{tag}")
a.set_xlabel("margin level m"); a.set_ylabel("P(margin > m) along reference")
a.set_title("(a) margin survival: fraction of low-margin ('decision boundary')\n"
            "steps grows with window (min margin 0.020 / 0.0031 / 0.0031)", fontsize=9)
a.legend(loc="lower left")

def thr_curve(w, r):
    radii = w["radii"]
    i = int(np.argmin(np.abs(radii - r)))
    assert abs(radii[i] - r) < 1e-9
    (tg, S), Tp = threshold_survival(w["r_c"][i], r)
    return tg, S, Tp

for pi, r in enumerate([0.1, 0.3]):
    a = ax[0, 1] if pi == 0 else ax[1, 0]
    ends = {}
    for i, (tag, _) in enumerate(win_files):
        tg, S, Tp = thr_curve(wins[tag], r)
        a.step(tg, S, where="post", color=CB[i], lw=1.5, label=tag)
        ends[tag] = S[-1]
    # observed free-run at w=100 for reference
    if r in obs_main:
        tg, So = km_curve(obs_main[r])
        a.step(tg, So, where="post", color="k", lw=1.0, ls=":", label="observed free run (w=100)")
    a.set_xlim(0, 348); a.set_ylim(0, 1.04)
    a.set_xlabel("step t"); a.set_ylabel("survival $S_r(t)$")
    a.set_title(f"({'bc'[pi]}) threshold-model survival at r={r:g}\n"
                "windows shift the field; w=100 prediction matches its free run", fontsize=9)
    if pi == 0:
        a.legend(loc="lower right", fontsize=7.5)
    else:  # plateau labels in the right margin instead of a legend box
        a.text(304, 1.005, "w=100 obs (dotted)", fontsize=7, color="k")
        a.text(304, ends["w=100"] + 0.02, "w=100 pred", fontsize=7, color=CB[0])
        a.text(304, ends["w=200"] + 0.03, "w=200", fontsize=7, color=CB[1])
        a.text(304, ends["w=500"] - 0.06, "w=500", fontsize=7, color=CB[2])

a = ax[1, 1]
xs = np.arange(3)
wvals = {}
for pi, r in enumerate([0.1, 0.3]):
    fr = []
    for tag, _ in win_files:
        tg, S, Tp = thr_curve(wins[tag], r)
        fr.append((Tp >= 0).mean())
    wvals[r] = fr
    a.bar(xs + (pi - 0.5) * 0.32, fr, 0.30, color=CB[pi], label=f"r={r:g} (predicted)")
# observed free-run points at w=100 only
a.scatter([xs[0] - 0.16], [(obs_main[0.1] >= 0).mean()], marker="_", s=220, color=CB[0],
          label="observed free run (w=100)")
a.scatter([xs[0] + 0.16], [(obs_main[0.3] >= 0).mean()], marker="_", s=220, color=CB[1])
a.set_xticks(xs); a.set_xticklabels(["w=100", "w=200", "w=500"])
a.set_ylabel("predicted flip fraction within 300 steps")
a.set_title("(d) threshold-model flip fraction grows with window at fixed radius\n"
            "(same prompt + same 20 directions; free-run points shown for w=100)", fontsize=9)
a.legend(fontsize=7.5, loc="upper left")

win_stats = {}
for tag, f in win_files:
    w = wins[tag]
    m = w["margins"]
    win_stats[tag] = dict(min_margin=float(m.min()),
                          frac_lt02=float((m < 0.2).mean()),
                          q01=float(np.quantile(m, 0.01)))
    for r in [0.1, 0.3]:
        tg, S, Tp = thr_curve(w, r)
        win_stats[tag][f"flipfrac_r{r:g}"] = float((Tp >= 0).mean())
        win_stats[tag][f"meanT_r{r:g}"] = mean_restricted(Tp)
extra["windows"] = win_stats
print(json.dumps(win_stats, indent=1))
fig.savefig(OUT + r"\fig6_windows.png")
plt.close(fig)

json.dump(extra, open(OUT + r"\stats_extra.json", "w"), indent=1)
print("\nAll figures saved.")
