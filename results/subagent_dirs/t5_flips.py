"""TEST 5: do directions differ in FLIP behavior, and why?

A "flip" = the free-running generated text actually changes (obs in
fr_w100_large.npz, first differing token index, -1 = never, at r=0.5, 1, 2).
Predicted threshold per direction: r*(d) = min_t r_c[d,t] with
r_c = baseline margin / signed margin-response (inf where response >= 0).

Two ways to flip early:
  (A) GAIN: the direction's |dm| is large everywhere -> low r* at any token;
  (B) NEAR-TIE LUCK: the direction's response happens to be negative at one of
      the few near-tie tokens (margin 0.02-0.2), while its typical |dm| is modest.
Separation: log r*(d) = const + [log margin(t*) - log median margin] (tie part)
                                - [log s_d(t*) - log median s]  (response part).
Also: per-direction SIGN bias of dm (harmful vs harmless directions), and whether
the sign pattern at a given token is shared across directions or direction-specific.
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L27, load_fr, load_w100, save_json, savefig, spear

z = load_w100()
dh, dm, margins, r_c = z["dh_norm"], z["dm"], z["margins"], z["r_c"]
radii = z["radii"]
fr = load_fr()
obs, fr_radii = fr["obs"], fr["radii"]     # [3, 20], radii 0.5,1,2
D, T = dm.shape[1], dm.shape[-1]
EX = [10, 9, 13]
cols = ["#d62728", "#1f77b4", "#2ca02c"]
BIG = 1e20

N = {}

def rstar(ridx):
    rc = np.where(r_c[ridx] < BIG, r_c[ridx], np.inf)   # [D, T]
    return rc.min(axis=1), rc.argmin(axis=1)

# observed flip category per direction: 0.5, 1, 2, or >2 (earliest radius wins)
cat = np.full(D, 3)
for i in reversed(range(len(fr_radii))):      # r=2 first, r=0.5 last
    flipped = obs[i] >= 0
    cat[flipped] = i

# ---------------- pred vs obs ----------------
rstars, tstars = {}, {}
for i, rr in enumerate(fr_radii):
    ridx = int(np.where(np.isclose(radii, rr))[0][0])
    rs, ts = rstar(ridx)
    rstars[rr], tstars[rr] = rs, ts
    pred_flip = rs < rr
    obs_flip = obs[i] >= 0
    N[f"r={rr}"] = {
        "n_pred_flips": int(pred_flip.sum()), "n_obs_flips": int(obs_flip.sum()),
        "misses (obs flip, no pred)": int((obs_flip & ~pred_flip).sum()),
        "false alarms (pred flip, no obs)": int((pred_flip & ~obs_flip).sum()),
    }
    N[f"r={rr}"]["spearman_rstar_vs_obsstep"] = spear(
        rs, np.where(obs[i] >= 0, obs[i], 300))

# flip-risk vs gain (is early flipping just gain?)
gain_dm = np.median(np.abs(dm[9]), axis=1)      # median |dm| at r=1
gain_dh = np.median(dh[4, :, L27], axis=1)      # dh gain at r=0.1
N["spearman_logrstar_r1_vs_dmgain"] = spear(np.log(rstars[1.0]), np.log(gain_dm))
N["spearman_logrstar_r1_vs_dhgain"] = spear(np.log(rstars[1.0]), np.log(gain_dh))
N["spearman_logrstar_r0.5_vs_dhgain"] = spear(np.log(rstars[0.5]), np.log(gain_dh))
N["earliest_flippers_r0.5"] = np.argsort(cat)[:3].tolist()

# ---------------- two causes decomposition at r=1 and r=2 ----------------
# NOTE (verified): stored r_c = margin * r / |dm| -- a MAGNITUDE threshold that
# ignores the sign of dm. So r*(d) = min_t r_c is the radius at which direction
# d's margin *changes* by as much as the margin itself; whether that is a real
# flip depends on the SIGN of dm at the bottleneck token (panel f).
decomp = {}
for rr in (1.0, 2.0):
    ridx = int(np.where(np.isclose(radii, rr))[0][0])
    ts = tstars[rr]
    s = np.abs(dm[ridx]) / rr                     # magnitude response per (d,t)
    tie = np.log(margins[ts] / np.median(margins))            # <0 if near-tie helps
    resp = np.log(s[np.arange(D), ts] / np.median(s))          # >0 if response spikes there
    lr = np.log(rstars[rr][rstars[rr] < BIG])
    v_tie, v_resp = float(np.var(tie)), float(np.var(resp))
    decomp[rr] = {
        "var_logrstar": float(np.var(lr)),
        "var_tie_part": v_tie, "var_response_part": v_resp,
        "cov": float(np.cov(tie, resp)[0, 1]),
        "var_lr_explained_by_parts": float(np.var(tie - resp)),
    }
    decomp[rr]["tie"] = tie.tolist()
    decomp[rr]["resp"] = resp.tolist()
    decomp[rr]["tstar"] = ts.tolist()
N["decomp"] = {f"r={rr}": {k: v for k, v in d.items() if isinstance(v, float)}
               for rr, d in decomp.items()}

# bottleneck sharing
ntt = margins.argsort()[:6]
for rr in (1.0, 2.0):
    ts = np.array(decomp[rr]["tstar"])
    N[f"tstar_stats_r={rr}"] = {
        "n_distinct": int(len(np.unique(ts))),
        "frac_in_top6_neartie": float(np.mean([t in ntt for t in ts])),
        "counts": {int(t): int(c) for t, c in zip(*np.unique(ts, return_counts=True))},
    }

# ---------------- sign structure ----------------
for ridx, rr in [(9, 1.0), (11, 2.0)]:
    fn = (dm[ridx] < 0).mean(axis=1)
    se = np.sqrt(0.25 / T)
    N[f"sign_bias_r={rr}"] = {
        "fracneg": fn.tolist(),
        "n_beyond_2se": int((np.abs(fn - 0.5) > 2 * se).sum()),
        "min": float(fn.min()), "max": float(fn.max()),
    }
fn1 = (dm[9] < 0)
agree = np.array([[ (fn1[a] == fn1[b]).mean() for b in range(D)] for a in range(D)])
ag_off = agree[np.triu_indices(D, 1)]
# sign agreement restricted to near-tie tokens
fn_nt = dm[9][:, ntt] < 0
agree_nt = np.array([[(fn_nt[a] == fn_nt[b]).mean() for b in range(D)] for a in range(D)])
ag_nt_off = agree_nt[np.triu_indices(D, 1)]
N["sign_agreement"] = {
    "all_tokens_median": float(np.median(ag_off)),
    "all_tokens_iqr": [float(x) for x in np.percentile(ag_off, [25, 75])],
    "neartie_median": float(np.median(ag_nt_off)),
    "neartie_iqr": [float(x) for x in np.percentile(ag_nt_off, [25, 75])],
}
# signed response at the two most shared bottleneck tokens, r=1
tok_report = {}
for t0 in [123, 2, 143]:
    resp_t0 = -dm[9, :, t0] / 1.0
    tok_report[t0] = {
        "margin": float(margins[t0]),
        "resp_min": float(resp_t0.min()), "resp_max": float(resp_t0.max()),
        "n_dirs_would_flip(r=1)": int((resp_t0 > margins[t0]).sum()),
    }
N["bottleneck_token_responses_r1"] = tok_report

N["r=1"] = N.pop("r=1.0")
N["r=2"] = N.pop("r=2.0")
print("=== TEST 5 ===")
for k in ["r=0.5", "r=1", "r=2"]:
    print(k, N[k])
print("spearman(log r*, dm gain):", round(N["spearman_logrstar_r1_vs_dmgain"], 3),
      "| vs dh gain:", round(N["spearman_logrstar_r1_vs_dhgain"], 3))
print("decomp r=2:", {k: round(v, 3) for k, v in decomp[2.0].items() if isinstance(v, float)})
print("t* stats r=2:", N["tstar_stats_r=2.0"])
print("sign bias r=2: n beyond 2se =", N["sign_bias_r=2.0"]["n_beyond_2se"],
      "range", round(N["sign_bias_r=2.0"]["min"], 2), round(N["sign_bias_r=2.0"]["max"], 2))
print("sign agreement: all-tokens median", round(N["sign_agreement"]["all_tokens_median"], 3),
      " near-tie median", round(N["sign_agreement"]["neartie_median"], 3))
print("bottleneck tokens:", tok_report)

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 3, figsize=(15, 9))

# (a) direction thresholds + observed outcomes
ax = axs[0, 0]
rs2 = np.where(rstars[2.0] < BIG, rstars[2.0], 3.0 * np.median(rstars[2.0][rstars[2.0] < BIG]))
o = np.argsort(rs2)
cmapcat = {0: "#d62728", 1: "#ff7f0e", 2: "#bcbd22", 3: "gray"}
for k, d in enumerate(o):
    ax.bar(k, rs2[d] if rstars[2.0][d] < BIG else 2 * rs2[o].max(), color=cmapcat[int(cat[d])])
ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.axhline(1, color="k", ls=":", lw=0.8)
ax.axhline(2, color="k", ls=":", lw=0.8)
ax.text(20.3, 0.42, "r=0.5", fontsize=7); ax.text(20.3, 0.92, "r=1", fontsize=7)
ax.text(20.3, 1.92, "r=2", fontsize=7)
ax.set_yscale("log")
ax.set_xticks(range(D)); ax.set_xticklabels([f"{d}" for d in o], fontsize=6)
ax.set_xlabel("direction (sorted by predicted threshold r*)")
ax.set_ylabel("r*(d) = min_t [margin_t / |dm response|]  (log scale)")
ax.set_title(f"(a) Flip threshold per direction (bars colored by OBSERVED flip)\n"
             f"red=flipped at 0.5, orange=at 1, yellow=at 2, gray=never. "
             f"Dirs 10,11,12 (hot trio) flip first\n"
             f"spearman(r*, observed step) r=1: {N['r=1']['spearman_rstar_vs_obsstep']:.2f}; "
             f"false alarms r=1: {N['r=1']['false alarms (pred flip, no obs)']}, "
             f"misses: {N['r=1']['misses (obs flip, no pred)']}")

# (b) two causes at r=2
ax = axs[0, 1]
tie, resp = np.array(decomp[2.0]["tie"]), np.array(decomp[2.0]["resp"])
sz = 14 + 70 * (np.log(gain_dh) - np.log(gain_dh.min())) / (np.log(gain_dh.max()) - np.log(gain_dh.min()))
ax.scatter(tie, resp, s=sz, color="gray")
for k, d in enumerate(EX):
    ax.scatter(tie[d], resp[d], s=sz[d] * 1.6, color=cols[k], zorder=3, label=f"dir {d}")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("tie part: log(margin at bottleneck token / median margin)\n<0 = flips because token is a near-tie")
ax.set_ylabel("response part: log(|dm| at bottleneck / typical |dm|)\n>0 = flips because response spikes there")
ax.set_title(f"(b) Why each direction's r* is what it is (r=2; r_c is a magnitude threshold)\n"
             f"var of log r*: tie {N['decomp']['r=2.0']['var_tie_part']:.3f} vs "
             f"response {N['decomp']['r=2.0']['var_response_part']:.3f}; "
             f"sum rule check: var(tie-resp) = {N['decomp']['r=2.0']['var_lr_explained_by_parts']:.3f} "
             f"vs var(log r*) = {N['decomp']['r=2.0']['var_logrstar']:.3f}")
ax.legend(fontsize=7)

# (c) bottleneck tokens on the margin profile
ax = axs[0, 2]
ax.semilogy(np.arange(T), margins, color="gray", lw=0.8, label="baseline logit margin per token")
ts2 = np.array(decomp[2.0]["tstar"])
ax.scatter(ts2, margins[ts2] * np.ones(D), marker="|", s=200, color="k",
           label="bottleneck token t* of each direction (r=2)")
ax.set_xlabel("generated-token position t")
ax.set_ylabel("baseline top1-top2 logit margin (logits, log scale)")
ax.set_title(f"(c) Where flips happen: {N['tstar_stats_r=2.0']['n_distinct']} distinct t* among 20 dirs; "
             f"{N['tstar_stats_r=2.0']['frac_in_top6_neartie']*100:.0f}% are in the 6 nearest-tie tokens\n"
             f"nearest ties: t=123 (0.020), t=143 (0.020), t=116 (0.069), t=2 (0.070)")
ax.legend(fontsize=7)

# (d) sign bias
ax = axs[1, 0]
fnv = N["sign_bias_r=2.0"]["fracneg"]
o = np.argsort(fnv)
ax.bar(np.arange(D) - 0.2, np.array(fnv)[o], width=0.4, color="#1f77b4", label="r=2")
fnv1 = N["sign_bias_r=1.0"]["fracneg"]
ax.bar(np.arange(D) + 0.2, np.array(fnv1)[o], width=0.4, color="#aec7e8", label="r=1")
se = np.sqrt(0.25 / T)
ax.axhspan(0.5 - 2 * se, 0.5 + 2 * se, color="gray", alpha=0.3, label="2-sigma band if signs were coin flips")
ax.axhline(0.5, color="k", lw=0.6)
ax.set_xticks(range(D)); ax.set_xticklabels([f"{d}" for d in o], fontsize=6)
ax.set_xlabel("direction (sorted by fraction of margin-shrinking tokens, r=2)")
ax.set_ylabel("fraction of 300 tokens where dm < 0 (margin shrinks)")
ax.set_title(f"(d) Sign bias: {N['sign_bias_r=2.0']['n_beyond_2se']}/20 directions outside the "
             f"coin-flip band at r=2\nrange {N['sign_bias_r=2.0']['min']:.2f}..{N['sign_bias_r=2.0']['max']:.2f} "
             f"(0.5 = unbiased)")
ax.legend(fontsize=7)

# (e) sign agreement across direction pairs
ax = axs[1, 1]
ax.hist(ag_off, bins=20, alpha=0.7, color="C0", label="all 300 tokens")
ax.hist(ag_nt_off, bins=20, alpha=0.7, color="C1", label="6 nearest-tie tokens only")
ax.axvline(0.5, color="k", ls="--", lw=1, label="0.5 = independent signs")
ax.set_xlabel("fraction of tokens where two directions' dm agree in sign")
ax.set_ylabel("number of direction pairs (of 190)")
ax.set_title(f"(e) Is the sign pattern shared or direction-specific? (r=1)\n"
             f"median agreement: all tokens {N['sign_agreement']['all_tokens_median']:.2f}, "
             f"near-ties {N['sign_agreement']['neartie_median']:.2f}")
ax.legend(fontsize=7)

# (f) the shared bottleneck token 123: direction-specific vulnerability
ax = axs[1, 2]
resp123 = -dm[9, :, 123]
o = np.argsort(resp123)
ax.bar(range(D), resp123[o], color="gray")
ax.axhline(margins[123], color="r", lw=1.5, label=f"margin at t=123 = {margins[123]:.3f} (near-tie)")
for k, d in enumerate(EX):
    ax.bar(np.where(o == d)[0], resp123[d], color=cols[k])
fliphere = resp123 > margins[123]
ax.set_xticks(range(D)); ax.set_xticklabels([f"{d}" for d in o], fontsize=6)
ax.set_xlabel("direction (sorted by signed response at token 123)")
ax.set_ylabel("-dm / r at token 123 (r=1): margin-shrinking response")
ax.set_title(f"(f) Same near-tie token, different directions (r=1)\n"
             f"{fliphere.sum()}/20 directions would flip AT THIS TOKEN at r=1; "
             f"range of responses: {resp123.min():.3f} .. {resp123.max():.3f} "
             f"({resp123.max()/max(resp123.min(),1e-6):.0f}x spread, sign flips included)")
ax.legend(fontsize=7)

fig.suptitle("TEST 5: flip behavior - directions differ in WHEN they flip and WHY "
             "(free-run obs vs predicted thresholds)", y=1.005)
savefig(fig, "fig5_flips.png")
save_json("numbers_t5.json", N)
