"""Aggregate all experiment JSONs into the exact numbers cited in REPORT.md."""
import json
import numpy as np
from scipy import stats
from common_d2c import OUT, PROMPTS, REP

with open(OUT / "d2_results.json") as f:
    d2 = json.load(f)
with open(OUT / "pred_results.json") as f:
    pred = json.load(f)
with open(OUT / "delay_results.json") as f:
    dl = json.load(f)
with open(OUT / "props_results.json") as f:
    pr = json.load(f)

print("=" * 70)
print("Q1: D2 scaling (N=1000)")
print("=" * 70)
for n in PROMPTS:
    r = d2[n]["cfgs"]["N1000_w0"]
    run = r["run"]
    nbf = r["nbf"]
    print(f"{n[:34]:36s} er_tot={r['er_tot']:+.3f} er_best={r['er_best']:+.3f} "
          f"run={run['width_dec'] if run else None} "
          f"legal={r['erlegal_run']} "
          f"nbf={nbf['slope'] if nbf else None}@{nbf['width_dec'] if nbf else None}dec "
          f"C_lo={nbf['C_lo'] if nbf else None}")
n_valid = sum(1 for n in PROMPTS if d2[n]["cfgs"]["N1000_w0"]["erlegal_run"]
              and d2[n]["cfgs"]["N1000_w0"]["run"]
              and d2[n]["cfgs"]["N1000_w0"]["run"]["width_dec"] >= 0.5)
ers = [d2[n]["cfgs"]["N1000_w0"]["er_tot"] for n in PROMPTS]
print(f"valid scaling (ER-legal stable run >=0.5 dec): {n_valid}/16; "
      f"er_tot median {np.median(ers):.3f} max {np.max(ers):.3f}")
# naive fits legal?
nn = 0
for n in PROMPTS:
    nb = d2[n]["cfgs"]["N1000_w0"]["nbf"]
    if nb and nb["C_lo"] >= 10 / 1000 and nb["slope"] <= 2 * np.log10(1000):
        nn += 1
print(f"naive r2>=0.99 fits that are ER-legal: {nn}/16 (the rest are depletion-zone artifacts)")
w0 = [max(d2[n]["shuffle_w0_max_int_dev"]) for n in PROMPTS]
w50 = [d2[n]["shuffle_w50_max_dlogC"] for n in PROMPTS]
print(f"time-shuffle w=0: max integer pair-count deviation = {max(w0)} (exact identity)")
print(f"time-shuffle w=50: max |dlogC| for C>1e-4: median {np.median(w50):.3f} max {np.max(w50):.3f}")
fid = np.array([np.mean(d2[n]["miiaft_crosscov_relerr"]) for n in PROMPTS])
print(f"cloud-MIAAFT top-50 crosscov rel err: median {np.median(fid):.3f}")

print()
print("=" * 70)
print("RQA DET at RR=10% (data vs surrogates)")
print("=" * 70)
rows = []
for n in PROMPTS:
    g = lambda v, k=0: next(x["det"] for x in v if x["rr"] == 0.10)
    d = g(d2[n]["rqa_data"])
    ia = np.mean([g(v) for v in d2[n]["rqa_iaaft"]])
    ph = np.mean([g(v) for v in d2[n]["rqa_phase"]])
    sh = np.mean([g(v) for v in d2[n]["rqa_shuffle"]])
    rows.append((n, d, ia, ph, sh))
for n, d, ia, ph, sh in sorted(rows, key=lambda r: -r[1]):
    print(f"{n[:34]:36s} data={d:.3f} MIAAFT={ia:.3f} phase={ph:.3f} shuffle={sh:.3f}")
d = np.array([r[1] for r in rows]); ia = np.array([r[2] for r in rows])
try:
    p = stats.wilcoxon(d, ia).pvalue
except Exception:
    p = float("nan")
print(f"DET data - MIAAFT: median {np.median(d - ia):+.3f}, Wilcoxon p={p:.2g}")

print()
print("=" * 70)
print("Q2: predictability")
print("=" * 70)
HS = [1, 2, 4, 8, 16, 32]
for hi, h in enumerate(HS):
    dd = np.array([pred[n]["nse_knn"][hi] for n in PROMPTS])
    ia = np.array([np.mean(pred[n]["knn_iaaft"][hi]) for n in PROMPTS])
    ph = np.array([np.mean(pred[n]["knn_phase"][hi]) for n in PROMPTS])
    sh = np.array([np.mean(pred[n]["knn_shuffle"][hi]) for n in PROMPTS])
    rg = np.array([pred[n]["nse_ridge"][hi] for n in PROMPTS])
    try:
        p = stats.wilcoxon(dd, ia).pvalue
    except Exception:
        p = float("nan")
    print(f"h={h:2d} knn {np.median(dd):+.3f} ridge {np.median(rg):+.3f} "
          f"iaaft {np.median(ia):+.3f} phase {np.median(ph):+.3f} shuf {np.median(sh):+.3f} "
          f"| gapIA med {np.median(dd - ia):+.3f} max {np.max(dd - ia):+.3f} p={p:.3g}")
print()
for n in sorted(PROMPTS, key=lambda n: -pred[n]["gap_best_surr_h8"])[:5]:
    r = pred[n]
    print(f"{n[:34]:36s} h8: knn {r['nse_knn'][3]:+.3f} ridge {r['nse_ridge'][3]:+.3f} "
          f"gap_best {r['gap_best_surr_h8']:+.3f} gap_ridge {r['gap_ridge_h8']:+.3f} "
          f"z {r['z_iaaft_h8']} h_half {r['h_half_knn']}")
zs = [pred[n]["z_iaaft_h8"] for n in PROMPTS if pred[n]["z_iaaft_h8"] is not None]
print(f"z(MIAAFT) at h=8: max {max(zs)}, #>3: {sum(1 for z in zs if z > 3)}")
big = [n for n in PROMPTS if pred[n]["gap_best_surr_h8"] > 0.15]
print(f"trajectories with gap vs best surrogate > 0.15 at h=8: {big}")

print()
print("=" * 70)
print("Kaplan-Glass small-r ratio, h=8: data vs MIAAFT")
print("=" * 70)
for n in PROMPTS:
    kg = pred[n]["kg"]["8"]
    d = kg["data"]["small"]
    ia = np.median([q["small"] for q in kg["iaaft"]])
    print(f"{n[:34]:36s} data {d:.3f}  MIAAFT {ia:.3f}  diff {d - ia:+.3f}")

print()
print("=" * 70)
print("Delay embedding")
print("=" * 70)
below = 0; ins = 0; tot = 0
saturators = []
for n in PROMPTS:
    r = dl[n]
    ia = np.array(r["d2_iaaft"])
    b = sum(1 for mi in range(2, 9) if r["d2_data"][mi] < ia[mi].min())
    below += b; tot += 7
    d68 = r["d2_data"][4] - r["d2_data"][2]  # m=6 minus m=4 growth
    if r["d2_data"][8] < r["d2_data"][2]:
        saturators.append((n, r["d2_data"]))
print(f"(traj,m>=4) with data D2 below IAAFT band min: {below}/{tot}")
print(f"trajectories whose D2(m=10) < D2(m=4): {saturators}")

print()
print("=" * 70)
print("Properties and correlations")
print("=" * 70)
for n in PROMPTS:
    p = pr["props"][n]
    print(f"{n[:34]:36s} uniq {p['unique_frac']:.2f} H {p['entropy_bits']:.2f} "
          f"loop {p['loop_dist']:.3f}@{p['loop_P']} ngram {p['longest_rep_ngram']} "
          f"normCV {p['norm_cv']:.3f} PR {p['pc_participation']:.1f} n90 {p['n90']}")
for k, v in pr["spearman_vs"].items():
    print(f"{k:22s} rho={v['gap_surr'][0]:+.2f} p={v['gap_surr'][1]:.3f}   "
          f"(knn_h1 rho={v['knn_h1'][0]:+.2f} p={v['knn_h1'][1]:.3f})")
