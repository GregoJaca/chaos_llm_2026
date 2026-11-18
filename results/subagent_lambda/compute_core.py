"""Core number extraction for the lambda(t) analysis. Prints a compact report."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa

np.seterr(divide="ignore", invalid="ignore")
import warnings
warnings.filterwarnings("ignore")
z = load("w100_layers/a1.npz")
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
n0 = z["norm_h0"]
margins = z["margins"]

print("=" * 100)
print("A. w100_layers (prompt 65 tok, W=100, 300 gen tokens, 20 dirs, r=0.1 idx 4)")
d = dh[4]  # [20, 8, 300]
t = tvec(d)
lam1_mask = (t >= 2) & (t <= 6)
tail_mask = t > TAIL_T0
early_mask = (t >= 6) & (t <= 30)

for j, L in enumerate(layers):
    A = d[:, j, :]
    la5 = lam_k_2d(A, 5)
    la10 = lam_k_2d(A, 10)
    lam1 = np.diff(np.log(A), axis=1)  # [20, T-1], index i -> t=i+2
    dead = np.where(np.all(A <= 0, axis=0))[0]
    dead_t = int(dead[0]) + 1 if len(dead) else -1
    # early rate: mean of lambda_1 over t=2..6 (index 0..4)
    e1 = np.nanmean(lam1[:, :5], axis=1)
    e_m, e_lo, e_hi = boot_dirs_stat(e1)
    # OLS slope of log a over t in [2,30]
    sl = np.array([ols_slope(t[1:30], np.log(A[u, 1:30]))[0] for u in range(20)])
    sl_m, sl_lo, sl_hi = boot_dirs_stat(sl)
    # peak of smoothed median lambda5
    med = smooth(np.nanmedian(la5, axis=0), 9)
    if np.all(~np.isfinite(med)):
        print(f" L{L:2d}: response identically zero (embedding layer) - no rates")
        continue
    pk_t = int(np.nanargmax(med)) + 1
    pk_v = float(np.nanmax(med))
    # tail rate (mean lambda_10 over t>150) per direction + CIs
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    tr_m, tr_lo, tr_hi = boot_dirs_stat(tr)
    two_m, two_lo, two_hi = two_way_boot(la10, tail_mask, block=10)
    # cumulative
    cum = cum_rate_2d(A)
    L299 = cum[:, 299]
    c_m, c_lo, c_hi = boot_dirs_stat(L299)
    print(f" L{L:2d}: dead_t={dead_t:4d} | lam1(2-6)={e_m:+.4f}[{e_lo:+.4f},{e_hi:+.4f}]"
          f" | OLS loga(2-30)={sl_m:+.4f}[{sl_lo:+.4f},{sl_hi:+.4f}]"
          f" | peak lam5={pk_v:+.3f}@t={pk_t} | tail lam10={tr_m:+.4f}[{tr_lo:+.4f},{tr_hi:+.4f}]"
          f" | 2way=[{two_lo:+.4f},{two_hi:+.4f}] | Lam(299)={c_m:+.4f}[{c_lo:+.4f},{c_hi:+.4f}]")

# early growth ratio a(1)/a(0) at L27
A27 = d[:, 7, :]
r10 = np.log(A27[:, 1]) - np.log(A27[:, 0])
print(" L27 ln[a(2)/a(1)] per-dir mean: %+.4f  median %+.4f" % (r10.mean(), np.median(r10)))

print()
print("=" * 100)
print("B. radius sweep, w100_layers, L27 (idx7) and L14 (idx4): tail rate & early slope")
for i, r in enumerate(radii):
    A = dh[i][:, 7, :]
    la10 = lam_k_2d(A, 10)
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    tr_m, tr_lo, tr_hi = boot_dirs_stat(tr)
    sl = np.array([ols_slope(t[1:30], np.log(A[u, 1:30]))[0] for u in range(20)])
    sl_m, sl_lo, sl_hi = boot_dirs_stat(sl)
    nflip = int(np.sum(z["pred_T"][i] >= 0))
    med_a = np.nanmedian(A, axis=0)
    print(f" r={r:5.3f}: flips={nflip:2d}/20 | tail={tr_m:+.4f}[{tr_lo:+.4f},{tr_hi:+.4f}]"
          f" | earlyOLS={sl_m:+.4f}[{sl_lo:+.4f},{sl_hi:+.4f}]"
          f" | med a(30)/r={med_a[29]/r:8.3f} | med a(299)/r={med_a[298]/r:8.3f}")

print()
print("=" * 100)
print("C. window sweep (prompt 63, r=0.1 idx1, 10 dirs), L27 idx3 / L14 idx1")
rows = {}
for W in [50, 100, 200, 400, 800, 4096]:
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]  # r=0.1, L27
    A14 = zz["dh_norm"][1][:, 1, :]
    la5 = lam_k_2d(A, 5)
    la10 = lam_k_2d(A, 10)
    e5 = np.nanmean(la5[:, (t >= 6) & (t <= 30)], axis=1)
    e_m, e_lo, e_hi = boot_dirs_stat(e5)
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    tr_m, tr_lo, tr_hi = boot_dirs_stat(tr)
    two_m, two_lo, two_hi = two_way_boot(la10, tail_mask, block=10)
    cum = cum_rate_2d(A)
    L299 = cum[:, 299]
    c_m, c_lo, c_hi = boot_dirs_stat(L299)
    medcum = np.nanmedian(cum, axis=0)
    pos = np.where(medcum > 0)[0]
    cross = (np.argmax(medcum < 0) + 1) if np.nanmax(medcum) > 0 else -1
    a0 = np.nanmedian(A[:, 0])
    sl = np.array([ols_slope(t[1:30], np.log(A[u, 1:30]))[0] for u in range(10)])
    sl_m, sl_lo, sl_hi = boot_dirs_stat(sl)
    rows[W] = (e_m, tr_m)
    nflip = int(np.sum(zz["pred_T"][1] >= 0))
    print(f" W={W:4d}: flips={nflip:2d}/10 | a(1)med={a0:7.3f} | earlyOLS={sl_m:+.4f}[{sl_lo:+.4f},{sl_hi:+.4f}]"
          f" | early lam5(6-30)={e_m:+.4f}[{e_lo:+.4f},{e_hi:+.4f}]"
          f" | tail={tr_m:+.4f}[{tr_lo:+.4f},{tr_hi:+.4f}] | 2way=[{two_lo:+.4f},{two_hi:+.4f}]"
          f" | Lam(299)={c_m:+.4f}[{c_lo:+.4f},{c_hi:+.4f}] | pos_t={len(pos)} cross0={cross}")
    A14b = la10
    tr14 = np.nanmean(lam_k_2d(A14, 10)[:, tail_mask], axis=1)
    m14, lo14, hi14 = boot_dirs_stat(tr14)
    print(f"        L14 tail={m14:+.4f}[{lo14:+.4f},{hi14:+.4f}]")

print()
print("=" * 100)
print("D. window sweep cyber prompt, L27, r=0.1")
for W in [50, 100, 200, 400, 800, 4096]:
    zz = load(f"window_sweep_cyber/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    la10 = lam_k_2d(A, 10)
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    tr_m, tr_lo, tr_hi = boot_dirs_stat(tr)
    e5 = np.nanmean(lam_k_2d(A, 5)[:, (t >= 6) & (t <= 30)], axis=1)
    e_m, e_lo, e_hi = boot_dirs_stat(e5)
    cum = cum_rate_2d(A)
    c_m, c_lo, c_hi = boot_dirs_stat(cum[:, 299])
    print(f" W={W:4d}: early={e_m:+.4f}[{e_lo:+.4f},{e_hi:+.4f}] tail={tr_m:+.4f}[{tr_lo:+.4f},{tr_hi:+.4f}]"
          f" Lam(299)={c_m:+.4f}[{c_lo:+.4f},{c_hi:+.4f}]")

print()
print("=" * 100)
print("E. rate_prompts p1..p6 (W=100, r=0.1, 8 dirs, layers 7,14,21,27) + rate_w50")
for tag, folder in [("W100", "rate_prompts"), ("W50", "rate_w50")]:
    for p in sorted(os.listdir(os.path.join(DATA, folder))):
        fp = os.path.join(DATA, folder, p, "a1.npz")
        if not os.path.exists(fp):
            print(f" {tag} {p}: MISSING")
            continue
        zz = dict(np.load(fp))
        A = zz["dh_norm"][0][:, 3, :]  # r=0.1, L27
        la10 = lam_k_2d(A, 10)
        tr = np.nanmean(la10[:, tail_mask], axis=1)
        tr_m, tr_lo, tr_hi = boot_dirs_stat(tr)
        e5 = np.nanmean(lam_k_2d(A, 5)[:, (t >= 6) & (t <= 30)], axis=1)
        e_m, e_lo, e_hi = boot_dirs_stat(e5)
        cum = cum_rate_2d(A)
        c_m, c_lo, c_hi = boot_dirs_stat(cum[:, 299])
        nflip = int(np.sum(zz["pred_T"][0] >= 0))
        print(f" {tag} {p}: flips={nflip}/8 | early={e_m:+.4f}[{e_lo:+.4f},{e_hi:+.4f}]"
              f" tail={tr_m:+.4f}[{tr_lo:+.4f},{tr_hi:+.4f}] Lam(299)={c_m:+.4f}[{c_lo:+.4f},{c_hi:+.4f}]")

print()
print("=" * 100)
print("F. Jensen gap: mean(lambda_u) vs lambda(mean_a). w100_layers r=0.1.")
for j, L in [(7, 27), (4, 14)]:
    A = d[:, j, :]
    la5 = lam_k_2d(A, 5)
    mean_lam = np.nanmean(la5, axis=0)
    lam_of_mean = lam_k(np.nanmean(A, axis=0), 5)
    gap = lam_of_mean - mean_lam
    m = np.nanmean(gap[(t > 20) & np.isfinite(gap)])
    print(f" L{L}: mean gap over t>20 = {m:+.5f};  max|gap|={np.nanmax(np.abs(gap)):.4f}")

print()
print("=" * 100)
print("G. tail fits of Lambda(t), t>150: const vs power law (prompt63 window_sweep, L27)")
for W in [50, 100, 200, 4096]:
    zz = load(f"window_sweep/w{W}/a1.npz")
    A = zz["dh_norm"][1][:, 3, :]
    cum = cum_rate_2d(A)
    med = np.nanmedian(cum, axis=0)
    tt = t[tail_mask]
    yy = med[tail_mask]
    fc = fit_const(tt, yy)
    c, clo, chi = fit_const_ci(yy)
    fpw = fit_pow_decay(tt, yy)
    # log-a slope for comparison
    medlog = np.nanmedian(np.log(A), axis=0)
    s, s_lo, s_hi, r2 = ols_slope(tt, medlog[tail_mask])
    rel = np.log(n0[3][t > TAIL_T0])
    sr = ols_slope(tt, rel)[0]
    print(f" W={W:4d}: const c={fc['params']['c']:+.5f} CI=[{clo:+.5f},{chi:+.5f}]"
          f" | pow A={fpw['params']['A']:+.4f} p={fpw['params']['p']:.3f} c0={fpw['params']['c']:+.5f}"
          f" | AIC const={fc['aic']:.1f} pow={fpw['aic']:.1f} | BIC const={fc['bic']:.1f} pow={fpw['bic']:.1f}"
          f" | loga slope={s:+.5f} CI=[{s_lo:+.5f},{s_hi:+.5f}] | rel-corr={sr:+.5f}")

print()
print("=" * 100)
print("H. tail fits per layer, w100_layers r=0.1")
for j, L in [(1, 1), (2, 2), (3, 3), (4, 7), (5, 14), (6, 21), (7, 27)]:
    A = d[:, j, :]
    cum = cum_rate_2d(A)
    med = np.nanmedian(cum, axis=0)
    tt = t[tail_mask]
    yy = med[tail_mask]
    if not np.any(np.isfinite(yy)):
        print(f" L{L}: dead before tail")
        continue
    fc = fit_const(tt, yy)
    c, clo, chi = fit_const_ci(yy)
    fpw = fit_pow_decay(tt, yy)
    medlog = np.nanmedian(np.log(A), axis=0)
    s, s_lo, s_hi, r2 = ols_slope(tt, medlog[tail_mask])
    print(f" L{L:2d}: const={fc['params']['c']:+.5f} CI=[{clo:+.5f},{chi:+.5f}]"
          f" | pow c0={fpw['params']['c']:+.5f} A={fpw['params']['A']:+.4f} p={fpw['params']['p']:.3f}"
          f" | AIC c/p={fc['aic']:.1f}/{fpw['aic']:.1f} BIC={fc['bic']:.1f}/{fpw['bic']:.1f}"
          f" | loga slope={s:+.5f}[{s_lo:+.5f},{s_hi:+.5f}]")

print()
print("=" * 100)
print("I. token-level: correlation of lambda_5 with margin; margin stats")
A = d[:, 7, :]
la5 = lam_k_2d(A, 5)
mask = np.isfinite(la5[:, 5:295])
lm = np.log(np.maximum(margins[5:295], 1e-6))
xs = np.tile(lm, (20, 1))[mask]
ys = la5[:, 5:295][mask]
r = np.corrcoef(xs, ys)[0, 1]
print(f" corr(lam5, log margin) pooled 20 dirs, L27: r={r:+.3f}")
lo_idx = np.argsort(margins)[:15]
print(" 15 lowest-margin tokens:", (lo_idx + 1).tolist())
print(" their margins:", np.round(margins[lo_idx], 3).tolist())
med = np.nanmedian(la5, axis=0)
print(" lam5 median at those t:", np.round(med[lo_idx], 3).tolist())
print(" lam5 median overall std:", np.nanstd(med[5:295]))

print()
print("=" * 100)
print("J. free-run predictability horizons (a1_ds_w100 / cyber), radii 1e-4..1e-2")
for tag, f in [("ds_w100", os.path.join(BIG, "a1_ds_w100", "free_run", "free_run.npz")),
               ("cyber_w100", os.path.join(BIG, "a1_cyber_w100", "free_run", "free_run.npz")),
               ("ds_w200_large", os.path.join(BIG, "a1_ds_w200_large", "free_run", "free_run.npz")),
               ("ds_w500_large", os.path.join(BIG, "a1_ds_w500_large", "free_run", "free_run.npz"))]:
    if not os.path.exists(f):
        print(f" {tag}: no file")
        continue
    zz = np.load(f, allow_pickle=True)
    if "pred" in zz.files:
        pred = zz["pred"]
        rr = zz["radii"] if "radii" in zz.files else None
        for i in range(pred.shape[0]):
            print(f" {tag} r={rr[i] if rr is not None else '?'}: pred n={len(pred[i])}"
                  f" mean={np.mean(pred[i]):.1f} min={np.min(pred[i])} max={np.max(pred[i])}")
