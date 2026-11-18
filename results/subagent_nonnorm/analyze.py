"""Non-normality / depth-composition analysis (subagent_nonnorm).

CPU-only re-analysis of:
  results/local_data/nonnorm.npz           (5-layer x 6-token full-block/MLP Jacobian metrics + angles)
  results/local_data/depth/depth.npz       (8-token x 28-layer tangent (QR) depth spectrum)
  results/local_data/depth/fsle_depth.json (depth separation curves, OLD perturbation runs)
  results/local_data/w100_layers/a1.npz    (CONTEXT ONLY: baseline state norms norm_h0,
                                            teacher-forced response dh_norm/cos)

Core question: "components amplify, composition may not" -- is the layer-0 jump the
whole story, what does non-normality do, and what remains after normalizing by the
residual-stream scale?

Outputs (written next to this script):
  derived.npz   - all derived quantities
  summary.json - every number quoted in REPORT.md
  analysis_log.txt - console dump
"""
import json
import numpy as np
from scipy import stats

ROOT = 'chaos_llm_2026/results/local_data'
OUT = 'chaos_llm_2026/results/subagent_nonnorm'

log_lines = []


def log(s=''):
    print(s)
    log_lines.append(str(s))


# ---------------------------------------------------------------- load
dz = np.load(f'{ROOT}/depth/depth.npz')
lam = dz['lam'].astype(np.float64)          # [nt, nL, k] mean per-layer log SV of product
local = dz['local_gain'].astype(np.float64)  # [nt, nL] mean ||J_l q|| over propagated basis
hd = dz['h_depth'].astype(np.float64)        # [nt] sum of positive top-8 final exponents
dtok = np.asarray(dz['tokens'])              # generated-token positions
nL = lam.shape[1]
k = int(dz['k'])
nt = lam.shape[0]

dn = np.load(f'{ROOT}/nonnorm.npz')
mlp_s = dn['mlp_smax'].astype(np.float64)
mlp_r = dn['mlp_rho'].astype(np.float64)
mlp_k = dn['mlp_kappa'].astype(np.float64)
ful_s = dn['full_smax'].astype(np.float64)
ful_r = dn['full_rho'].astype(np.float64)
ful_k = dn['full_kappa'].astype(np.float64)
ang = dn['angles'].astype(np.float64)        # [6, 27] deg, angle(v_l, v_{l+1}) top input dirs
ntok = np.asarray(dn['tokens'])
sel_layers = np.asarray(dn['sel_layers'])     # MLP sampled layers
full_layers = np.array([0, 14, 27])          # full-block sampled layers (NaN elsewhere)

fsle = json.load(open(f'{ROOT}/depth/fsle_depth.json'))

da = np.load(f'{ROOT}/w100_layers/a1.npz')   # context only
layers_a = np.asarray(da['layers'])          # [0,1,2,3,7,14,21,27]
norm_h0 = da['norm_h0'].astype(np.float64)   # [8, 300] baseline state norms by layer/position
margins = da['margins'].astype(np.float64)   # [300] logit margins of baseline run

# ---------------------------------------------------------------- state-scale profile
# s(l) = ||h^{(l)}|| at the tracked generated position, log-interpolated from the
# 8 sampled layers; s(28) extrapolated with the local (21->27) slope.
def log_interp_norm(norm_row):
    y = np.log(np.asarray(norm_row, dtype=np.float64))
    out = np.interp(np.arange(28), layers_a, y)
    slope = (y[-1] - y[-2]) / (layers_a[-1] - layers_a[-2])
    return np.append(out, out[-1] + slope)   # length 29: layers 0..28

log_s_tok = np.stack([log_interp_norm(norm_h0[:, int(t)]) for t in dtok])   # [nt, 29]
gs_tok = np.exp(np.diff(log_s_tok, axis=1))                                 # [nt, 28] per-block scale ratio
log_s_agg = log_interp_norm(np.median(norm_h0, axis=1))
gs_agg = np.exp(np.diff(log_s_agg))

# ---------------------------------------------------------------- depth spectrum bookkeeping
S = lam * (np.arange(nL) + 1)[None, :, None]          # cumulative log SVs of product
g_inc = np.diff(np.concatenate([np.zeros((nt, 1, k)), S], axis=1), axis=1)  # [nt,nL,k]
g_mean = g_inc.mean(axis=2)                            # mean top-8 log gain per block
g_top1 = g_inc[:, :, 0]

# relative exponents: subtract the mean per-layer log scale growth
scale_rate = (np.cumsum(np.log(gs_tok), axis=1) / (np.arange(1, nL + 1))[None, :])[:, :, None]
lam_rel = lam - scale_rate                             # [nt, nL, k]
g_rel = g_inc - np.log(gs_tok)[:, :, None]             # [nt, nL, k] per-block relative log gains

# sustained tail exponents (blocks 1..27), per direction and token
c_abs = (S[:, -1, :] - S[:, 0, :]) / (nL - 1)          # [nt, k]
c_rel = c_abs - np.log(gs_tok[:, 1:]).mean(axis=1)[:, None]

# expansion entropy curves (top-8 lower bound)
h_curve = np.maximum(lam, 0).sum(axis=2)              # [nt, nL]
h_rel_curve_maxexp = lam_rel.max(axis=2)              # [nt, nL] largest relative exponent

# ---------------------------------------------------------------- scalars for the report
med = lambda a: float(np.median(a))
res = {}

res['scale_block0_median'] = med(gs_tok[:, 0])
res['scale_block0_range'] = [float(gs_tok[:, 0].min()), float(gs_tok[:, 0].max())]
res['scale_total_median'] = med(np.exp(log_s_tok[:, -1] - log_s_tok[:, 0]))
res['scale_bands'] = {}
res['aligned_bands'] = {}
res['relative_bands'] = {}
BANDS = [(1, 7), (7, 14), (14, 21), (21, 28)]
for a, b in BANDS:
    key = f'{a}-{b - 1}'
    res['scale_bands'][key] = med(gs_tok[:, a:b])
    res['aligned_bands'][key] = med(local[:, a:b])
    res['relative_bands'][key] = med(local[:, a:b] / gs_tok[:, a:b])

res['aligned_block0'] = local[:, 0].tolist()
res['relative_block0'] = (local[:, 0] / gs_tok[:, 0]).tolist()

res['max_relative_exponent_all'] = float(lam_rel.max())
res['per_token_max_relative_exponent'] = lam_rel.max(axis=(1, 2)).tolist()
res['per_token_maxrel_at_final'] = lam_rel[:, -1, :].max(axis=1).tolist()

res['transient_share_top1'] = (S[:, 0, 0] / S[:, -1, 0]).tolist()  # S(0)/S(27)
res['sustained_abs_mean_top8'] = c_abs.mean(axis=1).tolist()
res['sustained_rel_mean_top8'] = c_rel.mean(axis=1).tolist()
res['sustained_top1'] = ((S[:, -1, 0] - S[:, 0, 0]) / (nL - 1)).tolist()
fitc = []
layv = np.arange(nL)
Amat = np.stack([1 / (layv + 1), np.ones_like(layv)], 1)
for i in range(nt):
    coef, *_ = np.linalg.lstsq(Amat, lam[i, :, 0], rcond=None)
    fitc.append(float(coef[1]))
res['sustained_fit_c_top1'] = fitc
res['h_depth'] = hd.tolist()
res['h_final_rel_sum'] = np.minimum(lam_rel[:, -1, :], 0).sum(axis=1).tolist()

res['product_top1_abs'] = np.exp(S[:, -1, 0]).tolist()
res['product_top1_rel'] = np.exp(S[:, -1, 0] + log_s_tok[:, 0] - log_s_tok[:, -1]).tolist()

# final spectra
res['final_spectra'] = {int(t): lam[i, -1].tolist() for i, t in enumerate(dtok)}

# ---------------------------------------------------------------- FSLE depth curves
fsle_c = {float(r): np.asarray(v) for r, v in fsle['curves'].items()}
fsle_tot = {r: float(c[-1] / c[0]) for r, c in fsle_c.items()}
fsle_peak = {r: float(c.max() / c[0]) for r, c in fsle_c.items()}
res['fsle_total_growth'] = {str(r): v for r, v in fsle_tot.items()}
res['fsle_peak_growth'] = {str(r): v for r, v in fsle_peak.items()}
res['fsle_lambda'] = {str(rho): [x['lam'] for x in rows]
                      for rho, rows in fsle['res'].items()}
# growth concentrated in first two recorded layers?
res['fsle_share_by_layer2'] = {str(r): float(c[2] / c[-1]) for r, c in fsle_c.items()}
# scale-normalized FSLE (using aggregate median norm profile)
res['fsle_rel_total'] = {str(r): float((c[-1] / c[0]) / (np.exp(log_s_agg[27]) / np.exp(log_s_agg[0])))
                         for r, c in fsle_c.items()}

# ---------------------------------------------------------------- non-normality numbers
nn = {}
nn['full_smax_by_layer'] = {int(l): [float(x) for x in ful_s[ful_s[:, l] == ful_s[:, l], l]]
                            for l in full_layers}
nn['full_rho_by_layer'] = {int(l): [float(x) for x in ful_r[ful_r[:, l] == ful_r[:, l], l]]
                           for l in full_layers}
nn['full_kappa_by_layer'] = {int(l): [float(x) for x in ful_k[ful_k[:, l] == ful_k[:, l], l]]
                             for l in full_layers}
nn['mlp_smax_by_layer'] = {int(l): mlp_s[mlp_s[:, l] > 0, l].tolist() for l in sel_layers}
nn['mlp_rho_by_layer'] = {int(l): mlp_r[mlp_r[:, l] > 0, l].tolist() for l in sel_layers}
nn['mlp_kappa_by_layer'] = {int(l): mlp_k[mlp_k[:, l] > 0, l].tolist() for l in sel_layers}
# token 143 / layer 14 outlier
ti143 = list(ntok).index(143)
nn['token143_layer14'] = dict(smax=float(ful_s[ti143, 14]), rho=float(ful_r[ti143, 14]),
                              kappa=float(ful_k[ti143, 14]))
others = [i for i in range(len(ntok)) if i != ti143]
nn['layer14_others_smax'] = [float(ful_s[i, 14]) for i in others]
nn['layer14_others_kappa'] = [float(ful_k[i, 14]) for i in others]
nn['token143_margin'] = float(margins[143])
nn['token143_hdepth'] = float(hd[list(dtok).index(143)])

# angles
nn['angle_median'] = med(ang)
nn['angle_iqr'] = [float(x) for x in np.percentile(ang, [25, 75])]
nn['angle_min'] = float(ang.min())
nn['angle_argmin'] = [int(ntok[np.unravel_index(np.argmin(ang), ang.shape)[0]]),
                      int(np.unravel_index(np.argmin(ang), ang.shape)[1])]
nn['angle_max'] = float(ang.max())
nn['abs_cos_median'] = med(np.abs(np.cos(np.radians(ang))))
nn['random_abs_cos_expected'] = float(np.sqrt(2 / np.pi) / np.sqrt(1536))
slope, icpt, rval, pval, stderr = stats.linregress(np.arange(ang.shape[1]), np.median(ang, axis=0))
nn['angle_layer_trend_slope'] = float(slope)
nn['angle_layer_trend_p'] = float(pval)

# per-token angle medians
nn['angle_median_by_token'] = {int(t): med(ang[i]) for i, t in enumerate(ntok)}

# rank-one misalignment budget: predicted per-layer log gain from measured angles
cos2 = np.cos(np.radians(ang)) ** 2
nn['cos2_median'] = med(cos2)
nn['cos2_iqr'] = [float(x) for x in np.percentile(cos2, [25, 75])]

sig14 = med(ful_s[ful_s[:, 14] == ful_s[:, 14], 14])   # ~3.2
sig27 = med(ful_s[ful_s[:, 27] == ful_s[:, 27], 27])   # ~7.8
pred14 = np.log1p((sig14 - 1) * nn['cos2_median'])
pred27 = np.log1p((sig27 - 1) * nn['cos2_median'])
meas14 = med(g_mean[:, 14:21])    # blocks 14..20 median per-layer log gain
meas27 = med(g_mean[:, 21:28])
nn['misalign_pred_gain_L14'] = float(pred14)
nn['misalign_pred_gain_L27'] = float(pred27)
nn['measured_gain_blocks14_21'] = float(meas14)
nn['measured_gain_blocks21_27'] = float(meas27)

# gain budget at sampled full layers: component vs aligned vs scale
# anchor block band: block 0 alone; blocks 14..20 for L14; blocks 21..27 for L27
budget = {}
for l in full_layers:
    rows = ful_s[:, l] == ful_s[:, l]
    smax = med(ful_s[rows, l])
    sl = slice(0, 1) if l == 0 else (slice(14, 21) if l == 14 else slice(21, 28))
    band = g_mean[:, sl].mean()
    sc = med(gs_tok[:, sl])
    budget[int(l)] = dict(log_smax=float(np.log(smax)), log_aligned=float(band),
                          log_scale=float(np.log(sc)))
nn['budget'] = budget

# naive chains vs measured product (conservative: use the TAMEST measured full-block sigma
# for all 27 post-embedding blocks; layer 0 uses its measured sigma)
sig0 = med(ful_s[:, 0])
sig_min = np.nanmin(ful_s[:, full_layers[1:]])
rho0 = med(ful_r[:, 0])
rho_mid = med(ful_r[:, 14])
rho_late = med(ful_r[:, 27])
nn['naive_chain_full'] = float(sig0 * sig_min ** 27)
nn['naive_chain_rho'] = float(rho0 * rho_mid ** 13 * rho_late ** 13)
nn['measured_product_top1_median'] = med(np.exp(S[:, -1, 0]))
nn['fsle_growth_median'] = med(list(fsle_tot.values()))
nn['scale_total_median_dup'] = res['scale_total_median']
nn['mlp_chain_L27_single'] = float(sig0 * med(mlp_s[:, 27]) ** 1)  # one late MLP block alone
res['nonnorm'] = nn

# ---------------------------------------------------------------- exploratory correlations (N=8)
m_at = margins[dtok]
rho_m, p_m = stats.spearmanr(hd, m_at)
rho_g0, p_g0 = stats.spearmanr(hd, local[:, 0])
rho_sc, p_sc = stats.spearmanr(hd, np.exp(log_s_tok[:, -1] - log_s_tok[:, 0]))
res['spearman'] = dict(h_vs_margin=[float(rho_m), float(p_m)],
                       h_vs_block0gain=[float(rho_g0), float(p_g0)],
                       h_vs_scalegrowth=[float(rho_sc), float(p_sc)])
res['margins_at_depth_tokens'] = m_at.tolist()

# ---------------------------------------------------------------- context: teacher-forced response
# (a1.npz used only for state norms and a fixed-t depth profile; the t-axis response
#  regime analysis belongs to the A1 subagent.)
radii = np.asarray(da['radii'])
ri03 = list(radii).index(0.03)
dhn = da['dh_norm'][ri03].astype(np.float64)      # [20 dirs, 8 layers, 300 pos]
ctx_t = [0, 16, 50, 100, 143]
s_layers = norm_h0.astype(np.float64)
ctx = {}
for t in ctx_t:
    p = np.median(dhn[:, :, t], axis=0)           # [8]
    ctx[str(t)] = dict(abs=p.tolist(), rel=(p / np.median(norm_h0, axis=1)).tolist())
res['context_response_r0.03'] = ctx

# tangential fraction of the response (r=2, where cos is numerically resolvable)
ri2 = list(radii).index(2.0)
dhn2 = da['dh_norm'][ri2].astype(np.float64)
csl = da['cos'][ri2].astype(np.float64)
d_perp = np.median(norm_h0, axis=1)[None, :, None] * np.sqrt(2 * np.maximum(0, 1 - csl))
frac = np.sqrt(np.median((d_perp ** 2) / np.maximum(dhn2 ** 2, 1e-30), axis=(0, 2)))
res['tangential_fraction_r2_by_layer'] = frac.tolist()

# ---------------------------------------------------------------- console dump
log('=== DEPTH SPECTRUM (diagonal-block tangent product, 8 tokens) ===')
log('state scale: block0 x{:.1f} (range {:.1f}-{:.1f}), total x{:.0f}'.format(
    res['scale_block0_median'], *res['scale_block0_range'], res['scale_total_median']))
log('aligned gain block0 per token: {}'.format(np.round(local[:, 0], 2)))
log('relative gain block0 per token: {}'.format(np.round(local[:, 0] / gs_tok[:, 0], 3)))
for a, b in BANDS:
    log('blocks {}-{}: aligned {:.3f} scale {:.3f} rel {:.3f}'.format(
        a, b - 1, med(local[:, a:b]), med(gs_tok[:, a:b]), med(local[:, a:b] / gs_tok[:, a:b])))
log('max relative exponent over all tokens/layers/dirs: {:.4f}'.format(lam_rel.max()))
log('transient share S(0)/S(27): {}'.format(np.round(res['transient_share_top1'], 3)))
log('sustained abs/rel mean top8: {}'.format(np.round(c_abs.mean(axis=1), 3)))
log('  relative:                   {}'.format(np.round(c_rel.mean(axis=1), 3)))
log('product top-1 abs: {}'.format(np.round(np.exp(S[:, -1, 0]), 1)))
log('product top-1 rel: {}'.format(np.round(np.exp(S[:, -1, 0] + log_s_tok[:, 0] - log_s_tok[:, -1]), 3)))
log('')
log('=== FSLE depth ===')
log('total growth d(27)/d(0): {}'.format({r: round(v, 1) for r, v in fsle_tot.items()}))
log('share of final reached by recorded layer 2: {}'.format(
    {r: round(v, 2) for r, v in res['fsle_share_by_layer2'].items()}))
log('scale-normalized total: {}'.format({r: round(v, 3) for r, v in res['fsle_rel_total'].items()}))
log('')
log('=== NON-NORMALITY ===')
log('full L0: smax {:.0f}-{:.0f} kappa {:.1f}-{:.1f}'.format(
    min(nn['full_smax_by_layer'][0]), max(nn['full_smax_by_layer'][0]),
    min(nn['full_kappa_by_layer'][0]), max(nn['full_kappa_by_layer'][0])))
log('full L14 (excl t143): smax {:.2f}-{:.2f}, kappa {:.2f}-{:.2f}'.format(
    min(nn['layer14_others_smax']), max(nn['layer14_others_smax']),
    min(nn['layer14_others_kappa']), max(nn['layer14_others_kappa'])))
log('token143 L14: smax {:.2f} rho {:.2f} kappa {:.2f} (margin {:.3f})'.format(
    nn['token143_layer14']['smax'], nn['token143_layer14']['rho'],
    nn['token143_layer14']['kappa'], nn['token143_margin']))
log('full L27: smax {:.1f}-{:.1f} kappa {:.1f}-{:.1f}'.format(
    min(nn['full_smax_by_layer'][27]), max(nn['full_smax_by_layer'][27]),
    min(nn['full_kappa_by_layer'][27]), max(nn['full_kappa_by_layer'][27])))
log('MLP smax by layer: ' + str({int(l): round(float(np.median(v)), 1) for l, v in nn['mlp_smax_by_layer'].items()}))
log('MLP rho  by layer: ' + str({int(l): round(float(np.median(v)), 1) for l, v in nn['mlp_rho_by_layer'].items()}))
log('angles: median {:.1f} IQR {:.1f}-{:.1f} min {:.1f} (t={}, l={}) max {:.1f}'.format(
    nn['angle_median'], *nn['angle_iqr'], nn['angle_min'], *nn['angle_argmin'], nn['angle_max']))
log('|cos| median {:.3f} vs random-expected {:.3f}; cos^2 median {:.4f}'.format(
    nn['abs_cos_median'], nn['random_abs_cos_expected'], nn['cos2_median']))
log('angle layer trend: {:+.3f} deg/layer, p={:.2f}'.format(nn['angle_layer_trend_slope'], nn['angle_layer_trend_p']))
log('misalignment model: pred ln-gain L14 {:.3f} (measured blocks 14-21 {:.3f}); '
    'L27 {:.3f} (measured 21-27 {:.3f})'.format(
        nn['misalign_pred_gain_L14'], nn['measured_gain_blocks14_21'],
        nn['misalign_pred_gain_L27'], nn['measured_gain_blocks21_27']))
log('gain budget (nats/layer): ' + json.dumps(nn['budget']))
log('naive sigma-chain x{:.1e}; rho-chain x{:.1e}; measured aligned product x{:.1f}; '
    'FSLE x{:.0f}; state scale x{:.0f}'.format(
        nn['naive_chain_full'], nn['naive_chain_rho'], nn['measured_product_top1_median'],
        nn['fsle_growth_median'], nn['scale_total_median_dup']))
log('')
log('=== exploratory (N=8) ===')
log('margins at depth tokens: {}'.format(np.round(m_at, 3)))
log('spearman h_depth vs margin: {:.2f} (p={:.2f}); vs block0 gain: {:.2f} (p={:.2f}); '
    'vs scale growth: {:.2f} (p={:.2f})'.format(
        rho_m, p_m, rho_g0, p_g0, rho_sc, p_sc))
log('tangential fraction of response (r=2) by layer: {}'.format(np.round(frac, 3)))
log('context fixed-t response rel (r=0.03): ' + json.dumps(
    {t: [round(x, 6) for x in ctx[t]['rel']] for t in ctx}))

# ---------------------------------------------------------------- save
np.savez(f'{OUT}/derived.npz',
         S=S, g_inc=g_inc, g_mean=g_mean, g_top1=g_top1,
         lam=lam, lam_rel=lam_rel, g_rel=g_rel,
         local=local, hd=hd, dtok=dtok,
         log_s_tok=log_s_tok, gs_tok=gs_tok, log_s_agg=log_s_agg, gs_agg=gs_agg,
         c_abs=c_abs, c_rel=c_rel, h_curve=h_curve,
         h_rel_maxexp=h_rel_curve_maxexp,
         mlp_s=mlp_s, mlp_r=mlp_r, mlp_k=mlp_k,
         ful_s=ful_s, ful_r=ful_r, ful_k=ful_k,
         ang=ang, ntok=ntok, sel_layers=sel_layers, full_layers=full_layers,
         margins_at=m_at)
res['fsle_curves'] = {str(r): c.tolist() for r, c in fsle_c.items()}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(res, f, indent=1)
with open(f'{OUT}/analysis_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
print('\nsaved derived.npz, summary.json, analysis_log.txt')
