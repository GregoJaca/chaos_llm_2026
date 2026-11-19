"""digest.py -- print a compact digest of all numbers needed for REPORT.md.
Purely a reader: recomputes nothing, prints from numbers.json + caches.
"""
import json
import os
import numpy as np

from dimlib import OUT, CACHE

TAGS = ["deepseek", "qwen3_4b", "phi4mini", "qwen3_5"]


def jget(js, *keys, default=None):
    cur = js
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur if cur is not None else default


def main():
    with open(os.path.join(OUT, "numbers.json")) as f:
        js = json.load(f)

    print("=" * 72)
    print("CHECK:", js.get("check", {}).get("tags_found"))
    A = js.get("A", {}).get("per_tag", {})
    print("\n===== A: vocabulary")
    for t in TAGS:
        a = A.get(t, {})
        print(f"\n-- {t} (D={a.get('D')}, V={a.get('V')}, L={a.get('L')}, "
              f"tied={a.get('tied')})")
        print(f"   E: PR={a['E_spec']['PR_lambda_s2']:.1f} "
              f"srank={a['E_spec']['stable_rank_sum_s2_over_s1_2']:.0f} "
              f"topvar={a['E_spec']['top_var_frac_s1_2_over_sum_s2']:.3f} "
              f"eff90={a['E_spec']['eff_rank_90']} "
              f"eff99={a['E_spec']['eff_rank_99']} "
              f"alpha={a['E_spec']['powerlaw_tail_alpha_i0_10_i1_halfD']:.2f}"
              f" (R2={a['E_spec']['powerlaw_tail_R2']:.3f})")
        print(f"   W: PR={a['W_spec']['PR_lambda_s2']:.1f} "
              f"eff90={a['W_spec']['eff_rank_90']} "
              f"topvar={a['W_spec']['top_var_frac_s1_2_over_sum_s2']:.3f}")
        print(f"   rows: cosE={a['mean_cos_rows_E']:+.3f} "
              f"cosW={a['mean_cos_rows_W']:+.3f} coneE={a['cone_rows_E']:.2f}"
              f"(floor {a['cone_rows_E_noise_floor_2sigma']:.3f}) "
              f"s1^2/Vmu^2={a['s1sq_over_V_mu2_E']:.2f} "
              f"dimcos={a['mean_cos_dims_from_GE_20k']:+.3f}")
        print(f"   hidden: cos@1={a['mean_cos_h_block1']:+.3f} "
              f"cos@last={a['mean_cos_h_layerLastRaw']:+.3f} "
              f"cone@1={a['cone_h_block1']:.2f} "
              f"cone@last={a['cone_h_layerLastRaw']:.2f} "
              f"topvar@last={a['top_var_frac_h_layerLastRaw']:.3f}")
        print(f"   norm-vs-u1: r={a['pearson_abs_proj_u1_vs_norm_E']:.2f} "
              f"frac||e||^2 in u1={a['frac_norm2_in_u1_E']:.3f} "
              f"in u2={a['frac_norm2_in_u2_E']:.3f}")

    print("\n===== B: hidden states")
    B = js.get("B", {}).get("per_tag", {})
    for t in TAGS:
        b = B.get(t, {})
        z = np.load(os.path.join(CACHE, f"{t}_layermeas.npz"))
        L = int(z["L"])
        print(f"\n-- {t} (D={b.get('D')}, L={b.get('L')})")
        print(f"   emb idx0: PR={b['embedding_idx0']['pr_c']:.1f} "
              f"eff90={b['embedding_idx0']['eff90_c']} "
              f"cos={b['embedding_idx0']['mean_cos']:.3f}")
        print(f"   blk1: PR={b['block1_idx1']['pr_c']:.1f} "
              f"eff90={b['block1_idx1']['eff90_c']} "
              f"mle20={b['block1_idx1']['mle20']:.1f} "
              f"2nn={b['block1_idx1']['twonn']:.1f} "
              f"cd={b['block1_idx1']['corrdim']:.1f} "
              f"cos={b['block1_idx1']['mean_cos']:.3f} "
              f"cone={b['block1_idx1']['cone']:.2f} "
              f"slope={b['block1_idx1']['slope']:.2f}")
        lr = b["last_raw_block"]
        print(f"   last raw: PR={lr['pr_c']:.1f} eff90={lr['eff90_c']} "
              f"eff99={lr['eff99_c']} mle20={lr['mle20']:.1f} "
              f"(excl {lr['mle20_nexcl']}) "
              f"perprompt={np.round(lr['mle20_perprompt'],1)} "
              f"2nn={lr['twonn']:.1f} cd={lr['corrdim']:.1f} "
              f"cos={lr['mean_cos']:.3f} cone={lr['cone']:.2f} "
              f"topvar={lr['topfrac_c']:.3f} slope={lr['slope']:.2f}")
        print(f"   postnorm: PR={b['postnorm_idxL']['pr_c']:.1f} "
              f"eff90={b['postnorm_idxL']['eff90_c']} "
              f"mle20={b['postnorm_idxL']['mle20']:.1f} "
              f"cos={b['postnorm_idxL']['mean_cos']:.3f}")
        wm = b["write_mean_over_blocks"]
        print(f"   write mean: ||d||/||h||={wm['norm_ratio']:.3f} "
              f"eff90={wm['eff90_c']:.0f} pr={wm['pr_c']:.1f} "
              f"cos={wm['cos_h']:.3f} cone={wm['cone']:.2f} "
              f"rad>0.5={wm['frac_radial_gt_0p5']:.3f}")
        print(f"   write first: ||d||/||h||="
              f"{b['write_first_block']['norm_ratio']:.2f} "
              f"eff90={b['write_first_block']['eff90_c']}")
        print(f"   write last (block+norm): "
              f"||d||/||h||={b['write_last_blockish']['norm_ratio']:.2f} "
              f"cos={b['write_last_blockish']['cos_h']:.3f}")
        # state-vs-write rank comparison
        wr = z["w_eff90_c"][:-1]
        sr = z["m_eff90_c"][1:-1]
        print(f"   mean write eff90/state eff90 = "
              f"{wr.mean()/sr.mean():.2f}")
        print(f"   PR with-mean/centered at last raw: "
              f"{z['m_pr_m'][L-1]/z['m_pr_c'][L-1]:.1f}x")
        print(f"   localdim_t: idx1 median="
              f"{np.nanmedian(z['localdim_t'][1]):.1f}, idx{L-1} median="
              f"{np.nanmedian(z['localdim_t'][L-1]):.1f}")

    print("\n===== C: logits")
    C = js.get("C", {}).get("per_tag", {})
    for t in TAGS:
        c = C.get(t, {})
        ls, ss = c["logit_spectrum"], c["postnorm_state_same_positions"]
        print(f"\n-- {t}: logit PR={ls['PR_lambda']:.1f} "
              f"eff90={ls['eff_rank_90']}/{ls['n_positions']} "
              f"eff99={ls['eff_rank_99']} "
              f"topvar={ls['top_var_frac']:.3f} "
              f"alpha={ls['powerlaw_alpha_i3_64']:.2f}")
        print(f"   state same-pos: PR={ss['PR_lambda']:.1f} "
              f"eff90={ss['eff_rank_90']} (collapse ratio "
              f"{ls['eff_rank_90']/ss['eff_rank_90']:.2f})")
        g = c["top2_gap"]
        print(f"   gap: mean={g['mean']:.2f} med={g['median']:.2f} "
              f"q05={g['q05']:.2f} q95={g['q95']:.2f}")
        print(f"   centered ranks: {c['centered_logit_ranks']}")

    print("\n===== D")
    D = js.get("D", {})
    print("\n--- D1 spearman table (vs kick gain):")
    st = D.get("D1", {}).get("spearman_table", {})
    for m, row in st.items():
        print(f"   {m:38s} rho_kick={row['spearman_vs_kickgain']:+.2f} "
              f"rho_lambda={row['spearman_vs_lambda']:+.2f} "
              f"rho_spread={row['spearman_vs_dirspread']:+.2f} "
              f"vals={np.round(list(row['values'].values()),2)}")
    print("\n--- D1 scalars:")
    for t in TAGS:
        s = D.get("D1", {}).get("scalars", {}).get(t, {})
        print(f"   {t}: IDpp={s.get('ID_mle20_perprompt_late'):.1f} "
              f"IDpool={s.get('ID_mle20_pooled_late'):.1f} "
              f"PR={s.get('PR_c_late'):.1f} eff90={s.get('eff90_c_late'):.0f}"
              f" cos={s.get('mean_cos_late'):.2f} "
              f"dhfield_eff90={s.get('dhfield_eff90_c')} "
              f"dhfield_PR={s.get('dhfield_PR_c'):.1f}")
    print("\n--- D2 within-model spearman vs gain:")
    for t in TAGS:
        d2t = D.get("D2", {}).get(t, {})
        if not d2t:
            continue
        n = d2t["n_layers_a1"]
        cors = d2t["correlations"]
        print(f"   {t} (n_layers={n}, dh0/r={d2t['dh_layer0_over_r']:.3f}):")
        for m, cc in cors.items():
            print(f"      {m:22s} vs gain {cc['spearman_vs_gain']:+.2f} "
                  f"(p={cc['pearson_vs_gain']:+.2f}) "
                  f"vs rate {cc['spearman_vs_rate']:+.2f} "
                  f"vs marg {cc['spearman_vs_marginal']:+.2f}")
        print(f"      G_l first->last: {d2t['G_l'][1]:.2f} -> "
              f"{d2t['G_l'][-1]:.2f}; lambda_l: "
              f"{np.round(d2t['lambda_l'][1:4],4)} ... "
              f"{np.round(d2t['lambda_l'][-2:],4)}")
    print("\n--- D3 radial:")
    for t in TAGS:
        d3t = D.get("D3", {}).get(t, {})
        if not d3t:
            continue
        print(f"   {t}: |cos| per layer "
              f"{d3t['mean_abs_cos'][0]:.2f} -> "
              f"{d3t['mean_abs_cos'][-1]:.2f}; frac|cos|>0.5: "
              f"{d3t['frac_radial_gt_0.5'][0]:.2f} -> "
              f"{d3t['frac_radial_gt_0.5'][-1]:.2f}")
    print("   subspace overlap k=32:",
          D.get("D3", {}).get("subspace_overlap_k32"))
    print("\n--- D4 window sweep:")
    for t, r in D.get("D4", {}).items():
        print(f"   {t}: margins_equal={r['margins_equal_across_W']} "
              f"last-layer gain {r['last_layer_gain']} "
              f"ratio={r['gain_ratio_W4096_over_W60']:.2f} "
              f"predT={r['predT_mean']}")
    print("\n--- data notes:", js.get("D", {}).get("data_notes"))


if __name__ == "__main__":
    main()
