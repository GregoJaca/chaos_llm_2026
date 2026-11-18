import json
r = json.load(open(r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2\exp4_dynamics.json'))
for name in r:
    d = r[name]
    print(f'== {name} n={d["n"]} W={d["W"]} scale={d["scale"]:.1f}')
    print('  NN r1/scale quantiles:', {k: round(v, 3) for k, v in d['nn_dist_quantiles'].items()})
    print('  lag1 acorr median:', round(d['lag1_autocorr_median'], 3))
    for tag in d['pred']:
        nse = d['pred'][tag]['nse_knn']
        row = '  '.join(f'h{h}: {nse[h]["mean"]:+.3f}' for h in ['1', '2', '4', '8', '16', '32'])
        det = d['pred'][tag]['det']
        print(f'  {tag:9s} NSE {row}   det(h1/h8/h16): {det["1"]["mean"]:.2f}/{det["8"]["mean"]:.2f}/{det["16"]["mean"]:.2f}')
    print('  ridge data:', {h: round(v['nse_ridge'], 3) for h, v in d['ridge_data'].items()})
    print('  ridge phase:', {h: round(v['nse_ridge'], 3) for h, v in d['ridge_phase'].items()})
    print('  persist(data):', {h: round(v, 3) for h, v in d['pred']['data']['nse_persist'].items()})
