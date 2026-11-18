import json, numpy as np
d = json.load(open(r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2\exp1_d2_scaling.json'))
e2 = json.load(open(r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2\exp2_synthloop.json'))

def slopes_at(C, eps_frac=None):
    C = np.array(C); n = len(C)
    # eps grid: 80 points, geomspace 0.002..2.2 (times scale)
    eps = np.geomspace(0.002, 2.2, n)
    out = {}
    for tg in (0.05, 0.3, 0.5, 0.9):
        k = int(np.argmax(C >= tg)) if (C >= tg).any() else -1
        if k <= 0:
            out[tg] = None; continue
        lo = max(k - 2, 0); hi = min(k + 2, n - 1)
        seg = C[lo:hi + 1]
        m = seg > 0
        if m.sum() < 3:
            out[tg] = None; continue
        s = np.polyfit(np.log(eps[lo:hi + 1][m]), np.log(seg[m]), 1)[0]
        out[tg] = round(float(s), 2)
    return out

print('exp1 Theiler effect on slope-at-C (smoothed 5-pt):')
for k in ['cyber', 'interstellar', 'launch']:
    for w, v in d[k]['theiler'].items():
        print(f'  {k:13s} w={w:>4s}:', slopes_at(v['C']))

print()
print('exp2 fidelity + cycle-var + slopes:')
for k, v in e2.items():
    print(f'  {k}: P={v["P"]} cycle-var={v["frac_var_explained_by_cycle"]:.3f} '
          f'rho_res={v["resid_cycle_autocorr"]:.2f} fidelity={ {t: round(x,3) for t,x in v["fidelity"].items()} }')
