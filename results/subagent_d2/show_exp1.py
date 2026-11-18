import json
d = json.load(open(r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2\exp1_d2_scaling.json'))
print('trajectory   slopes-at-C (w=0):  .05    .1    .3    .5    .7    .9')
for k in d:
    s = d[k]['slope_at_C_w0']
    vals = '  '.join(f'{float(v):5.2f}' if v == v else '  nan' for v in s.values())
    print(f'{k:14s} {vals}')
print()
print('shuffle identity: max|dlogC| (all) / (C>1e-4)')
for k in d:
    si = d[k]['shuffle_identity']
    print(f"{k:14s} {si['max_abs_dlogC']:.5f} / {si['max_abs_dlogC_C_gt_1e-4']:.8f}")
