import numpy as np
base = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data'
np.set_printoptions(precision=4, suppress=True, linewidth=140)

d1 = np.load(base + r'\state_deep_emb\state2.npz')
dh, cosv, dm = d1['dh'], d1['cos'], d1['dm']
steps, radii, layers, margins = d1['steps'], d1['radii'], d1['layers'], d1['margins']
d2 = np.load(base + r'\state_freerun\sf.npz')
pred, ref = d2['pred'], d2['ref']
fsteps = d2['steps']

print('--- margin context ---')
print('median margin', np.median(margins), ' q25', np.percentile(margins, 25), ' q75', np.percentile(margins, 75))
print('margins[14..18]', margins[14:19])
print('margins[2]', margins[2])
print('margins[122..124]', margins[122:125], ' margins[142..144]', margins[142:145])

print('\n--- locate pre-injection leak (t < s), r=0.1 ---')
for i_s, s in enumerate(steps):
    sub = dh[3, i_s, :, :, :s]
    idx = np.unravel_index(np.argmax(sub), sub.shape)
    print(f's={s:3d}: max {sub.max():.3e} at (dir={idx[0]}, layer={idx[1]}, t={idx[2]})')

print('\n--- dm at t=s should be ~0 (dm[t] is gap change for token t, logits read at t-1) ---')
for i_s, s in enumerate(steps):
    print(f's={s:3d}: max|dm(t=s)| = {np.abs(dm[3, i_s, :, s]).max():.3e}   max|dm(t=s+1)| = {np.abs(dm[3, i_s, :, s+1]).max():.3e}')

print('\n--- SIGNED crossing test: dm[t] < -margins[t] (gap top1-top2 flips sign) ---')
flip_sets = {}
for i_s, s in enumerate(steps):
    dmm = dm[3, i_s]                      # [24, 300]
    fl = dmm[:, s+1:] < -margins[None, s+1:]
    n = fl.any(axis=1)
    if n.any():
        dirs = np.where(n)[0]
        firsts = [s + 1 + int(np.argmax(fl[i])) for i in dirs]
        flip_sets[s] = list(zip(dirs.tolist(), firsts))
        print(f's={s:3d}: {n.sum()}/24 dirs flip: {list(zip(dirs.tolist(), firsts))}')
    else:
        print(f's={s:3d}: 0/24')

print('\nfreerun actual changes:')
actual = {}
for i_s in range(2):
    s = int(fsteps[i_s])
    ch = []
    for idir in range(24):
        p = pred[i_s, idir]
        diff = (p != ref) & (p != -1)
        if diff.any():
            ch.append((idir, int(np.argmax(diff))))
    actual[s] = ch
    print(f'  s={s}: {ch}')
print('match with teacher-forced prediction:',
      {s: (sorted(flip_sets.get(s, [])) == sorted(actual.get(s, []))) for s in [16, 50]})

print('\n--- the 8 |dm|>margin crossings at s=16, t=123, sorted by dm ---')
dmm = dm[3, 1]  # s=16
order = np.argsort(dmm[:, 123])
print('margins[123] =', margins[123], ' 2x =', 2 * margins[123])
for i in order:
    mark = ' <== freerun flip' if i in [12, 13, 17, 23] else ''
    print(f'  dir={i:2d}: dm(123) = {dmm[i,123]:+.4f}  |dm(147)| = {abs(dmm[i,147]):.4f}{mark}')

print('\n--- s=50 t=143 ---')
dmm50 = dm[3, 3]
order = np.argsort(dmm50[:, 143])
for i in order[:6]:
    mark = ' <== freerun flip' if i == 13 else ''
    print(f'  dir={i:2d}: dm(143) = {dmm50[i,143]:+.4f}{mark}')
print('margins[143] =', margins[143])

print('\n--- base-state norm estimate ||h|| ~ dh/sqrt(2(1-cos)) at tau=0 (median over dirs) ---')
for i_l, L in enumerate(layers):
    row = []
    for i_s, s in enumerate(steps):
        c = 1 - cosv[3, i_s, :, i_l, s]
        h = dh[3, i_s, :, i_l, s] / np.sqrt(np.maximum(2 * c, 1e-12))
        row.append(f's={s}: {np.median(h):7.1f} (IQR {np.percentile(h,75)-np.percentile(h,25):5.1f})')
    print(f'  L{L}: ' + '  '.join(row))

print('\n--- L14 injection: dh at t=s, layer 14, vs r (is it exactly r?) ---')
d3 = np.load(base + r'\state_deep_L14\state2.npz')
dh3 = d3['dh']
for i_r, r in enumerate(d3['radii']):
    print(f'  r={r}: dh(t=s,L14) per s: ', [f'{dh3[i_r, i_s, :, 0, int(d3["steps"][i_s])].mean():.4f}' for i_s in range(4)])
print('  dirs norm check: ||v|| =', np.linalg.norm(d3['dirs'], axis=1)[:4], ' (emb file:',
      np.linalg.norm(d1['dirs'], axis=1)[:4], ')')

print('\n--- old state.npz (r=1e-3, L27, 32 dirs): fraction of head remaining at t=299 ---')
d4 = np.load(base + r'\state.npz')
dh4, steps4 = d4['dh'], d4['steps']
for i_s, s in enumerate(steps4):
    head = dh4[i_s, :, s].mean()
    tail = dh4[i_s, :, 299].mean()
    print(f'  s={s:3d}: head {head:.4f}  tail(t=299) {tail:.4f}  fraction {tail/head:.4f}')
