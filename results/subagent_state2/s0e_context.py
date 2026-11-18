import numpy as np
base = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data'
d1 = np.load(base + r'\state_deep_emb\state2.npz')
dh, margins = d1['dh'], d1['margins']
steps, layers = d1['steps'], d1['layers']
M = dh[3].mean(axis=1)

print('margins at the 8 injection positions:')
for s in steps:
    print(f'  s={s:3d}: margin={margins[s]:8.3f}   (percentile among 300: {100*np.mean(margins<margins[s]):.0f}%)')

print('\nlinearity of the echo: dh(r=0.1)/dh(r=0.003) at echo peaks (expect 33.3):')
for (i_s, s, t) in [(0, 8, 105), (1, 16, 113), (2, 32, 124), (3, 50, 149)]:
    a = dh[3, i_s, :, 3, t]
    b = dh[0, i_s, :, 3, t]
    print(f'  s={s:3d} t={t}: ratio = {a.mean()/b.mean():.1f}   (head ratio same s: {dh[3,i_s,:,3,s].mean()/dh[0,i_s,:,3,s].mean():.1f})')

print('\necho across layers (dh at echo t, per layer, r=0.1):')
for (i_s, s, t) in [(0, 8, 105), (1, 16, 113), (2, 32, 124), (3, 50, 149)]:
    vals = [f'L{L}:{M[i_s, i_l, t]:.2f}' for i_l, L in enumerate(layers)]
    print(f'  s={s:3d} t={t}: ' + '  '.join(vals))

print('\nold state.npz (r=1e-3, L27) same echo positions?')
d4 = np.load(base + r'\state.npz')
dh4, steps4 = d4['dh'], d4['steps']
for i_s, s in enumerate(steps4):
    y = dh4[i_s].mean(axis=0)
    peaks = []
    for t in range(max(2, s+2), 297):
        if y[t] == y[t-3:t+4].max() and y[t] > 3*np.median(y[s+1:]):
            peaks.append((t, y[t]))
    print(f'  s={s:3d}: peaks {[(t, round(float(v),4)) for t,v in peaks][:6]}')

print('\ns=150: initial growth check (dh(tau=0..6), L27):', M[6, 3, 150:157])
print('margins[148..152]:', margins[148:153])
print('\ns=200 head:', M[7, 3, 200:206], ' margins[200..205]:', margins[200:206])
