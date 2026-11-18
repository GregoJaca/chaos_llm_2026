import numpy as np
from scipy.ndimage import median_filter

base = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data'
d1 = np.load(base + r'\state_deep_emb\state2.npz')
dh, margins = d1['dh'], d1['margins']
steps, radii, layers = d1['steps'], d1['radii'], d1['layers']
M = dh[3].mean(axis=1)   # [8, 4, 300]

print('=== per-s curve summary, L27, r=0.1, mean over dirs ===')
print(f"{'s':>4} {'dh(0)':>7} {'dh(3)':>7} {'dh(10)':>7} | envelope(median21) at tau= 25, 50, 75, 99 | bumps (t, dh, margin)")
for i_s, s in enumerate(steps):
    y = M[i_s, 3, s:]
    env = median_filter(y, size=21, mode='nearest')
    # peaks: local max clearly above envelope
    bumps = []
    for tau in range(2, len(y) - 2):
        if y[tau] > 2.5 * env[tau] and y[tau] == y[max(0, tau-3):tau+4].max():
            bumps.append((s + tau, y[tau], margins[s + tau]))
    senv = '  '.join(f'{env[q]:7.3f}' for q in [25, 50, 75, 99])
    bumpstr = '  '.join(f't={t}:{v:.2f}(m={m:.2f})' for t, v, m in bumps[:6])
    print(f'{s:>4} {y[0]:7.3f} {y[3]:7.3f} {y[10]:7.3f} | {senv} | {bumpstr}')

print()
print('=== same at L7 ===')
for i_s, s in enumerate(steps):
    y = M[i_s, 0, s:]
    env = median_filter(y, size=21, mode='nearest')
    bumps = []
    for tau in range(2, len(y) - 2):
        if y[tau] > 2.5 * env[tau] and y[tau] == y[max(0, tau-3):tau+4].max():
            bumps.append((s + tau, y[tau]))
    senv = '  '.join(f'{env[q]:7.3f}' for q in [25, 50, 75, 99])
    bumpstr = '  '.join(f't={t}:{v:.2f}' for t, v in bumps[:6])
    print(f's={s:>4} {y[0]:7.3f} {y[3]:7.3f} {y[10]:7.3f} | {senv} | {bumpstr}')

print()
print('=== where in t are the bumps across all s (union)? ===')
allb = {}
for i_s, s in enumerate(steps):
    y = M[i_s, 3, s:]
    env = median_filter(y, size=21, mode='nearest')
    for tau in range(2, len(y) - 2):
        if y[tau] > 2.5 * env[tau] and y[tau] == y[max(0, tau-3):tau+4].max():
            allb.setdefault(int(s + tau), []).append(int(s))
for t in sorted(allb):
    print(f'  t={t}: bump for s={allb[t]}  margin={margins[t]:.3f}')

print()
print('=== dh at the two flip positions for every s (t=123 and t=143), L27 ===')
for i_s, s in enumerate(steps):
    if s < 123:
        print(f'  s={s:3d}: dh(t=123)={M[i_s,3,123]:.3f}  dh(t=143)={M[i_s,3,143]:.3f}')
