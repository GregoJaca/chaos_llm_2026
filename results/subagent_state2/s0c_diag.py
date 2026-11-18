import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data'
out = r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_state2'
d1 = np.load(base + r'\state_deep_emb\state2.npz')
dh, dm, margins = d1['dh'], d1['dm'], d1['margins']
steps, radii, layers = d1['steps'], d1['radii'], d1['layers']

M = dh[3].mean(axis=1)   # [8 s, 4 layers, 300 t], r=0.1, mean over 24 dirs

fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
for i_s, s in enumerate(steps):
    ax = axes.flat[i_s]
    for i_l, L in enumerate(layers):
        y = M[i_s, i_l, s:]
        ax.plot(np.arange(len(y)), y, label=f'L{L}')
    ax.set_yscale('log')
    ax.set_title(f's={s}')
    ax.set_xlabel('tau')
frag = np.where(margins < 0.3)[0]
for ax in axes.flat:
    for t in frag:
        ax.axvline(t - steps[0], color='r', ls=':', lw=0.5)
axes.flat[0].legend()
fig.suptitle('raw decay curves per s, r=0.1, mean over 24 dirs (red = fragile t, wrong offset per panel, ignore)')
fig.savefig(out + r'\diag_raw.png', dpi=110)

# stripe check: for s=8 (tau = t-s), where are the local maxima of L27 curve?
y = M[0, 3, 8:]
order = np.argsort(y)[::-1][:15]
print('s=8, L27: top-15 dh values at tau:', sorted(order + 8), ' t:', sorted((order + 8).tolist()))
print('   values:', y[order])
print('margins at those t:', margins[sorted((order + 8).tolist())])
print()
for i_s, s in enumerate(steps):
    y = M[i_s, 3, s:]
    for q in [0, 10, 30, 50, 99, 150, 299 - s]:
        if q < len(y):
            print(f's={s:3d} L27: tau={q:3d} dh={y[q]:.4f}', end='   ')
    print()
print()
print('s=16 L27, dh at t=116..125:', M[1, 3, 116:126])
print('margins t=116..125:', margins[116:126])
print('s=16 L27, dh at t=140..150:', M[1, 3, 140:151])
print('margins t=140..150:', margins[140:151])
