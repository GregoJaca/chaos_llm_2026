"""Task 4: depth — how much of the final-layer response exists already at
layers 7/14/21, and where the near-tie kick (s=16) behaves differently.

(a) ratio dh(tau, L7) / dh(tau, L27) vs lag, per s          -> early-layer share
(b) ratio dh(tau, L21) / dh(tau, L27) vs lag, per s
(c) immediate gain  dh(0, L27)/dh(0, L7)  vs  echo gain  dh(echo, L27)/dh(echo, L7):
    late layers amplify recall echoes (x8-28) far more than the immediate response
(d) response relative to the state norm ||h|| (estimated from cos) vs layer:
    the s=16 near-tie kick stays ~flat with depth, ordinary positions shrink.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import load_emb, style, OUT, SCOL, base_state_norms

D = load_emb()
steps, layers = D['steps'], D['layers']
M = D['M'][3]                     # r=0.1, mean over dirs: [s, layer, t]
NORMS = base_state_norms(D)       # ||h_base(s, L)|| from cos at tau=0
ECHOT = {8: 105, 16: 113, 32: 124, 50: 149}

print('=== immediate head dh(tau=0) per layer (r=0.1, mean dirs) ===')
print('       ' + '  '.join(f'L{L:>7d}' for L in layers))
for i_s, s in enumerate(steps):
    print(f'  s={s:3d} ' + '  '.join(f'{M[i_s, i_l, int(s)]:8.3f}' for i_l in range(4)))

print('\n=== share of final-layer head present at each layer (dh(0,L)/dh(0,L27)) ===')
for i_s, s in enumerate(steps):
    shares = [M[i_s, i_l, int(s)] / M[i_s, 3, int(s)] for i_l in range(4)]
    print(f'  s={s:3d}: L7 {shares[0]:5.2f}  L14 {shares[1]:5.2f}  L21 {shares[2]:5.2f}')

print('\n=== tail share (mean ratio over tau in [80,99]) ===')
for i_s, s in enumerate(steps):
    row = []
    for i_l in range(4):
        a = M[i_s, i_l, int(s) + 80:int(s) + 100]
        b = M[i_s, 3, int(s) + 80:int(s) + 100]
        row.append(np.nanmean(a / b))
    print(f'  s={s:3d}: L7/L27 {row[0]:5.2f}  L14/L27 {row[1]:5.2f}  L21/L27 {row[2]:5.2f}')

print('\n=== spread across s at tau=0 (max/min) grows with layer depth ===')
for i_l, L in enumerate(layers):
    heads = M[:, i_l, [int(s) for s in steps]].diagonal() if False else \
        np.array([M[i_s, i_l, int(s)] for i_s, s in enumerate(steps)])
    print(f'  L{L}: max/min across s = {heads.max()/heads.min():5.2f}  '
          f'(s=16 is max: {heads.max() == heads[1]})')

print('\n=== late-layer gain: head vs echo ===')
for i_s, s in enumerate(steps):
    g_head = M[i_s, 3, int(s)] / M[i_s, 0, int(s)]
    line = f'  s={s:3d}: head gain L7->L27 = {g_head:5.2f}'
    if s in ECHOT:
        t = ECHOT[s]
        g_echo = M[i_s, 3, t] / M[i_s, 0, t]
        line += f'   echo gain at t={t}: {g_echo:5.1f}'
    print(line)

print('\n=== relative-to-state response dh(0,L)/||h(L)|| in % ===')
for i_s, s in enumerate(steps):
    rel = [100 * M[i_s, i_l, int(s)] / NORMS[int(L)][i_s] for i_l, L in enumerate(layers)]
    print(f'  s={s:3d}: ' + '  '.join(f'{v:6.2f}%' for v in rel))

# ---------------- figure ----------------
style()
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

def ratio_panel(ax, i_l, title):
    for i_s, s in enumerate(steps):
        y27 = M[i_s, 3, int(s):int(s) + 100]
        yl = M[i_s, i_l, int(s):int(s) + 100]
        ax.plot(np.arange(len(y27)), yl / y27, color=SCOL[int(s)],
                lw=2.4 if s == 16 else 1.0, alpha=1.0 if s == 16 else 0.75,
                label='s=16' if s == 16 else None)
    ax.axhline(1.0, color='k', ls=':', lw=0.8)
    ax.set_xlabel('lag τ = t − s (tokens)')
    ax.set_ylabel(f'‖Δh(τ, L{layers[i_l]})‖ / ‖Δh(τ, L27)‖')
    ax.set_ylim(0, 1.6)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25, lw=0.4)

share7 = {int(s): M[i_s, 0, int(s)] / M[i_s, 3, int(s)] for i_s, s in enumerate(steps)}
share21 = {int(s): M[i_s, 2, int(s)] / M[i_s, 3, int(s)] for i_s, s in enumerate(steps)}
ratio_panel(axes[0, 0], 0,
            f'(a) layer 7 already carries {share7[50]:.0%}–{share7[150]:.0%} of the final response '
            f'for ordinary s — but only {share7[16]:.0%} for s=16 (near-tie)')
ratio_panel(axes[0, 1], 2,
            f'(b) layer 21 carries {share21[50]:.0%}–{share21[150]:.0%} for ordinary s, '
            f'{share21[16]:.0%} for s=16 — the kick is built late (L21→L27)')

ax = axes[1, 0]
xs = np.arange(len(steps))
gh = [M[i_s, 3, int(s)] / M[i_s, 0, int(s)] for i_s, s in enumerate(steps)]
ax.bar(xs, gh, color=[SCOL[int(s)] for s in steps], width=0.62,
       label='immediate response (τ=0)')
for i_s, s in enumerate(steps):
    if s in ECHOT:
        t = ECHOT[s]
        ge = M[i_s, 3, t] / M[i_s, 0, t]
        ax.plot([xs[i_s]], [ge], marker='D', ms=6, color='k', zorder=5)
        ax.annotate(f'{ge:.0f}×', (xs[i_s], ge), textcoords='offset points',
                    xytext=(12, -2), fontsize=7)
ax.set_yscale('log')
ax.set_xticks(xs)
ax.set_xticklabels([f's={int(s)}' for s in steps])
ax.set_ylabel('gain of layer 27 over layer 7 (log)')
ax.legend(loc='upper left')
ax.set_title('(c) late layers are an ECHO amplifier: recall bumps gain 8–28×, '
             'immediate heads only 1–4× (s=16 head: 4.2×)', fontweight='bold')
ax.grid(alpha=0.3, lw=0.4, axis='y')

ax = axes[1, 1]
for i_s, s in enumerate(steps):
    rel = [100 * M[i_s, i_l, int(s)] / NORMS[int(L)][i_s] for i_l, L in enumerate(layers)]
    ax.plot(layers, rel, marker='o', ms=3.5, color=SCOL[int(s)],
            lw=2.2 if s == 16 else 1.0, alpha=1.0 if s == 16 else 0.7,
            label='s=16 (near-tie)' if s == 16 else None)
ax.set_yscale('log')
ax.set_xlabel('layer (7 → 27; final layer is 27)')
ax.set_ylabel('‖Δh(τ=0)‖ / ‖h_base‖  (%)')
ax.legend(loc='upper right')
ax.set_title('(d) relative to the state norm the near-tie kick stays ~flat with depth\n'
             '(s=16: 4.3%→2.7%) while ordinary positions shrink ×3–10',
             fontweight='bold')
ax.grid(alpha=0.25, lw=0.4)

fig.suptitle('Depth: the ordinary response is complete by layer 7; the two things the deep '
             'layers add are the near-tie kick and the echo amplification',
             fontsize=11, fontweight='bold')
fig.savefig(OUT + r'\fig4_depth.png')
plt.close(fig)
print('\nsaved fig4_depth.png')
