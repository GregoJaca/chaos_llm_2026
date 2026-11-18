"""Task 2: separability — is dh(s, t) ~ a(s) * b(tau)?

Build the log-response matrix A[s, tau] = log10(mean-dirs dh) at r=0.1 on the
common lag range tau in [0, 99] (valid for all 8 injection positions s), SVD it,
and report how much of the variance is in the first singular value (pure
separable product form a(s)*b(tau)), plus the residual structure and a direct
collapse test of the earlier claim "R(t,s) is not a function of lag".
"""
import numpy as np
import matplotlib.pyplot as plt
from common import load_emb, style, OUT, SCOL

D = load_emb()
steps, layers = D['steps'], D['layers']
TAU = 99

def logmat(i_l):
    A = np.zeros((len(steps), TAU + 1))
    for i_s, s in enumerate(steps):
        y = D['M'][3, i_s, i_l, int(s):int(s) + TAU + 1]
        A[i_s] = np.log10(y)
    return A

def svd_frac(A):
    U, S, Vt = np.linalg.svd(A - A.mean(), full_matrices=False)
    var = S ** 2
    return S, var / var.sum(), U, Vt

print('=== separability: variance fraction in singular value 1 (L=27) ===')
A = logmat(3)
S, frac, U, Vt = svd_frac(A)
print('singular values:', np.round(S, 3))
print('variance fractions:', np.round(frac, 4))
r1 = np.outer(U[:, 0] * S[0], Vt[0])  # rank-1 part of centered A
R = (A - A.mean()) - r1
print(f'first singular value holds {100*frac[0]:.1f}% of variance '
      f'(first two: {100*(frac[0]+frac[1]):.1f}%)')
print(f'residual RMS = {np.sqrt(np.mean(R**2)):.3f} decades '
      f'vs signal RMS = {np.sqrt(np.mean((A-A.mean())**2)):.3f} decades')
print('\nper layer: f1 fraction, residual RMS (decades):')
for i_l, L in enumerate(layers):
    S2, f2, _, _ = svd_frac(logmat(i_l))
    A2 = logmat(i_l)
    R2 = (A2 - A2.mean()) - np.outer(np.linalg.svd(A2 - A2.mean())[0][:, 0] * S2[0],
                                     np.linalg.svd(A2 - A2.mean())[2][0])
    print(f'  L{L}: f1={100*f2[0]:.1f}%  (f1+f2={100*(f2[0]+f2[1]):.1f}%)  residRMS={np.sqrt(np.mean(R2**2)):.3f}')

print('\n=== collapse test: dh(s,tau)/dh(s,0) — is the SHAPE a function of lag alone? ===')
norm = np.zeros((len(steps), TAU + 1))
for i_s, s in enumerate(steps):
    y = D['M'][3, i_s, 3, int(s):int(s) + TAU + 1]
    norm[i_s] = y / y[0]
for tau in [10, 25, 50, 75, 99]:
    col = norm[:, tau]
    print(f'  tau={tau:3d}: retained fraction across s: '
          f'min={col.min():.4f} max={col.max():.4f} spread={col.max()/col.min():.2f}x '
          f'CV={col.std()/col.mean():.2f}')

print('\n=== where the residual lives (L=27): rows/cols with largest residual RMS ===')
rowr = np.sqrt(np.mean(R ** 2, axis=1))
colr = np.sqrt(np.mean(R ** 2, axis=0))
for i_s, s in enumerate(steps):
    print(f'  s={s:3d}: residual RMS by row = {rowr[i_s]:.3f} decades')
worst = np.argsort(colr)[::-1][:8]
print('  worst lags (tau):', [(int(t), round(float(colr[t]), 3)) for t in worst])
print('  -> those lags correspond to t =', [(int(t + steps[i]), int(t)) for i, t in enumerate(worst)])
# residual at the echo lags vs elsewhere
echo_tau = {int(105 - s): i for i, s in enumerate(steps) if s < 105}
print('  echo tau=105-s rows exist for s:', list(echo_tau.keys()))

# ---------------- figure ----------------
style()
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

ax = axes[0, 0]
im = ax.imshow(A, aspect='auto', cmap='viridis', origin='upper',
               extent=[0, TAU, len(steps) - 0.5, -0.5])
ax.set_yticks(range(len(steps)))
ax.set_yticklabels([f's={s}' for s in steps])
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('injection position s')
ax.set_title(f'(a) log-response field log₁₀‖Δh‖ (r=0.1, layer 27): head size varies ×14 with s',
             fontweight='bold')
plt.colorbar(im, ax=ax, label='log₁₀ ‖Δh‖', shrink=0.9)

ax = axes[0, 1]
ax.bar(range(1, 9), S, color=['#d94801'] + ['#9ecae1'] * 7)
ax.set_xticks(range(1, 9))
ax.set_xlabel('singular value index')
ax.set_ylabel('singular value of centered log-field')
ax.set_title(f'(b) SVD of (a): 1 component = {100*frac[0]:.0f}% of variance — '
             f'field is mostly separable, but {100*(1-frac[0]):.0f}% is not',
             fontweight='bold')
ax.grid(alpha=0.3, lw=0.4, axis='y')

ax = axes[1, 0]
vmax = np.nanpercentile(np.abs(R), 98)
im = ax.imshow(R, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper',
               extent=[0, TAU, len(steps) - 0.5, -0.5])
ax.set_yticks(range(len(steps)))
ax.set_yticklabels([f's={s}' for s in steps])
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('injection position s')
ax.set_title(f'(c) non-separable residual: head-relaxation mismatch (τ<5, vertical)\n'
             f'+ echo stripes at fixed t (diagonal), RMS={np.sqrt(np.mean(R**2)):.2f} decades',
             fontweight='bold')
plt.colorbar(im, ax=ax, label='residual (decades)', shrink=0.9)

ax = axes[1, 1]
for i_s, s in enumerate(steps):
    ax.plot(norm[i_s], color=SCOL[int(s)], lw=2.0 if s == 16 else 1.0,
             alpha=1.0 if s == 16 else 0.8, label=f's={s}' if s in (16, 150, 32) else None)
ax.set_yscale('log')
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('fraction of immediate response remaining: ‖Δh(τ)‖/‖Δh(0)‖')
sp50 = norm[:, 50]
ax.set_title(f'(d) shapes do not collapse: at τ=50 retained fraction spans '
             f'{sp50.min():.3f}–{sp50.max():.3f} ({sp50.max()/sp50.min():.1f}×) '
             f'— not a function of lag alone',
             fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.25, lw=0.4)

fig.suptitle('Separability: response ≈ a(s) × b(τ) to first order, '
             'but the non-separable remainder is exactly the echo structure',
             fontsize=11, fontweight='bold')
fig.savefig(OUT + r'\fig2_separability.png')
plt.close(fig)
print('\nsaved fig2_separability.png')
