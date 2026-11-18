"""Task 3: where does the long memory live?

(a) angle structure: 1 - cos between perturbed and baseline hidden states vs
    lag (the relative response squared / 2), with the float32 precision floor;
    plus the cos-based estimate of the baseline state norms ||h|| per layer.
(b) coupling of hidden response to margin response: Spearman corr between
    dh and |dm| as a function of lag bin (pooled over s), using the correct
    one-position offset (dm[t] is the gap change for token t, read from the
    logits at position t-1, so it pairs with dh at position t-1).
(c) is the tail carried by the same directions as the head? per s, correlation
    across the 24 injected directions between immediate dh(tau=0) and the
    tail dh (mean over tau in [80, 99]).
(d) hidden-response bumps are NOT margin fragility: dh(t) vs baseline margin
    margins(t) along t for s=16 — the flip position t=123 has a *small*
    baseline margin but no dh bump; the dh bump at t=113 has a large margin.
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from common import load_emb, style, OUT, SCOL, LAYCOL

D = load_emb()
steps, layers = D['steps'], D['layers']
dh, dm, margins = D['dh'], D['dm'], D['margins']
cosv = D['cos']
FLOOR = 1.2e-7   # float32 resolution of cos near 1

print('=== (a) 1 - cos at tau=0 and usable range (above 1e-6), L27, r=0.1 ===')
for i_s, s in enumerate(steps):
    c = 1 - cosv[3, i_s, :, 3, :]
    med = np.nanmedian(c, axis=0)          # over dirs, per t
    tau = np.arange(300) - s
    ok = (med > 1e-6) & (tau >= 0)
    last = np.max(np.where(ok)[0]) if ok.any() else -1
    print(f'  s={s:3d}: 1-cos(tau=0)={med[s]:.2e}   measurable up to tau={last - s}')

print('\n=== cos-based baseline state norms ||h_base(s, L)|| (median over dirs, tau=0) ===')
for L in [7, 14, 21, 27]:
    i_l = list(layers).index(L)
    vals = []
    for i_s, s in enumerate(steps):
        c = 1 - cosv[3, i_s, :, i_l, s]
        h = dh[3, i_s, :, i_l, s] / np.sqrt(2 * c)
        vals.append(np.median(h))
    print(f'  L{L}: ' + '  '.join(f'{v:6.0f}' for v in vals))

print('\n=== (b) coupling dh <-> |dm| by lag bin (pooled over all 8 s, L27, r=0.1) ===')
bins = [(1, 5), (5, 15), (15, 40), (40, 100), (100, 10**9)]
def pooled(offset):
    X, Y, TAU = [], [], []
    for i_s, s in enumerate(steps):
        for t in range(int(s) + 1, 300):
            X.append(dh[3, i_s, :, 3, t - offset])
            Y.append(np.abs(dm[3, i_s, :, t]))
            TAU.append(np.full(24, t - s))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(TAU)

for offset, lbl in [(0, 'dh[t] (same position)'), (1, 'dh[t-1] (logits position)')]:
    X, Y, TAU = pooled(offset)
    print(f'-- pairing |dm[t]| with {lbl} --')
    for lo, hi in bins:
        m = (TAU >= lo) & (TAU < hi) & (X > 0) & (Y > 0)
        r = spearmanr(X[m], Y[m]).statistic
        rp = pearsonr(np.log10(X[m]), np.log10(Y[m])).statistic
        print(f'   tau {lo:3d}-{hi if hi < 10**6 else 299:3d}: n={m.sum():5d}  '
              f'Spearman={r:+.3f}  log-log Pearson={rp:+.3f}')
X, Y, TAU = pooled(1)   # use the logits-position pairing for the figure

print('\n=== (c) direction persistence: corr across 24 dirs, head dh(tau=0) vs tail dh ===')
for i_s, s in enumerate(steps):
    head = dh[3, i_s, :, 3, int(s)]
    tail = np.nanmean(dh[3, i_s, :, 3, int(s) + 80:int(s) + 100], axis=1)
    r = spearmanr(head, tail).statistic
    # also: corr of head with |dm| at the fragile positions 123/143
    dmt = np.abs(dm[3, i_s, :, 123]) if s < 123 else np.full(24, np.nan)
    r2 = spearmanr(head, dmt).statistic if s < 123 else np.nan
    print(f'  s={s:3d}: Spearman(head dh, tail dh) = {r:+.3f}'
          + (f'   Spearman(head dh, |dm(123)|) = {r2:+.3f}' if s < 123 else ''))

print('\n=== (d) dh(t) vs margins(t): fragility is not where the response is big ===')
for i_s, s in enumerate([16, 32]):
    med = np.nanmedian(dh[3, i_s, :, 3, int(s):], axis=0)
    t = np.arange(int(s), 300)
    r = spearmanr(med, margins[t]).statistic
    print(f'  s={s}: Spearman(dh(t), margins(t)) over t>=s: {r:+.3f}')
print('  s=16: dh at t=113 (echo) =', np.nanmedian(dh[3, 1, :, 3, 113]),
      ' margin=', margins[113])
print('  s=16: dh at t=123 (flip) =', np.nanmedian(dh[3, 1, :, 3, 123]),
      ' margin=', margins[123], ' <- smallest margin of whole sequence')

# ---------------- figure ----------------
style()
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

ax = axes[0, 0]
for i_s, s in enumerate(steps):
    med = np.nanmedian(1 - cosv[3, i_s, :, 3, int(s):], axis=0)
    tau = np.arange(len(med))
    ok = med > 5 * FLOOR
    ax.plot(tau[ok], med[ok], color=SCOL[int(s)],
            lw=2.2 if s == 16 else 1.0, alpha=1.0 if s == 16 else 0.8)
ax.axhline(FLOOR, color='k', ls=':', lw=1)
ax.text(150, 2 * FLOOR, 'float32 floor of cos', fontsize=7)
ax.set_yscale('log')
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('1 − cos(h_pert, h_base)   [≈ (Δh/‖h‖)²/2]')
ax.set_title('(a) response is a small sideways displacement of the state (cos > 0.9997);\n'
             'relative size 1−cos shrinks with lag — float32 floor limits the range',
             fontweight='bold')
ax.grid(alpha=0.25, lw=0.4)

ax = axes[0, 1]
labels, sp_vals, n_pts = [], [], []
for lo, hi in bins:
    m = (TAU >= lo) & (TAU < hi)
    labels.append(f'{lo}–{hi if hi < 10**6 else 299}')
    sp_vals.append(spearmanr(X[m], Y[m]).statistic)
    n_pts.append(int(m.sum()))
ax.bar(range(len(sp_vals)), sp_vals, color='#4292c4')
for i, (v, n) in enumerate(zip(sp_vals, n_pts)):
    ax.text(i, v + 0.01, f'{v:+.2f}\nn={n}', ha='center', fontsize=7)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel('lag bin τ = t − s (tokens)')
ax.set_ylabel("Spearman corr( ‖Δh(t−1)‖ , |Δ margin(t)| )")
ax.set_title('(b) the margin response tracks the hidden response at every lag\n'
             '(Spearman +0.48 near the head, rising to +0.65 in the tail)',
             fontweight='bold')
ax.grid(alpha=0.3, lw=0.4, axis='y')
ax.set_ylim(0, 1.0)

ax = axes[1, 0]
pers = []
for i_s, s in enumerate(steps):
    head = dh[3, i_s, :, 3, int(s)]
    tail = np.nanmean(dh[3, i_s, :, 3, int(s) + 80:int(s) + 100], axis=1)
    pers.append(spearmanr(head, tail).statistic)
ax.bar([str(int(s)) for s in steps], pers, color=[SCOL[int(s)] for s in steps])
for i, v in enumerate(pers):
    ax.text(i, v + 0.01, f'{v:+.2f}', ha='center', fontsize=7)
ax.set_xlabel('injection position s')
ax.set_ylabel('Spearman corr over 24 dirs:\nhead ‖Δh(τ=0)‖ vs tail ‖Δh(τ=80–99)‖')
ax.set_title('(c) the same injected directions carry head and tail: corr +0.4…+0.9\n'
             'for 7/8 s (exception s=150, whose head itself oscillates)',
             fontweight='bold')
ax.grid(alpha=0.3, lw=0.4, axis='y')

ax = axes[1, 1]
med = np.nanmedian(dh[3, 1, :, 3, 16:], axis=0)
t = np.arange(16, 300)
ax.plot(t, med, color='#d62728', lw=1.2, label='‖Δh(t)‖ (s=16, log left)')
ax.set_yscale('log')
ax.set_xlabel('token position t (s=16 injected)')
ax.set_ylabel('‖Δh‖ (red, log)')
ax2 = ax.twinx()
ax2.plot(t, margins[t], color='#08519c', lw=1.0, alpha=0.8, label='baseline margin (linear right)')
ax2.set_ylabel('baseline top1−top2 margin (blue, linear)')
ax2.set_yscale('log')
ax.axvline(123, color='k', ls='--', lw=1)
ax.annotate('flip t=123\nmargin=0.02', xy=(123, med[123 - 16]), xytext=(140, med.max()),
            arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=7)
ax.axvline(113, color='0.5', ls=':', lw=1)
ax.annotate('echo t=113\nmargin=2.6', xy=(113, med[113 - 16]), xytext=(30, med[113 - 16] * 0.4),
            arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=7)
ax.set_title('(d) big hidden responses (echo t=113) ≠ fragile decisions (t=123, margin 0.02):\n'
             'dh mildly anticorrelates with margin (Spearman −0.48) but flips need BOTH',
             fontweight='bold')

fig.suptitle('Where the memory lives: same directions, sideways to the state, '
             'coupled to logits at every lag — but token flips need small margins',
             fontsize=11, fontweight='bold')
fig.savefig(OUT + r'\fig3_memory.png')
plt.close(fig)
print('\nsaved fig3_memory.png')
