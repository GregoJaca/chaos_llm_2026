"""Tasks 5 + 7: which perturbation directions eventually change a token, and
what the autoregressive runs show.

Flip mechanism (teacher-forced): the baseline top1-top2 gap margins[t] flips
sign iff dm[t] < -margins[t]. Verified: this predicts the freerun token
changes exactly (5/5, no false positives among 48 runs).

(a) s=16: dm at t=123 for all 24 directions vs the -margin threshold.
(b) s=50: dm at t=143 vs threshold (1 flip).
(c) does the IMMEDIATE response (tau=0) predict a flip? AUC with bootstrap CI
    (n is tiny: 4 vs 20, 1 vs 23) — reported honestly.
(d) autoregressive latency/survival: first-change positions and how much of
    the 300-token continuation survives.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import load_emb, load_freerun, style, OUT, SCOL, flips
from scipy.stats import spearmanr, mannwhitneyu

D = load_emb()
steps, layers, margins = D['steps'], D['layers'], D['margins']
dh, dm = D['dh'], D['dm']
FR = load_freerun()
pred, ref, fsteps = FR['pred'], FR['ref'], FR['fsteps']
FL = flips(D, ir=3)          # teacher-forced predicted flips
ACT = {}
for i_s in range(2):
    s = int(fsteps[i_s])
    ACT[s] = [(idir, int(np.argmax((pred[i_s, idir] != ref) & (pred[i_s, idir] != -1))))
              for idir in range(24)
              if ((pred[i_s, idir] != ref) & (pred[i_s, idir] != -1)).any()]
print('teacher-forced flip prediction:', FL[16], FL[50])
print('freerun actual flips          :', ACT[16], ACT[50])
print('exact match (dirs and steps):',
      sorted(FL[16]) == sorted(ACT[16]) and sorted(FL[50]) == sorted(ACT[50]))

print('\n=== (c) does the immediate response predict a flip? ===')
rng = np.random.default_rng(0)
def auc_ci(pos, neg, nboot=2000):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pool = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    aucs = []
    for _ in range(nboot):
        # stratified bootstrap: resample within each group (groups never empty)
        pp = rng.choice(pos, len(pos), replace=True)
        nn = rng.choice(neg, len(neg), replace=True)
        aucs.append((pp[:, None] > nn[None, :]).mean() +
                    0.5 * (pp[:, None] == nn[None, :]).mean())
    a = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return a, np.percentile(aucs, [2.5, 97.5])

for s, t in [(16, 123), (50, 143)]:
    flipdirs = [d for d, _ in FL[s]]
    head = dh[3, int(np.where(steps == s)[0][0]), :, 3, s]
    pos, neg = head[flipdirs], np.delete(head, flipdirs)
    a, ci = auc_ci(pos, neg)
    ranks = [int((head > head[d]).sum()) + 1 for d in flipdirs]
    print(f'  s={s}: flip dirs head-dh ranks among 24: {ranks}')
    print(f'         AUC(head dh predicts flip) = {a:.2f}  bootstrap95% CI [{ci[0]:.2f}, {ci[1]:.2f}]')
    r = spearmanr(head, dm[3, int(np.where(steps == s)[0][0]), :, t]).statistic
    print(f'         Spearman(head dh, dm(t={t})) = {r:+.3f}  <- flip driver is the readout, not the size')
# pooled
flipdirs = {16: [d for d, _ in FL[16]], 50: [d for d, _ in FL[50]]}
allpos, allneg = [], []
for s in [16, 50]:
    head = dh[3, int(np.where(steps == s)[0][0]), :, 3, s]
    fl = flipdirs[s]
    allpos += list(head[fl])
    allneg += list(np.delete(head, fl))
a, ci = auc_ci(np.array(allpos), np.array(allneg))
print(f'  pooled (5 flips vs 43 non-flips): AUC = {a:.2f}  CI [{ci[0]:.2f}, {ci[1]:.2f}]')

print('\n=== margin of safety at the threshold ===')
short = {}
for s, t in [(16, 123), (50, 143)]:
    dmm = dm[3, int(np.where(steps == s)[0][0]), :, t]
    thr = -margins[t]
    fl = [d for d, _ in FL[s]]
    closest_nonflip = min(x for x in dmm if thr < x < 0)
    short[s] = 100 * (1 - abs(closest_nonflip) / abs(thr))
    print(f'  s={s}, t={t}: threshold={thr:.4f}; deepest flip={dmm[fl].min():.4f}; '
          f'closest non-flip={closest_nonflip:.4f} (stops {short[s]:.0f}% short of the threshold)')

print('\n=== (d) autoregressive survival and latency ===')
for i_s in range(2):
    s = int(fsteps[i_s])
    surv = np.ones(300)
    firsts = []
    for idir in range(24):
        p = pred[i_s, idir]
        diff = (p != ref) & (p != -1)
        if diff.any():
            t0 = int(np.argmax(diff))
            firsts.append(t0)
            surv[t0:] -= 1 / 24
            agree_after = float((p[t0 + 1:] == ref[t0 + 1:]).mean())
            print(f'  s={s} dir={idir}: first change t={t0} (latency {t0 - s} after injection); '
                  f'agreement with ref after flip = {agree_after:.2f}')
    print(f'  s={s}: survival at t=299: {surv[-1]*24:.0f}/24 runs still identical to reference')
print('total: 43/48 autoregressive runs never change a token; '
      '5/5 changes are at t=123 (s=16) or t=143 (s=50) — the two smallest margins in the whole 300')

# ---------------- figure ----------------
style()
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

def mech_panel(ax, s, t):
    i_s = int(np.where(steps == s)[0][0])
    dmm = np.sort(dm[3, i_s, :, t])
    fl = [d for d, _ in FL[s]]
    colors = ['#d62728' if (d in fl) else '#9ecae1' for d in np.argsort(dm[3, i_s, :, t])]
    ax.bar(range(24), dmm, color=colors)
    ax.axhline(-margins[t], color='k', ls='--', lw=1.2)
    ax.text(0.4, -margins[t] + 0.001, f'−margin = −{margins[t]:.4f}  (flip iff below)', fontsize=7.5)
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlabel(f'24 perturbation directions (sorted by dm at t={t})')
    ax.set_ylabel(f'Δ(top1−top2 gap) at token t={t}')
    ax.set_title(f'({"a" if s == 16 else "b"}) s={s}: {len(fl)}/24 directions cross the threshold '
                 f'at t={t}\nred = flips in the autoregressive run; closest non-flip stops '
                 f'{short[s]:.0f}% short', fontweight='bold')
    ax.grid(alpha=0.3, lw=0.4, axis='y')

mech_panel(axes[0, 0], 16, 123)
mech_panel(axes[0, 1], 50, 143)

ax = axes[1, 0]
labels, aucs, cis = [], [], []
for s in [16, 50]:
    fl = [d for d, _ in FL[s]]
    head = dh[3, int(np.where(steps == s)[0][0]), :, 3, s]
    a, ci = auc_ci(head[fl], np.delete(head, fl))
    labels.append(f's={s}\n({len(fl)} vs {24-len(fl)})')
    aucs.append(a); cis.append(ci)
a, ci = auc_ci(np.array(allpos), np.array(allneg))
labels.append(f'pooled\n(5 vs 43)')
aucs.append(a); cis.append(ci)
xpos = np.arange(len(labels))
ax.bar(xpos, aucs, color=['#d62728', '#ff9896', '#9e9e9e'], width=0.55)
for i, (v, c) in enumerate(zip(aucs, cis)):
    ax.errorbar(i, v, yerr=[[v - c[0]], [c[1] - v]], fmt='k_', capsize=4, lw=1)
    ax.text(i, 0.06, f'{v:.2f}\nCI [{c[0]:.2f},{c[1]:.2f}]', ha='center', fontsize=7.5)
ax.axhline(0.5, color='k', ls=':', lw=1)
ax.text(2.4, 0.52, 'chance', fontsize=7.5)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel('AUC: does the immediate response (τ=0) predict a flip?')
ax.set_ylim(0, 1)
ax.set_title('(c) immediate-response magnitude barely predicts flips '
             '(AUC 0.3–0.7, CI touches chance)\nthe margin response dm at the fragile '
             'position is the real predictor (=1.00 by construction)', fontweight='bold')
ax.grid(alpha=0.3, lw=0.4, axis='y')

ax = axes[1, 1]
for i_s in range(2):
    s = int(fsteps[i_s])
    surv = np.ones(300)
    for idir in range(24):
        p = pred[i_s, idir]
        diff = (p != ref) & (p != -1)
        if diff.any():
            surv[int(np.argmax(diff)):] -= 1 / 24
    ax.step(range(300), surv, where='post', color=SCOL[s], lw=1.8,
            label=f's={s}: {surv[-1]*24:.0f}/24 unchanged')
    # the injected position
    ax.axvline(s, color=SCOL[s], ls=':', lw=1)
for t, m in [(123, margins[123]), (143, margins[143])]:
    ax.axvline(t, color='0.4', ls='--', lw=0.8)
    ax.text(t + 1, 1.03 - 0.05 * (t == 123), f't={t}\nmargin={m:.3f}', fontsize=7)
ax.set_xlabel('token position t')
ax.set_ylabel('fraction of the 24 autoregressive runs still identical to the reference')
ax.set_ylim(0.75, 1.03)
ax.legend(loc='center right')
ax.set_title('(d) 43/48 runs never diverge; the 5 that do flip exactly at the two\n'
             'smallest-margin positions, 93–107 tokens after the injection',
             fontweight='bold')
ax.grid(alpha=0.25, lw=0.4)

fig.suptitle('Token changes are threshold crossings of the margin response at fragile positions — '
             'not predicted by response size', fontsize=11, fontweight='bold')
fig.savefig(OUT + r'\fig5_directions_latency.png')
plt.close(fig)
print('\nsaved fig5_directions_latency.png')
