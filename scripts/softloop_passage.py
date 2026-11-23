"""Figure: gate-passage prediction of the separation horizon.

Model: the response accumulates the per-step readout gain,
dh(t) ~ dh0 * exp(sum_s<t log(1 + c sigma_s / T)); separation when the
cumulative log-gain exceeds log(sat/dh0). One constant c is calibrated at
T=0.05; every other temperature is a prediction.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'chaos_llm_2026/results/softloop/ds15v2'
OUT = 'chaos_llm_2026/results/figures/softloop'
L = -1
TS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]


def load(T):
    z = np.load(os.path.join(SRC, f'T{T}', 'softloop.npz'))
    return z


def measured(T):
    z = load(T)
    dh = z['dh'][:, L].astype(np.float64)
    h0 = float(np.median(z['h0n'][L]))
    ts = []
    for k in range(dh.shape[0]):
        sat = dh[k] > 0.25 * h0
        ts.append(int(np.argmax(sat)) if sat.any() else 600)
    return np.array(ts)


def cum_gain(T, c):
    z = load(T)
    s = z['sig_read'].astype(np.float64)
    gain = np.log1p(c * np.clip(s / T, 0, None))
    return np.cumsum(gain), z


def predict(T, c):
    cum, z = cum_gain(T, c)
    dh = z['dh'][:, L].astype(np.float64)
    h0 = float(np.median(z['h0n'][L]))
    dh0 = float(np.median(dh[:, 0]))
    target = np.log(0.25 * h0 / dh0)
    hit = np.where(cum > target)[0]
    return int(hit[0]) if len(hit) else 600


t005 = np.median(measured(0.05))
lo, hi = 1e-6, 10.0
for _ in range(60):
    mid = (lo + hi) / 2
    if predict(0.05, mid) < t005:
        hi = mid
    else:
        lo = mid
c = (lo + hi) / 2

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
pred = [predict(T, c) for T in TS]
meas = [float(np.median(measured(T))) for T in TS]
ax.plot(TS, pred, 'o-', color='#b8860b', label='gate-passage model '
        rf'(one constant $c={c:.3f}$ calibrated at $T=0.05$)')
ax.plot(TS, meas, 's-', color='#1a6faa', label='measured median separation step')
ax.axhline(600, color='gray', ls=':', lw=0.9, label='run length (censored)')
for T, p, m in zip(TS, pred, meas):
    ax.annotate(f'{p:.0f}/{m:.0f}', (T, max(p, m)), fontsize=7,
                xytext=(0, 6), textcoords='offset points', ha='center')
ax.set_xscale('log')
ax.set_xlabel('temperature $T$')
ax.set_ylabel('median separation step')
ax.set_title('(a) horizon predicted from the orbit\'s own margins and readout gains\n'
             'fails exactly where the straddle channel takes over ($T\\geq0.2$)', fontsize=9)
ax.legend(fontsize=7)

ax = axes[1]
# per-direction scatter at the two best temperatures + burst coincidence bars
coinc = []
for T in TS:
    z = load(T)
    m = z['m_ref'].astype(np.float64)
    dh = z['dh'][:, L].astype(np.float64)
    fl = z['flips']
    nb = on3 = 0
    for k in range(dh.shape[0]):
        ff = int(np.argmax(fl[k] > 0)) if (fl[k] > 0).any() else dh.shape[1]
        seg = dh[k, :min(ff, dh.shape[1])]
        if len(seg) < 20:
            continue
        ratio = seg[1:] / np.clip(seg[:-1], 1e-9, None)
        for t in np.where(ratio > 3.0)[0] + 1:
            nb += 1
            if m[t] < 3 * T:
                on3 += 1
    coinc.append((T, nb, on3))
Ts_ = [c2[0] for c2 in coinc]
fr = [100 * c2[2] / c2[1] if c2[1] else 0 for c2 in coinc]
ax.bar(np.arange(len(Ts_)), fr, color='#2e8b57', alpha=0.8)
ax.set_xticks(np.arange(len(Ts_)))
ax.set_xticklabels([f'T={t:g}' for t in Ts_], fontsize=8)
ax.set_ylabel(r'% of bursts ($\geq 3\times$) landing on live-gate steps ($m<3T$)')
ax.set_ylim(0, 105)
ax.set_title('(b) bursts happen at gates: coincidence of response bursts\n'
             'with the live-gate set, per temperature', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_passage.png'), dpi=150)
print('saved sl2_passage.png; c =', c)
for T, p, m in zip(TS, pred, meas):
    print(f'T={T:g}: predicted {p:4d}  measured {m:5.0f}')
