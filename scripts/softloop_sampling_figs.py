"""Sampling statistics: soft vs sampled-CRN vs sampled-independent."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'chaos_llm_2026/results/softloop_sample'
OUT = 'chaos_llm_2026/results/figures/softloop'
L = -1


def curve(d):
    f = os.path.join(d, 'softloop.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    dh = z['dh'][:, L].astype(np.float64)
    h0 = float(np.median(z['h0n'][L]))
    S = dh.shape[1]
    ts = [int(np.argmax(dh[k] > 0.25 * h0)) if (dh[k] > 0.25 * h0).any() else S
          for k in range(dh.shape[0])]
    return dict(med=np.median(dh, axis=0), lo=np.percentile(dh, 25, axis=0),
                hi=np.percentile(dh, 75, axis=0), ts=np.array(ts), S=S,
                K=dh.shape[0], m_med=float(np.median(z['m_ref'])))


def rate(dh_t, h0):
    y = dh_t.astype(np.float64)
    sat = y > 0.25 * h0
    ts = int(np.argmax(sat)) if sat.any() else len(y)
    if ts < 13:
        return np.nan, ts
    seg = np.log(np.clip(y[3:ts], 1e-12, None))
    if np.ptp(seg) < 1e-9:
        return 0.0, ts
    return float(np.polyfit(np.arange(3, ts), seg, 1)[0]), ts


rows = {}
for d in sorted(glob.glob(os.path.join(SRC, '*', '*'))):
    if not os.path.isdir(d):
        continue
    c = curve(d)
    if c is None:
        continue
    rows[d.replace(SRC + os.sep, '').replace(os.sep, '/')] = c

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax = axes[0]
COLOR = {'soft': '#1a6faa', 'crn': '#2e8b57', 'ind': '#c8102e'}
for key in sorted(rows):
    model, tag = key.split('/')
    mode = 'soft' if tag.startswith('soft') else ('crn' if 'crn' in tag else 'ind')
    c = rows[key]
    ax.plot(np.arange(c['S']), np.clip(c['med'], 1e-5, None), lw=1.5,
            color=COLOR[mode], alpha=0.9,
            label=f'{model} {tag}' if model == 'ds15' else None)
    ax.fill_between(np.arange(c['S']), np.clip(c['lo'], 1e-5, None),
                    np.clip(c['hi'], 1e-5, None), alpha=0.15, color=COLOR[mode])
ax.set_yscale('log')
ax.set_xlabel('step $t$')
ax.set_ylabel(r'median $\|\delta h_{t,L}\|$ (band = IQR over directions)')
ax.set_title('(a) response under the three feedback rules\n'
             'soft = expected embedding; CRN = sampled tokens shared across rows;\n'
             'ind = each row samples its own token', fontsize=9)
ax.legend(fontsize=6, ncol=1)

ax = axes[1]
labels, meds, cens = [], [], []
for key in sorted(rows):
    model, tag = key.split('/')
    c = rows[key]
    labels.append(f'{model}\n{tag}')
    meds.append(float(np.median(c['ts'])))
    cens.append(int((c['ts'] == c['S']).sum()))
ax.bar(np.arange(len(labels)), meds,
       color=[COLOR['soft'] if 'soft' in l else (COLOR['crn'] if 'crn' in l else COLOR['ind'])
              for l in labels])
ax.axhline(600, color='gray', ls=':')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6)
ax.set_ylabel('median separation step (600 = censored)')
ax.set_title('(b) horizons per run', fontsize=9)

ax = axes[2]
for key in sorted(rows):
    if not key.startswith('ds15/crn_T1.0'):
        continue
    r = key.split('r')[-1]
    c = rows[key]
    ax.plot(np.arange(c['S']), np.clip(c['med'], 1e-5, None), lw=1.6,
            label=f'r = {r}')
ax.set_yscale('log')
ax.set_xlabel('step $t$')
ax.set_ylabel(r'median $\|\delta h_{t,L}\|$')
ax.set_title('(c) CRN radius grid at $T=1.0$: the sampled-path\n'
             'conditional response at every radius', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_sampling.png'), dpi=150)
plt.close(fig)

table = {}
for key, c in rows.items():
    table[key] = dict(median_tsat=float(np.median(c['ts'])),
                      censored=int((c['ts'] == c['S']).sum()),
                      K=c['K'], m_med=c['m_med'])
with open(os.path.join(OUT, 'sl2_sampling.json'), 'w') as f:
    json.dump(table, f, indent=1)
for key in sorted(table):
    print(f"{key:28s} t_sat med {table[key]['median_tsat']:5.0f}  "
          f"censored {table[key]['censored']:3d}/{table[key]['K']}")
print('saved sl2_sampling.png')
