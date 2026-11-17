import os, glob, json, collections
import numpy as np

ROOT = os.path.expanduser('~/LLM/chaos_llm_2026')
OUT = os.path.join(ROOT, 'analysis_audit'); os.makedirs(OUT, exist_ok=True)

def first_div(b, row, P):
    n = min(len(b), len(row))
    mm = np.where(row[P:n] != b[P:n])[0]
    return int(mm[0]) if len(mm) else -1

def load_family(w):
    fam = {}
    for d in glob.glob(os.path.join(ROOT, 'outputs_OLD', f'run_{w}_*_interstellar_travel_seed_69')):
        c = json.load(open(os.path.join(d, 'config.json')))
        if c['perturbation']['num_conditions'] != 100 or c['perturbation']['subspace_seed'] != 69:
            continue
        r = float(c['runtime']['perturbation_magnitude'])
        if r in fam:
            continue
        z = np.load(os.path.join(d, 'tokens.npz'))
        b = z['baseline_ids']; P = int(z['prompt_len']); rows = z['perturbed_ids']; L = z['perturbed_lengths']
        T = [first_div(b, rows[i, :int(L[i])], P) for i in range(rows.shape[0])]
        fam[r] = np.array(T)
    return fam

res = {}
for w in [50, 100, 200, 500]:
    fam = load_family(w)
    if len(fam) < 2:
        print('not enough radii for window', w, sorted(fam)); continue
    rs = sorted(fam)
    M = np.stack([fam[r] for r in rs], axis=1)  # [n_dir, n_r]
    n = M.shape[0]
    never = np.all(M == -1, axis=1)
    flips = ~never
    const = flips & np.all(M == M[:, [0]], axis=1)
    var = flips & ~const
    # modal T among flipping directions (use first radius)
    Ts = M[flips, 0]
    modal = collections.Counter(Ts.tolist()).most_common(5)
    rec = dict(window=w, radii=rs, n_dirs=int(n), n_never=int(never.sum()),
               n_constant=int(const.sum()), n_variable=int(var.sum()),
               frac_never=round(float(never.mean()), 3),
               frac_constant=round(float(const.sum()) / max(int(flips.sum()), 1), 3),
               modal_T=[(int(t), int(c)) for t, c in modal])
    res[w] = rec
    print(json.dumps(rec), flush=True)

with open(os.path.join(OUT, 'fixeddir_audit.json'), 'w') as f:
    json.dump(res, f, indent=2)
print('saved', os.path.join(OUT, 'fixeddir_audit.json'))
