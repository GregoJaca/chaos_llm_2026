import os, glob, json, collections
import numpy as np

ROOT = os.path.expanduser('~/LLM/chaos_llm_2026')
OUT = os.path.join(ROOT, 'analysis_audit'); os.makedirs(OUT, exist_ok=True)

cands = sorted(glob.glob(ROOT + '/outputs_microsoft_0.00001/run_2000_*') +
               glob.glob(ROOT + '/outputs_microsoft_0.00001/run_4000_*') +
               glob.glob(ROOT + '/outputs_0.0001/run_2000_*'))
res = []
for d in cands:
    zf = os.path.join(d, 'tokens.npz')
    if not os.path.exists(zf): continue
    z = np.load(zf); b = z['baseline_ids']; P = int(z['prompt_len'])
    g = b[P:]
    if len(g) < 200: continue
    # longest run of identical token
    longest = 1; cur = 1; val = int(g[0]); best_val = val
    for i in range(1, len(g)):
        if g[i] == g[i-1]:
            cur += 1
            if cur > longest: longest, best_val = cur, int(g[i])
        else:
            cur = 1
    # shortest period p whose last 3p tokens are exactly periodic
    per = None
    for p in range(1, min(201, len(g)//3 + 1)):
        tail = g[-3*p:]
        if np.array_equal(tail, np.tile(g[-p:], 3)):
            per = p; break
    # repeated n-grams
    def repeated(k):
        c = collections.Counter(tuple(g[i:i+k]) for i in range(len(g)-k+1))
        return int(sum(1 for v in c.values() if v > 1)), int(max(c.values()) if c else 0)
    r5, m5 = repeated(5); r10, m10 = repeated(10)
    rec = dict(dir=os.path.relpath(d, ROOT), n_gen=int(len(g)), longest_identical_run=longest,
               longest_token=best_val, exact_period=per, repeated_5grams=r5, max_5gram_count=m5,
               repeated_10grams=r10, max_10gram_count=m10)
    res.append(rec); print(json.dumps(rec), flush=True)

with open(os.path.join(OUT, 'periodicity_audit.json'), 'w') as f:
    json.dump(res, f, indent=2)
print('saved', os.path.join(OUT, 'periodicity_audit.json'))
