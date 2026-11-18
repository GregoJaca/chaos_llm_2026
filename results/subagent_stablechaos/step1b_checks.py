"""Quick checks: source of fr_cyber/childhood pred; post-flip resync structure."""
import numpy as np, sys
sys.path.insert(0, r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos")
from stablechaos_lib import *

for tag, fn, wfold in [("cyber", "fr_cyber.npz", r"window_cyber\w100\a1.npz"),
                       ("childhood", "fr_childhood.npz", None)]:
    d = load_free(fn)
    print(f"--- {tag}: pred rows for radii {d['radii']}")
    for k in range(len(d['radii'])):
        print('   pred[%d]=' % k, d['pred'][k])
    if wfold:
        w = load_a1(wfold)
        for k in range(len(d['radii'])):
            r = d['radii'][k]
            i = int(np.argmin(np.abs(w['radii'] - r)))
            same = np.array_equal(d['pred'][k], w['pred_T'][i]) if w['radii'][i] == r else False
            print(f"   r={r}: window_cyber radii {w['radii']}, closest i={i} (r={w['radii'][i]}), pred==pred_T? {same}")

# ---- post-flip structure (flux tube) ----
print("\n=== post-flip agreement, main prompt (large+mid) ===")
for fn in ["fr_w100_large.npz", "fr_w100_mid.npz", "fr_cyber.npz", "fr_childhood.npz"]:
    d = load_free(fn)
    mm = d['mismatch']  # [60,300]
    obs = d['obs']
    # agreement as function of tokens-since-flip, pooled over flipped runs
    ks = np.arange(0, 60)
    A = np.zeros_like(ks, float); N = np.zeros_like(ks, float)
    for k in range(len(d['radii'])):
        for j in range(20):
            T = obs[k, j]
            if T < 0: continue
            for kk in ks:
                pos = T + kk
                if pos < 300:
                    A[kk] += (not mm[k * 20 + j, pos]); N[kk] += 1
    Agood = np.where(N > 0, A / np.maximum(N, 1), np.nan)
    print(fn)
    print("  A(k) k=0..15:", np.round(Agood[:16], 3), " N(0)=", int(N[0]))
    print("  A(k) k=16..40:", np.round(Agood[16:41], 3))
    # deep baseline: agreement of flipped runs at positions far from flip
    deep = []
    for k in range(len(d['radii'])):
        for j in range(20):
            T = obs[k, j]
            if T < 0: continue
            row = mm[k * 20 + j]
            if T + 80 < 300:
                deep.append(1 - row[T + 80:].mean())
    print(f"  deep-baseline agreement (t>=T+80): mean {np.mean(deep) if deep else float('nan'):.3f}")
    # resync after first flip: fraction of flipped runs whose token T+1 matches ref
    res1 = []
    for k in range(len(d['radii'])):
        for j in range(20):
            T = obs[k, j]
            if T < 0 or T + 1 >= 300: continue
            res1.append(not mm[k * 20 + j, T + 1])
    print(f"  P(rejoin at T+1) = {np.mean(res1):.3f}  (n={len(res1)})")
