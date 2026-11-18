import numpy as np, os
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"

# 1. prompts across datasets
d0 = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
for w in [100, 150, 200, 300, 1000]:
    dz = np.load(os.path.join(BASE, "window_fixed", f"w{w}", "a1.npz"))
    print(f"window_fixed w{w}: prompt == w100_layers prompt? {np.array_equal(dz['prompt_ids'], d0['prompt_ids'])}")
d200 = np.load(os.path.join(BASE, "w200", "a1.npz")); d500 = np.load(os.path.join(BASE, "w500", "a1.npz"))
print("w200 prompt equal:", np.array_equal(d200['prompt_ids'], d0['prompt_ids']),
      " gen equal (300):", np.mean(d200['gen_ids'][:300] == d0['gen_ids'][:300]))
print("w500 prompt equal:", np.array_equal(d500['prompt_ids'], d0['prompt_ids']),
      " gen equal (300):", np.mean(d500['gen_ids'][:300] == d0['gen_ids'][:300]))

# 2. layer-3 zeros location
dh = d0["dh_norm"]; radii = d0["radii"]
m3 = dh[:, :, 3, :] <= 0
print("layer3 zero pattern: per-t counts (first 10):", m3[0].sum(axis=0)[:10], " total per dir:", m3[0].sum(axis=1)[:8])
tt = np.where(m3.any(axis=(0, 1)))[0]
print("layer3 zero t positions:", tt[:20], "... n_unique_t:", len(np.unique(tt)))

# 3. fr chunk mapping: does first mismatch of gen rows match obs?
z = np.load(os.path.join(BASE, "fr_w100_mid.npz"), allow_pickle=True)
obs = z["obs"]; radii_fr = z["radii"]; gens = z["gen_ids"]; refs = z["ref_ids"]
print("\nfr_w100_mid radii:", radii_fr, " chunks:", len(gens))
ok, bad = 0, 0
for r_idx, r in enumerate(radii_fr):
    for c in range(4):
        k = r_idx * 4 + c
        if k >= len(gens): break
        g = np.asarray(gens[k]); ref = np.asarray(refs[k])
        for j in range(1, 7):
            d = 6 * c + (j - 1)
            if d >= 20: break
            mism = np.where(g[j] != ref)[0]
            first = int(mism[0]) if len(mism) else -1
            o = int(obs[r_idx, d])
            if (first == -1 and o < 0) or (first >= 0 and o >= 0 and first == o):
                ok += 1
            else:
                bad += 1
                if bad <= 6: print(f"  MISMATCH r={r} c={c} j={j} d={d}: first={first} obs={o}")
print(f"mapping check: ok={ok} bad={bad}")
# how many rows per chunk beyond dirs? check duplicates in last chunk
g3 = np.asarray(gens[3]); print("chunk3 rows equal each other?", [np.array_equal(g3[1], g3[j]) for j in range(2, 7)])
