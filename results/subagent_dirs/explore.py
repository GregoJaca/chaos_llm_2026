"""Stage A: verify semantics of every array we plan to use.
Read-only on local_data; prints facts."""
import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"

z = np.load(BASE + r"\w100_layers\a1.npz")
dh = z["dh_norm"].astype(np.float64)      # [R=12, D=20, L=8, T=300]
dm = z["dm"].astype(np.float64)           # [R, D, T]
cosz = z["cos"].astype(np.float64)        # [R, D, L, T]
margins = z["margins"].astype(np.float64) # [T]
r_c = z["r_c"].astype(np.float64)         # [R, D, T]
radii = z["radii"]; layers = z["layers"]
simplex = z["simplex"].astype(np.float64)
subspace = z["subspace"]
dh_last = z["dh_last"].astype(np.float64)  # [R, 3, T, 1536]
pred_T = z["pred_T"]; norm_h0 = z["norm_h0"].astype(np.float64)
gen_ids = z["gen_ids"]; top1 = z["top1"]; top2 = z["top2"]
prompt_ids = z["prompt_ids"]

print("radii:", radii)
print("layers:", layers)

# --- 1. dh scaling with r at layer 0 (embedding layer => dh ~ r ?) ---
print("\n--- dh vs r, layer 0, median over t, dir 0..2 ---")
for l in [0, 3, 7]:
    print(f"layer {layers[l]}:")
    for d in range(3):
        med = [np.median(dh[r, d, l]) for r in range(12)]
        print(f"  dir{d}: " + " ".join(f"{m:.3g}" for m in med))

# ratio dh/r for small r
print("\ndh/r at r=1e-3 (layer 27, median over t):", np.median(dh[0, :, 7] / radii[0]))
print("dh/r at r=3e-3 (layer 27):", np.median(dh[1, :, 7] / radii[1]))

# --- 2. dm scaling with r ---
print("\n--- dm/r for small r (dir 0, first 5 tokens) ---")
print("dm[0,0,:5]/r:", dm[0, 0, :5] / radii[0])
print("dm[1,0,:5]/r:", dm[1, 0, :5] / radii[1])
print("dm[2,0,:5]/r:", dm[2, 0, :5] / radii[2])
print("median |dm|/r over all d,t at each r:", [f"{np.median(np.abs(dm[r]))/radii[r]:.4g}" for r in range(12)])

# --- 3. r_c semantics: r-independent? compare with margins*r/(-dm) ---
rc_manual = margins[None, None, :] * radii[:, None, None] / (-dm)
finite = np.isfinite(r_c) & (r_c < 1e20)
print("\n--- r_c: fraction finite per radius ---", [f"{finite[r].mean():.2f}" for r in range(12)])
# compare r_c[r,d,t] to rc_manual for finite entries at r=4 (0.1)
sel = finite[4] & (np.abs(rc_manual[4]) < 1e20) & (dm[4] < 0)
print("r_c vs margins*r/(-dm) agreement (r=0.1, dm<0): corr=",
      np.corrcoef(r_c[4][sel], rc_manual[4][sel])[0, 1])
print("ratio r_c/manual median:", np.median(r_c[4][sel] / rc_manual[4][sel]))
# is r_c constant across r for same (d,t)?
print("r_c relative std across r (median over finite entries):")
ok = finite.all(axis=0)
relstd = np.std(r_c[:, ok], axis=0) / np.mean(r_c[:, ok], axis=0)
print("  median relstd:", np.median(relstd), " n(t,d) all-finite:", ok.sum(), "of", ok.size)

# --- 4. dh_last: which 3 layers? match norms to dh_norm ---
print("\n--- dh_last: match to dh_norm columns ---")
for i in range(3):
    n_last = np.linalg.norm(dh_last[4, i], axis=-1)  # r=0.2, [T,1536]
    for l in range(8):
        cc = np.corrcoef(n_last, dh[4, :, l].mean(axis=0))[0, 1]
        if cc > 0.9:
            print(f"  dh_last[:,{i}] best match layer {layers[l]} (corr {cc:.3f})")

# --- 5. simplex structure ---
print("\n--- simplex rows ---")
rownorm = np.linalg.norm(simplex, axis=1)
print("row norms:", np.array2string(rownorm, precision=3))
G = simplex @ simplex.T
off = G - np.diag(np.diag(G))
print("max |offdiag| of Gram:", np.abs(off).max(), " diag:", np.array2string(np.diag(G), precision=3))
l1l2 = np.abs(simplex).sum(1) / rownorm
print("L1/L2 per dir:", np.array2string(l1l2, precision=2))
print("argmax coord per dir:", simplex.argmax(1))
print("subspace token ids:", subspace)
print("max |coeff| per dir:", np.array2string(np.abs(simplex).max(1), precision=3))

# --- 6. prompt / generation identity across files ---
z2 = np.load(BASE + r"\a1.npz")
z3 = np.load(BASE + r"\w200\a1.npz")
z4 = np.load(BASE + r"\w500\a1.npz")
print("\n--- file identity checks ---")
print("prompt_ids same as root a1:", np.array_equal(prompt_ids, z2["prompt_ids"]))
print("prompt_ids same as w200:", np.array_equal(prompt_ids, z3["prompt_ids"]))
print("gen_ids same as root a1:", np.array_equal(gen_ids, z2["gen_ids"]))
print("top1 same as root a1:", np.array_equal(top1, z2["top1"]))
print("simplex same as root a1:", np.array_equal(z["simplex"], z2["simplex"]))
print("simplex same as w200:", np.array_equal(z["simplex"], z3["simplex"]))
print("subspace same as w200:", np.array_equal(z["subspace"], z3["subspace"]))
print("margins w100 mean:", margins.mean(), " root a1 margins mean:", float(z2["margins"].mean()))
print("w200 margins mean:", float(z3["margins"].mean()), " w500:", float(z4["margins"].mean()))
print("w200 radii:", z3["radii"], "layers:", z3["layers"])
print("w500 radii:", z4["radii"], "layers:", z4["layers"])

# --- 7. free-run flips ---
fr = np.load(BASE + r"\fr_w100_large.npz", allow_pickle=True)
obs = fr["obs"]; pred = fr["pred"]
print("\n--- fr_w100_large: obs / pred flip steps ---")
print("radii:", fr["radii"])
print("obs:\n", obs)
print("pred:\n", pred)
print("obs==-1 (no flip) counts per r:", (obs == -1).sum(axis=1))
print("pred==-1 counts per r:", (pred == -1).sum(axis=1))

# --- 8. state data ---
st = np.load(BASE + r"\state.npz")
dh_s = st["dh"]; dm_s = st["dm"]; steps = st["steps"]
print("\n--- state.npz ---")
print("dh median over t per (s):", [f"s={steps[i]}: {np.median(dh_s[i]):.4g}" for i in range(4)])
print("dh nonzero fraction:", (dh_s > 0).mean(), " (zeros where t < injection step?)")
for i in range(4):
    nz = dh_s[i] > 0
    print(f"  s={steps[i]}: nonzero tokens from index {np.argmax(nz.any(axis=0))} to {nz.any(axis=0).nonzero()[0].max()}")
print("dm_s stats: min", dm_s.min(), "max", dm_s.max())
print("corr(dh, |dm|) overall:", np.corrcoef(dh_s.ravel(), np.abs(dm_s).ravel())[0, 1])

stm = np.load(BASE + r"\state_mid.npz")
print("\nstate_mid dh median per s:", [f"{np.median(stm['dh'][i]):.4g}" for i in range(3)])

# --- 9. margins near-ties ---
print("\n--- margins (baseline logit margin, w100) ---")
print("min:", margins.min(), " #<0.05:", (margins < 0.05).sum(), " #<0.1:", (margins < 0.1).sum(), " #<0.2:", (margins < 0.2).sum())
print("5 smallest margins at tokens:", margins.argsort()[:5], margins[margins.argsort()[:5]])

# --- 10. cos semantics ---
print("\ncos at r=1e-3: min/max:", cosz[0].min(), cosz[0].max())
print("cos at r=2: per-layer min:", [f"L{layers[l]}:{cosz[11][:, l].min():.3f}" for l in range(8)])
print("norm_h0 per layer (median):", np.array2string(np.median(norm_h0, axis=1), precision=2))
