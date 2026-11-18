"""Stage A2: pin down dh_last columns, pred_T semantics, dm zeros."""
import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
z = np.load(BASE + r"\w100_layers\a1.npz")
dh = z["dh_norm"].astype(np.float64)
dm = z["dm"].astype(np.float64)
margins = z["margins"].astype(np.float64)
r_c = z["r_c"].astype(np.float64)
radii = z["radii"]; layers = z["layers"]
dh_last = z["dh_last"].astype(np.float64)
pred_T = z["pred_T"]

# --- dh_last columns: directions at last layer? ---
print("--- dh_last column identity ---")
for i in range(3):
    n_last = np.linalg.norm(dh_last[4, i], axis=-1)  # r=0.2, [T]
    best = None
    for d in range(20):
        for l in range(8):
            if dh[4, d, l].max() == 0:
                continue
            cc = np.corrcoef(n_last, dh[4, d, l])[0, 1]
            if best is None or cc > best[0]:
                best = (cc, d, l)
    print(f"  col {i}: best corr {best[0]:.4f} with dir {best[1]}, layer {layers[best[2]]}")
# check exact equality with a specific (dir, layer)
for i in range(3):
    for d in range(3):
        for l in [6, 7]:
            cc = np.corrcoef(np.linalg.norm(dh_last[4, i], axis=-1), dh[4, d, l])[0, 1]
            if cc > 0.995:
                print(f"  col {i} == dir {d} layer {layers[l]}: corr {cc:.5f}")

# --- pred_T semantics ---
print("\n--- pred_T ---")
manual = np.full((12, 20), -1, dtype=int)
for r in range(12):
    cross = (r_c[r] < radii[r])  # [D,T]
    for d in range(20):
        idx = np.argmax(cross[d])
        manual[r, d] = idx if cross[d, idx] else -1
print("pred_T == manual(first t with r_c<r):", (pred_T == manual).all())
if (pred_T != manual).any():
    diff = (pred_T != manual)
    print("  n diff:", diff.sum())
    rr, dd = np.where(diff)
    for k in range(min(5, len(rr))):
        r, d = rr[k], dd[k]
        print(f"  r={radii[r]} d={d}: stored {pred_T[r,d]} vs manual {manual[r,d]}")

fr = np.load(BASE + r"\fr_w100_large.npz", allow_pickle=True)
print("fr pred[0] == pred_T[7] (r=0.5):", (fr["pred"][0] == pred_T[7]).all())
print("fr pred[1] == pred_T[9] (r=1):", (fr["pred"][1] == pred_T[9]).all())
print("fr pred[2] == pred_T[11] (r=2):", (fr["pred"][2] == pred_T[11]).all())

# --- dm exact zeros / quantization at small r ---
print("\n--- dm zeros ---")
for r in range(12):
    zc = (dm[r] == 0).mean()
    print(f"  r={radii[r]:g}: frac exactly 0: {zc:.3f}, frac |dm|<1e-6: {(np.abs(dm[r])<1e-6).mean():.3f}")

# --- how well does linear r_c (from small r) predict r_c at large r? ---
print("\n--- r_c scale drift (median finite r_c per radius) ---")
for r in range(12):
    fin = np.isfinite(r_c[r]) & (r_c[r] < 1e20)
    print(f"  r={radii[r]:g}: median r_c={np.median(r_c[r][fin]):.3f}, frac dm<0: {(dm[r]<0).mean():.3f}")

# --- margins of near-tie tokens ---
tnt = margins.argsort()[:12]
print("\nnear-tie tokens:", tnt)
print("their margins:", np.array2string(margins[tnt], precision=3))
print("token 2 margin:", margins[2], " token 123:", margins[123], " token 143:", margins[143], " token 16:", margins[16], " token 14:", margins[14])

# --- state dm zeros before s? ---
st = np.load(BASE + r"\state.npz")
dm_s = st["dm"]
for i, s in enumerate(st["steps"]):
    print(f"\nstate s={s}: dm zero frac before s: {(dm_s[i, :, :s]==0).mean():.3f}, after: {(dm_s[i, :, s:]==0).mean():.3f}")
