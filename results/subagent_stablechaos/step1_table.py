"""Step 1: determine the exact threshold-model rule used for `pred` (incl. r=0.05),
   then tabulate the flip-time dataset."""
import numpy as np, json, sys
sys.path.insert(0, r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos")
from stablechaos_lib import *

a1 = load_a1(r"w100_layers\a1.npz")
r_c, radii_a1, pred_T = a1["r_c"], a1["radii"], a1["pred_T"]

mid = load_free("fr_w100_mid.npz")
pred005 = mid["pred"][0]
# candidates for the r=0.05 field: r_c at radius 0.1 (i=4), 0.03 (i=3), 0.01 (i=2)
for i in [2, 3, 4]:
    man = np.argmax(r_c[i] < 0.05, axis=1)
    man[r_c[i].min(axis=1) >= 0.05] = -1
    print(f"pred[0] (r=0.05) == first-t r_c[i={i}](r={radii_a1[i]})<0.05 ? {np.array_equal(pred005, man)}")
# also try scaled fields (linear-response rescale from radius ri to 0.05)
for i in [4, 5]:
    field = r_c[i] * (0.05 / radii_a1[i])
    man = np.argmax(field < 0.05, axis=1)
    man[field.min(axis=1) >= 0.05] = -1
    print(f"pred[0] == first-t (r_c[i={i}]*0.05/{radii_a1[i]})<0.05 ? {np.array_equal(pred005, man)}")

print()
# ---- flip-time tabulation for all four free-run files ----
files = [("main_w100_large", "fr_w100_large.npz"), ("main_w100_mid", "fr_w100_mid.npz"),
         ("cyber_w100", "fr_cyber.npz"), ("childhood_w100", "fr_childhood.npz")]
table = {}
for tag, fn in files:
    d = load_free(fn)
    for k, r in enumerate(d["radii"]):
        T = d["obs"][k]
        flips = T[T >= 0]
        key = f"{tag}_r{r:g}"
        table[key] = dict(radius=float(r), n=20,
                          n_flip=int((T >= 0).sum()),
                          Ts=sorted(flips.tolist()),
                          mean_T_obs_flips=float(flips.mean()) if len(flips) else None,
                          mean_T_censored=mean_restricted(T),
                          pred_n_flip=int((d["pred"][k] >= 0).sum()),
                          pred_Ts=sorted(d["pred"][k][d["pred"][k] >= 0].tolist()),
                          exact_pred_match=int((d["pred"][k] == T).sum()))
        row = table[key]
        print(f"{tag:18s} r={r:5.2f}: flips {row['n_flip']:2d}/20 (pred {row['pred_n_flip']:2d}), "
              f"exact T match w/ pred {row['exact_pred_match']:2d}/20, "
              f"meanT(cens) {row['mean_T_censored']:6.1f}, Ts={row['Ts']}")
        print(f"{'':18s}          pred_Ts={row['pred_Ts']}")

# margin dip steps
mg = a1["margins"]
dipsteps = np.argsort(mg)[:12]
print("\nsmallest-margin steps:", sorted(dipsteps.tolist()))
print("margins at those:", np.round(np.sort(mg)[:12], 4))

# T-value clustering: pooled Ts per prompt
main = [("fr_w100_large.npz"), ("fr_w100_mid.npz")]
allT = []
for fn in main:
    d = load_free(fn)
    for k in range(d["obs"].shape[0]):
        allT += d["obs"][k][d["obs"][k] >= 0].tolist()
allT = np.array(allT)
vals, cnts = np.unique(allT, return_counts=True)
print("\npooled main-prompt flip steps (value: count):")
for v, c in zip(vals, cnts):
    print(f"  T={v:3d}: {c}")
json.dump(table, open(OUT + r"\fliptimes.json", "w"), indent=1)
print("\nsaved fliptimes.json")
