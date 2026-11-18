import numpy as np, sys
sys.path.insert(0, r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos")
from stablechaos_lib import *
d = np.load(BASE + r"\fr_w100_extra.npz", allow_pickle=True)
print("keys:", d.files)
for k in d.files:
    a = d[k]
    print(f"  {k}: {a.shape} {a.dtype}")
print("radii:", d["radii"])
print("obs:\n", d["obs"])
print("pred:\n", d["pred"])
# batch structure
g = d["gen_ids"]
tot = 0
for el in g:
    tot += np.asarray(el).shape[0] - 1
print("perturbed runs:", tot)
# which r_c field reproduces pred?
a1 = load_a1(r"w100_layers\a1.npz")
r_c, radii_a1 = a1["r_c"], a1["radii"]
for k, r in enumerate(d["radii"]):
    for i in range(12):
        man = np.argmax(r_c[i] < r, axis=1)
        man[r_c[i].min(axis=1) >= r] = -1
        if np.array_equal(man, d["pred"][k]):
            print(f"r={r}: pred == first-t r_c[i={i}](r={radii_a1[i]})<{r}")
            break
    else:
        print(f"r={r}: no exact field match among i=0..11")
# ref match?
ref = np.asarray(d["ref_ids"])[0]
print("ref == w100 main ref?", (ref == np.asarray(np.load(BASE + r"\fr_w100_large.npz", allow_pickle=True)["ref_ids"])[0]).all())
