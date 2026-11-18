import numpy as np
base = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
a2 = np.load(base + r"\w200\a1.npz", allow_pickle=True)
a5 = np.load(base + r"\w500\a1.npz", allow_pickle=True)
print("w200 gen == w500 gen (first 300)?", (a2["gen_ids"][:300] == a5["gen_ids"][:300]).all())
print("w200 gen len:", len(a2["gen_ids"]), "w500:", len(a5["gen_ids"]))
d = np.where(np.asarray(a2["gen_ids"][:300]) != np.asarray(a5["gen_ids"][:300]))[0]
print("first differing step:", d[:5] if len(d) else "none")
mg2, mg5 = np.asarray(a2["margins"], float), np.asarray(a5["margins"], float)
print("margins equal?", np.allclose(mg2, mg5))
print("w200 margins[40:50]:", np.round(mg2[40:50], 4))
print("w500 margins[40:50]:", np.round(mg5[40:50], 4))
# also check prompt: same prompt, and where the reference starts to differ from w100
a1 = np.load(base + r"\w100_layers\a1.npz", allow_pickle=True)
d100 = np.where(np.asarray(a1["gen_ids"]) != np.asarray(a2["gen_ids"][:300]))[0]
print("w100 vs w200 reference first difference:", d100[:5] if len(d100) else "none")
