import numpy as np
base = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
a100 = np.load(base + r"\w100_layers\a1.npz", allow_pickle=True)
for w in ["w100", "w150", "w200", "w300", "w1000"]:
    d = np.load(base + rf"\window_fixed\{w}\a1.npz", allow_pickle=True)
    gen = d["gen_ids"]
    same_prompt = (d["prompt_ids"] == a100["prompt_ids"]).all()
    same_prompt_text = bool(same_prompt)
    pre = (np.asarray(gen[:300]) == a100["gen_ids"]).all()
    print(f"window_fixed {w}: gen len {len(gen)}, radii {d['radii']}, prompt same? {same_prompt_text}, first-300 tokens same as w100_layers? {pre}")
    print(f"   keys: {sorted(d.files)}")
    print(f"   min margin {float(np.min(d['margins'])):.4f}; n margins<0.2: {(d['margins']<0.2).sum()}/{len(d['margins'])}")
# margin stats for w100/w200/w500 a1
for w in ["w100_layers", "w200", "w500"]:
    d = np.load(base + rf"\{w}\a1.npz", allow_pickle=True)
    m = np.asarray(d["margins"], float)
    print(f"{w}: horizon {len(m)}, min margin {m.min():.4f} at {int(np.argmin(m))}, "
          f"q01 {np.quantile(m,0.01):.3f}, frac<0.2 {(m<0.2).mean():.3f}, frac<0.5 {(m<0.5).mean():.3f}")
