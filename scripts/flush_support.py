import numpy as np, os

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
for name, path in [("w100_layers (20 dirs)", os.path.join(B, "w100_layers", "a1.npz")),
                   ("flush w100 (4 dirs)", os.path.join(B, "flush_sweep", "w100", "a1.npz")),
                   ("flush w200 (4 dirs)", os.path.join(B, "flush_sweep", "w200", "a1.npz"))]:
    d = np.load(path, allow_pickle=True)
    P = int(d["prompt_ids"].shape[0]); D = 1536
    sub = d["subspace"]
    pos = np.sort(sub // D)
    print(f"{name}: P={P}, perturbed coordinates at prompt positions "
          f"min={pos.min()} max={pos.max()} (n={len(pos)})")
    for w, tstar in [(100, 92), (100, 78), (200, 178)]:
        pass
    # predicted death time of layer 1
    for w in [100, 200]:
        t_pred = pos.max() + w - 1 - P
        print(f"   W={w}: predicted t*_1 = {t_pred}")
