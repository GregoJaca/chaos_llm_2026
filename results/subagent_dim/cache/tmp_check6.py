import numpy as np
from dimlib import DIMROOT
import os
for tag in ["deepseek", "qwen3_5"]:
    v = np.load(os.path.join(DIMROOT, tag, "vocab.npz"))
    GE = v["GE"]; smp = v["sample_E"].astype(np.float32)
    normE = v["normE"]; V = int(v["V"][0]); s = v["sE"]
    # row cosines within the uniform 1024-row sample (exact pairs)
    Xn = smp / np.linalg.norm(smp, axis=1, keepdims=True)
    G = Xn @ Xn.T
    iu = np.triu_indices(len(smp), 1)
    c = G[iu]
    mu = smp.mean(0)
    print(f"== {tag}: V={V}")
    print(f"  row cos (1024-sample, all {len(c)} pairs): mean {c.mean():+.4f} "
          f"se {c.std()/np.sqrt(len(c)):.5f}  frac|c|>0.3: {(np.abs(c)>0.3).mean():.4f}")
    print(f"  ||mu_hat||/mean||e|| = {np.linalg.norm(mu)/normE.mean():.4f}"
         f"   2-sigma noise floor = {2*np.sqrt((normE.mean()**2)/1024)/normE.mean():.4f}")
    # dimension cosines from the D x D Gram (exact, 20k pairs)
    rng = np.random.default_rng(0)
    D = GE.shape[0]
    i = rng.integers(0, D, 20000); j = rng.integers(0, D, 20000)
    keep = i != j; i, j = i[keep], j[keep]
    d = GE[i, j] / np.sqrt(GE[i, i] * GE[j, j])
    print(f"  dim cos (from Gram, {len(d)} pairs): mean {d.mean():+.4f}")
    print(f"  top_frac s1^2/sum s^2 = {s[0]**2/(s**2).sum():.4f}")
