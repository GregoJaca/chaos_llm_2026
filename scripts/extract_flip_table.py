#!/usr/bin/env python3
"""Extract a compact table of flip statistics from results/sens (runs on x11)."""
import glob
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/LLM/chaos_llm_2026/results")
out = []
for model in ["deepseek", "qwen3_4b", "phi4mini", "qwen3_5", "qwen25_3b"]:
    for d in sorted(glob.glob(os.path.join(ROOT, "sens", f"{model}_p*"))):
        prompt = os.path.basename(d)
        geom = os.path.join(d, "geom", "a1.npz")
        if not os.path.exists(geom):
            continue
        m = np.load(geom)["margins"].astype(float)
        tie = m <= 0.02
        for r in ["0.1", "0.3", "0.6", "1.0"]:
            f = os.path.join(d, f"r{r}", "free_run.npz")
            if not os.path.exists(f):
                continue
            obs = np.load(f, allow_pickle=True)["obs"].ravel()
            n = int(len(obs))
            flips = obs >= 0
            nontrivial = flips.copy()
            for i in range(n):
                if obs[i] >= 0 and obs[i] < len(tie) and tie[obs[i]]:
                    nontrivial[i] = False
            out.append(dict(model=model, prompt=prompt, r=float(r), n=n,
                            flips=int(flips.sum()), nontrivial=int(nontrivial.sum()),
                            n_tie=int(tie.sum())))
json.dump(out, open(os.path.join(ROOT, "flip_table.json"), "w"))
print("rows", len(out))
