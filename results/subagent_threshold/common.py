"""Shared helpers: load perturbation/margin data for the DeepSeek-1.5B threshold analysis."""
import json
import os

import numpy as np

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_threshold"

# window -> (npz path, index of the reference (smallest usable) radius slice)
FILES = {
    "w100": (os.path.join(BASE, "w100_layers", "a1.npz"), 2),  # r_ref = 0.01
    "w200": (os.path.join(BASE, "w200", "a1.npz"), 0),         # r_ref = 0.01
    "w500": (os.path.join(BASE, "w500", "a1.npz"), 0),         # r_ref = 0.01
}

FREE = {
    "w100": os.path.join(BASE, "free_w100_large.json"),
    "w200": os.path.join(BASE, "free_w200.json"),
    "w500": os.path.join(BASE, "free_w500.json"),
    "w100L": os.path.join(BASE, "free_large.json"),  # w100_layers, r = 0.5/1/2
}


class Case:
    """One window case: baseline margins + direction-level perturbation response."""

    def __init__(self, name):
        path, ir_ref = FILES[name]
        d = np.load(path, allow_pickle=True)
        self.name = name
        self.window = int(name[1:])
        self.d = d
        self.radii = d["radii"]
        self.m = d["margins"].astype(float)          # (T,) baseline top1-top2 margin
        self.T = len(self.m)
        self.top1, self.top2 = d["top1"], d["top2"]
        self.gen_ids = d["gen_ids"]
        self.dm = d["dm"].astype(float)              # (n_r, 20, T) margin change
        self.n_dirs = self.dm.shape[1]
        self.rc = d["r_c"].astype(float)             # (n_r, 20, T) = r*m/|dm|, sentinel 1e30*m
        self.rc[self.rc > 1e29] = np.inf
        self.pred_T = d["pred_T"]                    # (n_r, 20)
        self.simplex = d["simplex"]                  # (20, 20) direction vectors
        self.ir_ref = ir_ref
        self.r_ref = float(self.radii[ir_ref])
        self.gain = np.abs(self.dm) / self.radii[:, None, None]  # |dm|/r  (n_r, 20, T)

    # ---- derived per-direction quantities at the reference radius ----
    def rc_ref(self):
        return self.rc[self.ir_ref]                  # (20, T)

    def r_star(self):
        """Local threshold per direction: r_* = min_t r_c (linear, small-r)."""
        rc = self.rc_ref()
        return rc.min(axis=1), rc.argmin(axis=1)     # (20,), (20,)

    def free(self):
        with open(FREE[self.name]) as f:
            return json.load(f)

    def free_L(self):
        with open(FREE["w100L"]) as f:
            return json.load(f)


def load_margin_traces():
    """Baseline margin traces m_t for windows 50..4096 (T=600) + summary stats."""
    traces = {}
    for w in [50, 100, 200, 500, 1000, 4096]:
        d = np.load(os.path.join(BASE, "margins", f"w{w}.npz"))
        traces[w] = d["margins"].astype(float)
    with open(os.path.join(BASE, "margins", "summary.json")) as f:
        summary = json.load(f)
    return traces, summary


def obs_pred_at(case_name, r):
    """Observed / predicted first-flip steps (free runs) for one window at radius r."""
    j = json.load(open(FREE[case_name]))
    for i, rr in enumerate(j["radii"]):
        if abs(rr - r) < 1e-9:
            return np.array(j["obs"][i]), np.array(j["pred"][i])
    return None, None
