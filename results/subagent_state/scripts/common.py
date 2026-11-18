"""Shared loading utilities for the state-perturbation (R(t,s)) analysis.

All data is read-only. Outputs go to results/subagent_state/{figures,stats}.

Data provenance (verified by scripts/00_audit.py):
- state.npz      : input-embedding injection at s=16,50,100,150, 32 dirs (seeds 1000+i),
                   r=1e-3, measure layer 27, baseline = results/a1_ds_w100/a1.npz (window-100
                   generated trajectory, prompt_len=65, T=300).
- state_mid.npz  : layer-14 residual injection at s=16,50,100, 8 dirs (seeds 2000+i),
                   r=1e-3, measure layer 27.
- NOTE: results/local_data/a1.npz is bit-identical to results/a1_ds_w200_large/a1.npz
  (a DIFFERENT baseline); the correct baseline margins for the state runs live in
  results/a1_ds_w100/a1.npz.
"""
import os
import numpy as np

REPO = r"C:\Users\hjaca\research\chaos_llm_2026"
OUT = os.path.join(REPO, "results", "subagent_state")
FIGS = os.path.join(OUT, "figures")
STATS = os.path.join(OUT, "stats")

STATE_NPZ = os.path.join(REPO, "results", "local_data", "state.npz")
STATE_MID_NPZ = os.path.join(REPO, "results", "local_data", "state_mid.npz")
A1_W100_NPZ = os.path.join(REPO, "results", "a1_ds_w100", "a1.npz")       # true baseline of the state runs
A1_LOCAL_NPZ = os.path.join(REPO, "results", "local_data", "a1.npz")     # = a1_ds_w200_large (NOT the baseline)
MARGINS_W100 = os.path.join(REPO, "results", "local_data", "margins", "w100.npz")

os.makedirs(FIGS, exist_ok=True)
os.makedirs(STATS, exist_ok=True)


def load_state():
    d = np.load(STATE_NPZ)
    return dict(dh=d["dh"].astype(np.float64), steps=d["steps"], r=float(d["r"]),
                layer=int(d["layer"]))


def load_state_mid():
    d = np.load(STATE_MID_NPZ)
    return dict(dh=d["dh"].astype(np.float64), steps=d["steps"], r=float(d["r"]),
                inject_layer=int(d["inject_layer"]), measure_layer=int(d["measure_layer"]))


def load_baseline():
    """Baseline margins (t=0..299) and layer-27 hidden norms for the w100 trajectory."""
    d = np.load(A1_W100_NPZ)
    return dict(margins=d["margins"].astype(np.float64),
                norm_h0=d["norm_h0"][4].astype(np.float64),  # layers saved: 0,7,14,21,27
                gen_ids=d["gen_ids"])


def log10safe(x, floor=1e-12):
    return np.log10(np.maximum(x, floor))


def save_json(obj, name):
    import json
    with open(os.path.join(STATS, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", os.path.join(STATS, name))


def savefig(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=160)
    plt_close(fig)
    print("wrote", path)


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)
