"""Regenerate every figure in notebook/update.md from raw data.

Run from the repo root:  python analysis_update/run_all.py
"""
import subprocess
import sys
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "pack_fragility.py",
    "pack_rstar_fitcheck.py",
    "pack_depth2.py",
    "pack_depth_tailfit.py",
    "pack_depth_late_rise.py",
    "pack_angles.py",
    "pack_angles2.py",
    "pack_rotation.py",
    "pack_rotation_controls.py",
    "pack_linearity.py",
    "pack_dense2.py",
    "pack_window2.py",
    "pack_window_exact.py",
    "pack_window_fit.py",
    "pack_attention2.py",
    "pack_time.py",
    "pack_time_controls2.py",
    "pack_long_horizon2.py",
    "pack_gate2.py",
    "pack_gate_linlog.py",
    "pack_softloop2.py",
    "pack_loop_feedback.py",
    "pack_loop_metrics.py",
    "pack_misc.py",
    "pack_live_rates.py",
    "pack_rank_entropy2.py",
    "pack_relative_controls.py",
    "pack_summary2.py",
    "pack_master_loglog.py",
    "pack_flips.py",
]
for s in SCRIPTS:
    p = os.path.join(HERE, s)
    if not os.path.exists(p):
        print("missing", s)
        continue
    print("=== ", s, flush=True)
    subprocess.run([sys.executable, p], cwd=os.path.dirname(HERE), check=False)

src = os.path.join(HERE, "..", "update_figures")
dst = os.path.join(HERE, "..", "notebook", "update_figures")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".png"):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print("copied figures to notebook/update_figures")
