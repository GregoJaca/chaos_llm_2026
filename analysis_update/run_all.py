"""Regenerate every figure in notebook/update.md from raw data.

Run from the repo root:  python analysis_update/run_all.py
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "pack_fragility.py",
    "pack_depth2.py",
    "pack_depth_localfit.py",
    "pack_angles.py",
    "pack_rotation.py",
    "pack_linearity.py",
    "pack_dense.py",
    "pack_window2.py",
    "pack_time.py",
    "pack_gate2.py",
    "pack_softloop2.py",
    "pack_misc.py",
    "pack_rank_entropy.py",
    "pack_flips.py",
    "fig_entropy.py",
    "fig_readout_rank.py",
]
for s in SCRIPTS:
    p = os.path.join(HERE, s)
    if not os.path.exists(p):
        print("missing", s)
        continue
    print("=== ", s, flush=True)
    subprocess.run([sys.executable, p], cwd=os.path.dirname(HERE), check=False)

# copy to the previewed notebook folder
src = os.path.join(HERE, "..", "update_figures")
dst = os.path.join(HERE, "..", "notebook", "update_figures")
os.makedirs(dst, exist_ok=True)
import shutil
for f in os.listdir(src):
    if f.endswith(".png"):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print("copied figures to notebook/update_figures")
