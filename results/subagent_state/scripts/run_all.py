"""Run all state-perturbation analyses in order (each is independent; ~1 min total)."""
import subprocess, sys, os

SCRIPTS = ["00_audit.py", "01_direction_groups.py", "02_shift_collapse.py",
           "03_s_dependence.py", "04_mid_compare.py", "05_viz.py", "06_r_scaling.py"]

here = os.path.dirname(os.path.abspath(__file__))
for s in SCRIPTS:
    print(f"\n=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(here, s)], cwd=here)
    if r.returncode != 0:
        sys.exit(f"{s} failed with code {r.returncode}")
print("\nAll done. Figures in ../figures, stats in ../stats, report in ../REPORT.md")
