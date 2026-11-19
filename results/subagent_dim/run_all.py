"""run_all.py -- reproduce everything from the raw npz, in order.

Runtime ~1 h on a laptop CPU (s2_hidden.py dominates).  Sections write
their own part of numbers.json; figures land in figs/.
"""
import subprocess
import sys

STEPS = ["s0_check.py", "s1_vocab.py", "s2_hidden.py", "s2b_figs.py",
         "s3_logits.py", "s4_cross.py"]

if __name__ == "__main__":
    for s in STEPS:
        print(f"\n########## {s}", flush=True)
        subprocess.run([sys.executable, "-u", s], check=True)
