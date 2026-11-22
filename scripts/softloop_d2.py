"""Prepare soft-loop attractor states for the D2 pipeline and run it.

Takes results/softloop_states/ds15/T*/softloop.npz (h_ref = last-layer hidden
state of the reference row at every step, from run_softloop --save-states),
writes h_last npz copies, and runs the spark-worktree D2 pipeline
(d2_local.py: correlation integral with Theiler windows, TwoNN, kNN MLE,
repetition/loop screens) on them, plus a recurrence plot at matched RR.
"""
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.normpath(os.path.join(HERE, '..', '..'))
D2 = os.path.join(RESEARCH, 'chaos_spark_wt', 'scripts_spark', 'd2_local.py')

SRC = os.path.join(HERE, '..', 'results', 'softloop_states', 'ds15')
TMP = os.path.join(HERE, '..', 'results', 'softloop_states', '_pt')
OUT = os.path.join(HERE, '..', 'results', 'figures', 'softloop')
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

files = []
for d in sorted(glob.glob(os.path.join(SRC, 'T*'))):
    f = os.path.join(d, 'softloop.npz')
    if not os.path.exists(f):
        continue
    z = np.load(f)
    if 'h_ref' not in z.files:
        print('no h_ref in', f)
        continue
    T = os.path.basename(d)
    out_f = os.path.join(TMP, f'loop_{T}.npz')
    np.savez(out_f, h_last=z['h_ref'].astype(np.float32))
    files.append(out_f)
    print('prepared', out_f, z['h_ref'].shape)

if not files:
    sys.exit('no state files yet')
cmd = [sys.executable, D2, '--pt'] + files + \
      ['--theiler', '0,50,500', '--out', OUT, '--tag', 'softloop']
print('RUN', ' '.join(cmd))
subprocess.run(cmd, check=False)
