"""s1_explore_divergence.py -- quick numbers on what a flip costs downstream.

For every free-run record with sequences (fr_w100_large, fr_cyber, fr_childhood):
  - per flipped (radius, direction): fraction of tokens identical to reference
    after the first flip, whether the run re-syncs, block-agreement windows
  - aggregate: mean agreement vs steps after the first flip, per radius
Writes divergence_explore.txt.
"""
import os

import numpy as np

import common2 as C

lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


np.set_printoptions(linewidth=200, precision=3, suppress=True)

records = C.load_free("all")
seq_recs = [e for e in records if e["npz"] is not None]

for e in seq_recs:
    f = e["npz"]
    ri = int(np.argmin(np.abs(f.radii - e["r"])))
    p("=" * 72)
    p(f"{e['source']}  r={e['r']:g}  prompt={e['prompt']}  obs flips: "
      f"{int((e['obs'] >= 0).sum())}/{e['obs'].size}")
    for di in range(f.K):
        T = int(e["obs"][di])
        if T < 0:
            continue
        ref, row = f.rows(ri, di)
        st = C.after_flip_stats(ref, row, T, block=10, tail=50)
        p(f"  d{di:2d} T={T:3d}  frac_ident_after={st['frac_identical_after']:.3f} "
          f"all={st['frac_identical_all']:.3f} resynced={st['resynced']} "
          f"resync_step={st['resync_step']} "
          f"first10={st['agree_first_block']:.2f} lasthalf={st['agree_last_half']:.2f}")

# aggregate curves: agreement vs steps after first flip
p("=" * 72)
p("AGGREGATE: mean token agreement vs steps after the first flip (block=10)")
for e in seq_recs:
    f = e["npz"]
    ri = int(np.argmin(np.abs(f.radii - e["r"])))
    curves = []
    for di in range(f.K):
        T = int(e["obs"][di])
        if T < 0:
            continue
        ref, row = f.rows(ri, di)
        a = C.agreement_curve(ref, row)[T:]
        curves.append(a)
    if not curves:
        p(f"{e['source']} r={e['r']:g}: no flips")
        continue
    L = min(len(c) for c in curves)
    M = np.stack([c[:L] for c in curves])          # (nflip, L)
    blocks = M[:, :(L // 10) * 10].reshape(len(curves), -1, 10).mean(axis=(0, 2))
    p(f"{e['source'][:-4]:>18s} r={e['r']:<4g} nflip={len(curves):2d} "
      f"blocks0-10,10-20,...: {np.array2string(blocks[:12], precision=2)}")

# pooled across everything, per radius bucket
p("=" * 72)
p("POOLED per prompt: frac of flipped runs fully re-synced (last 50 identical)")
for src in ["fr_w100_large.npz", "fr_cyber.npz", "fr_childhood.npz"]:
    e = [x for x in seq_recs if x["source"] == src]
    for x in e:
        f, ri = x["npz"], int(np.argmin(np.abs(x["npz"].radii - x["r"])))
        nflip = resync = 0
        fr_after = []
        for di in range(f.K):
            T = int(x["obs"][di])
            if T < 0:
                continue
            ref, row = f.rows(ri, di)
            st = C.after_flip_stats(ref, row, T, tail=50)
            nflip += 1
            resync += int(st["resynced"])
            fr_after.append(st["frac_identical_after"])
        p(f"{src[:-4]:>18s} r={x['r']:<4g}: flipped {nflip:2d}, re-synced {resync:2d}, "
          f"mean frac identical after flip {np.mean(fr_after) if fr_after else float('nan'):.3f}")

# what about single-token blips? count of differing tokens overall
p("=" * 72)
p("Difference counts: total differing tokens / 300 per flipped run")
for e in seq_recs:
    f = e["npz"]
    ri = int(np.argmin(np.abs(f.radii - e["r"])))
    cnts = []
    for di in range(f.K):
        T = int(e["obs"][di])
        if T < 0:
            continue
        ref, row = f.rows(ri, di)
        cnts.append(int((ref != row).sum()))
    if cnts:
        p(f"{e['source'][:-4]:>18s} r={e['r']:<4g}: n_diff tokens {sorted(cnts)}")

with open(os.path.join(C.OUT, "divergence_explore.txt"), "w") as fo:
    fo.write("\n".join(lines) + "\n")
print("\nwrote divergence_explore.txt")
