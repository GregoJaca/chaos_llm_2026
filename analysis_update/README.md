# analysis_update — how to reproduce the figures in `notebook/update.md`

All scripts read the raw `.npz` files in `../results/` and `../results_big_small/` and write
`../update_figures/*.png` plus a small JSON of numbers. Nothing is synthesised.

```
python analysis_update/fig_depth.py        # Fig 1, 2  (depth, block Jacobians)
python analysis_update/fig_angle.py        # Fig 3     (response angle, rotation)
python analysis_update/fig_time.py         # Fig 4     (A1 + replay time dynamics)
python analysis_update/fig_gate.py         # Fig 5     (gate law, 9 models)
python analysis_update/fig_rstar.py        # Fig 6     (r* ECDF, controlling steps)
python analysis_update/fig_rstar_pdf.py    # Fig 6b    (r* density)
python analysis_update/fig_window.py       # Fig 7     (window departure / flush)
python analysis_update/fig_softloop.py     # Fig 8     (closed loop vs replay)
python analysis_update/fig_phase.py        # Fig 8b    (horizon vs gate density)
python analysis_update/fig_entropy.py      # Fig 9     (entropy vs margin)
python analysis_update/fig_rank.py         # Fig 10    (readout effective rank)
python analysis_update/fig_relative.py     # Fig 11    (relative response vs depth)
python analysis_update/fig_response_r.py   # Fig 12    (response vs radius, superposition)
```

`big_a1_summary.json` was produced on the Spark (the 14B A1 files are 200–300 MB) with
`analysis_update/extract_big.py`; it is the only derived file that is not computed locally.

Numbers quoted in the document are in `depth_numbers.json`, `angle_numbers.json`,
`time_numbers.json`, `gate_numbers.json`, `rstar_numbers.json`, `softloop_numbers.json`,
`phase_numbers.json`, `radius_numbers.json`.

To rebuild the PDF:
```
cd ../notebook
python .preview/build.py
```
Then open `http://100.101.197.82:8899/update.pdf`.
