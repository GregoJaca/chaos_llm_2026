import io
P = "notebook/update.md"
s = io.open(P, encoding="utf-8").read()

block = r"""## N3. Loop figures from the previous revision (superseded, kept as requested)

These were the loop figures in the last PDF. Every one of them is recomputed *in
place* by the new scripts, so the numbers here are the older 600-step / 8-direction
version. Kept because you asked not to delete figures; the live versions are
F2-F21 above.

![Old F0](update_figures/pack_master_summary.png)

**Old F0 — the four-panel summary (600 steps, DeepSeek-1.5B).** Superseded by F2 (all models) and F5 (sampling).

![Old F0b](update_figures/pack_master_loglog.png)

**Old F0b — log-log closed vs replay for one model.** Superseded by F2/F2b.

![Old F0c](update_figures/pack_loop_loglog_all.png)

**Old F0c — the first all-models log-log grid.** Superseded by F2, which uses per-direction ratios and matched pairs.

![Old F0m](update_figures/pack_loop_saturation.png)

**Old F0m — the saturation summary.** The $L_\infty$/$\tau$ table in F2 is now computed per direction with the fixed-offset fit.

![Old F21](update_figures/pack_loop_temperature.png)

**Old F21 — temperature threshold for one model.** Superseded by F6, which shows the model dependence (the claim "any soft feedback gives the same gain" was too strong).

![Old selfreg](update_figures/pack_loop_selfreg.png)

**Old selfreg — self-regulation with green shading.** Kept, but the shading was removed in F3 as you asked.

"""
anchor = "# N2. Limitations"
assert anchor in s
s = s.replace(anchor, block + anchor, 1)
io.open(P, "w", encoding="utf-8").write(s)
print("appendix added")
