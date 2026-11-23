# TODO — soft-loop campaign v2 (user requests + my additions)

## Fix criticized figures
- [ ] sl_gate_collapse (a): remove unexplained dashed red lines / explain; add
      colorbar (colour = log T); legend for every element.
- [ ] sl_gate_collapse (c): redesign completely (data squished 2-5 with a bad
      proportionality line) -> show jump-factor distributions vs m/T bins,
      no fake line.
- [ ] Gate-collapse figure per model + controls (Mamba, Pythia, randQwen),
      not just pooled.
- [ ] sl_decomp (a): lambda_net (red) invisible -> own axis/panel; add
      linear-regime (r->0) rate points; explain pre-vs-sep positivity vs
      the r-dependence result (R2/R5).
- [ ] sl_decomp (c): the often-linear negative trend -> analyse; state which
      part is definitional (sigma saturates while T grows => mean
      log(sigma/T) ~ -log T) and which part is physics.
- [ ] sl_phase: clearer (split frozen point; per-prompt overlay; live-gate
      density as the mediator axis).
- [ ] sl_response / sl_survival / sl_models / sl_jumps: no visible trend ->
      aggregate across prompts/orbits; horizon vs live-gate density (not vs
      T) as the collapsing variable; pooled survival curves.
- [ ] All figures: full labels + legends everywhere.

## New analyses (ambitious)
- [ ] Gate law across ALL 9 models + Mamba + Pythia + randQwen control:
      slope/G_eff per model (from rstar a1 margins+top2 for small models,
      results_big a1 for ds14/qwen314/phi4/qwen35 on spark).
- [ ] Validate the gate law with fresh loop runs per small model
      (T in {0.05,0.1,0.2,1.0}, sig_read measured).
- [ ] TwoNN intrinsic dimension of loop attractors vs T and vs model
      (user: "TwoNN is always a great fit; show the slope as function of
      T"); per-model D2 pipelines + cross-model summary.
- [ ] Dense directions: K=128 grids (ds15, 3 prompts, 3 T) + K=64 (qwen25).
- [ ] Sampled feedback as a random dynamical system: common-random-number
      sampling vs independent sampling (new --feedback modes) — their gap.
- [ ] Gate-based passage-time prediction (gates replace their folds) —
      attempt on ds15 T=0.05-0.2.
- [ ] Cross-scale: 14B gates (from existing fp32 a1) now; 14B loop grid on
      spark when fp32 memory frees (watch loop).
- [ ] Qwen3.5-9B (hybrid) loop grid fp16 on spark.
- [ ] randQwen + Pythia loop grids on x11 (controls).

## Document
- [ ] new.md in notebook/ (with DOCS entry + preview server restart),
      figures copied to notebook/figures/, definitions for every axis,
      data-generation recipes, trivial-vs-nontrivial verdicts.
- [ ] Commit everything as I go (PortableGit).
