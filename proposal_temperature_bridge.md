# From thresholds to feedback: the temperature bridge

*Proposal + first results, 2026-11-22. Written after a full read of the
`neural-shrimp` paper (arXiv:2610.31692) and repo, and of our own corpus
(master/veto/controls/notebook_big + subagent reports). Companion figures:
`results/figures/softloop/*.png`; numbers: `results/figures/softloop/sl_numbers.json`.
New code: `src/chaos_llm/run_softloop.py`, `src/chaos_llm/run_gates.py`,
`scripts/softloop_figs.py`, `scripts/softloop_queue*.sh`.*

---

## 0. One-paragraph summary

Shrimp closes the LLM loop with a **soft token** (the expected embedding of the
model's own next-token distribution) on a **fixed prompt with one feedback
position**, and gets a genuine smooth dynamical system with chaos, periodic
windows, Feigenbaum cascades and computer-assisted orbit proofs. We study
**growing-context greedy decoding** and concluded it is *not* chaotic:
the smooth part is neutral-to-contracting and all divergence is threshold
crossing at logit near-ties. These are the two endpoints of one family that
nobody has measured in between: **the growing-context soft loop with
temperature as the interpolation parameter**, whose T→0 limit is bit-for-bit
our greedy system. First data (tonight, Tesla T4): the loop is **unstable at
every T ≥ 0.05 we tried**, the instability is **entirely in the readout**
(teacher-forced replay along the same orbits gives λ ≈ 0 at every T), and it
is **gated by the same margin dips that control discrete flips at T = 0**:
the per-step readout gain σ_max(Σ_p) is zero at confident steps and spikes by
orders of magnitude exactly at near-ties, with a collapse as a function of the
single ratio **m_t/T**. This turns our margin-landscape corpus into the
predictor of the continuous system's stability, and their temperature
bifurcations into a statement about our margins.

---

## 1. What shrimp does, precisely, and what it does not do

**Their system.** Qwen2.5-0.5B-Instruct, a fixed 37-token chat prefix, and ONE
feedback position. The state is a single embedding vector x ∈ R^896 (or K
vectors for a K-token memory window that slides). The map is
F_{T,α}(x) = E^T softmax(z_α(x)/T), autonomous, real-analytic, mapping the
convex hull of token embeddings into itself. Parameters: temperature T and a
steering vector αv added after block 12 at every position.

**Their results.** Chaotic bands (log10 T ∈ [−0.2, 0.28] roughly, mostly with
steering; 11/14 dense models chaotic unsteered); positive finite-time
Lyapunov exponents (twins at 1e-6, renormalised); periodic windows with
saddle-node scaling, passage-time prediction (42/48 within one iterate),
a Feigenbaum cascade converging to 4.6688 (0.008% from δ), shrimp-shaped
domains; Krawczyk/Arb certificates for three attracting periodic orbits of the
full 24-layer map; the derivative factorisation DF = (1/T)·Cov_p(e)·J_net
(tied) or cross-covariance (untied); the high-T contraction bound
T* = ¼·diam(E)²·L; a batch-layout reproducibility result (173/192 chaotic vs
7/192 periodic cells change text); and the attribution that steering-added
growth comes from the readout, not the network.

**What they explicitly do NOT do** (their own boundaries + repo log):
- growing-context generation ("a different system"; only a twin-growth pilot:
  growth inside the one-token band in 7B, absent in 0.5B/72B);
- teacher-forced linear stability of recorded token sequences ("the sibling
  project's natural object" — their gap list says this verbatim);
- margin/logit-gap statistics (margins appear only inside the low-T gain
  formula e^{−Δ/T}C/T; no distribution, no connection to flip thresholds);
- SSM/Mamba, hybrids, untied-embedding models in detail (3 orbits at 7B);
- sliding windows, KV-cache structure, perturbation-response kernels,
  per-layer attribution, D2 with a scaling region (their own attractor has
  none).

**What we do that they cannot**: we have five models + controls with a
calibrated perturbation protocol (in-batch reference, fp32 floors), the
margin/threshold machinery (r*, signed flip rule, 99.4% prediction), the
teacher-forcing apparatus, the window flush law, and the controls corpus
(Mamba provably contracting under TF, Pythia, random-init). None of that
exists in their repo.

---

## 2. The new object: the growing-context soft-generation loop

Definition (run_softloop.py; everything fp32, in-batch reference row, fixed
batch shapes):

    z_t = model(prompt + e_0 ... e_{t−1})[-1]
    p_t = softmax(z_t / T)                (T = temperature)
    e_t = E^T p_t                         (input-embedding mixture)

- **T→0 is exactly greedy decoding**: verified, the loop's argmax sequence
  reproduces `model.generate` token-for-token (120/120 steps, bit-level; the
  fed embedding of the argmax token is the same tensor the token id looks up).
- **Teacher-forced replay** (mode=replay): every row is fed the *recorded*
  soft tokens of the reference orbit; only the prompt perturbation differs.
  This measures the network-only propagation λ_net along the loop's own
  orbits — the quantity their gap list calls ours to measure.
- Perturbations: the repo-standard sparse simplex on the prompt embedding,
  r = 0.03, 8 directions; response ||δh_t|| read at saved layers.

So the one-parameter family interpolates between:
  T = 0:   our entire greedy corpus (flux tubes, thresholds, flush law);
  T > 0:   their continuous feedback, but *growing*, i.e. actual generation.

**Positioning vs shrimp's growing-context pilot**: their twin-growth result
(7B grows inside its one-token band) is consistent with what we find; what is
new here is (i) the replay decomposition (network vs feedback), (ii) the
margin-gating mechanism with the m/T collapse, (iii) architectures they do
not have, (iv) the bridge to a flip-threshold corpus.

---

## 3. First results (DeepSeek-R1-Distill-Qwen-1.5B, prompts.txt, 600 steps,
8 directions, r = 0.03; full table in sl_numbers.json)

1. **T→0 check: exact.** The loop's argmax equals token-feeding greedy on
   120/120 steps; the T→0 response is flat-to-contracting (λ = −0.0003) and
   0/8 directions flip at r = 0.03 within 600 steps — the frozen, flux-tube
   regime of the whole existing corpus.

2. **The network never destabilises the loop.** Teacher-forced replay along
   every orbit (T = 1e−9 … 0.7 so far) gives λ_net ∈ [−0.0019, +0.0002]
   nats/step — neutral to contracting, with no trend in T. Every bit of the
   temperature dependence is feedback/readout. (This is the cleanest
   possible answer to reviewers R1.1/R2.2 — "components amplify locally,
   but does the composition grow?" — now measured along closed-loop orbits
   at every temperature.)

3. **Separation is a margin-gated staircase, not a smooth exponential.**
   Between gates the response is flat (the cache carries it; the network is
   neutral); at a gate it multiplies by 1–2 orders of magnitude in one step.
   Example (T = 0.5): the orbit has m = 0.02 at t = 2 and m = 0.01 at t = 3;
   the response jumps 0.4 → 4.5 there, hovers ~1 for 70 steps, then a razor
   pair at t ≈ 13–14 (m = 0.11–0.15) throws it to 40–340 in two steps.

   Full grid (median over 8 directions, 600 steps, r = 0.03):

   | T | λ_sep (nats/step) | λ_net (replay) | median t_sat | orbit median margin |
   |---:|---:|---:|---:|---:|
   | 1e−9 | −0.0003 | −0.0003 | >600 (0/8 sep) | 1.50 |
   | 0.05 | +0.044 | −0.0005 | 116 | 1.37 |
   | 0.1 | +0.005 | −0.0017 | >600 (5/8 survive) | 3.74 (confident orbit) |
   | 0.15 | +0.009 | −0.0019 | 193 | 3.66 |
   | 0.2 | −0.010 | −0.0000 | 42 | 1.35 |
   | 0.3 | +0.017 | −0.0009 | 189 | 1.22 |
   | 0.5 | +0.005 | +0.0001 | 72 | 1.49 |
   | 0.7 | **+0.171** | −0.0014 | **22** | 1.39 |
   | 1.0 | +0.047 | −0.0018 | 68 | 5.80 |
   | 1.4 | **+0.426** | −0.0024 | **14** | 2.80 |
   | 2.0 | +0.035 | −0.0016 | 228 | 0.27 (near-flat orbit) |

   Two structural facts in this table. (i) **λ_net has no trend**: the
   network is neutral at every temperature; the whole T-dependence is the
   feedback term. (ii) There is an **instability band**: the horizon
   collapses from >600 below T≈0.1 to 14 steps at T=1.4 and weakens again
   by T=2 (where the orbit's own margins are already ≲ T and the readout
   gain σ/T shrinks — the high-T contraction side). Shrimp's fixed-prefix
   unsteered band sat at log10T ∈ [−0.2, 0.28] (T ≈ 0.6–1.9): the growing
   loop's band on a different model sits in the same place. The
   non-monotonic scatter *within* the band (T=0.1 confident orbit survives;
   T=1.0 queue orbit slow, smoke orbit fast) is orbit-dependence: at fixed T
   the loop is sensitive enough that the batch-shape roundoff selects
   different attractors — their 173/192 reproducibility phenomenon,
   reproduced here deliberately (same T, same prompt, different batch shape
   ⇒ different orbit ⇒ different horizon).

4. **The gate law: log σ_max(Σ_p) = −0.97·(m/T) + log G_eff, R² = 0.995.**
   We record σ_max(Σ_p) per step (cross-covariance of input/output
   embedding rows under the loop's own distribution; estimator validated
   exactly against float64 SVD). Pooled over the five low temperatures
   (T = 0.05…0.3, 1806 live-gate steps), log σ_max is a straight line in
   m_t/T with slope −0.97 (theory −1 for the two-token softmax identity
   σ ≈ p(1−p)·G, p = sigmoid(−m/T)) and G_eff ≈ 1.6 (per-temperature
   slopes −0.95…−0.98, G_eff 1.4–1.8; the direct pair geometry
   G_t = ‖e_i−e_j‖‖w_i−w_j‖ has median 2.35). Per-step values match the
   two-token law point by point (e.g. m/T = 6.7: measured 3.5e-3 vs
   predicted 3.2e-3). At m/T ≲ 2 the amplifier saturates at the p(1−p)
   plateau; at m/T ≳ 30 it hits the one-hot floor (σ = 0 exactly).
   Burst locations confirm the same law from the response side: jumps of
   ≥3× in ‖δh‖ occur at steps with median m/T = 0.6, versus 15.5 at
   non-jump steps, with jump factor falling from ×10 (m/T < 0.5) to ×3
   (m/T > 5). **One measured landscape — the margin/top-2 geometry of the
   orbit — determines the readout amplifier at every temperature, through
   the exact softmax-Jacobian identity.**

5. **The horizon is orbit-dependent through the margin landscape.** Median
   time-to-full-separation: 600 (T→0, censored) → 116 (0.05) → 600 (0.1 —
   that orbit happened to be confident, median margin 3.74) → 193 (0.15) →
   42 (0.2) → 189 (0.3) → 72 (0.5). Non-monotonic in T *because* each T
   generates different text with its own near-ties; the pattern is readable
   from each orbit's own margins. Same structure as our greedy survival
   statistics, one floor up. Above T≈0.5 the horizon is 14–80 steps
   throughout (T=1.4: 14 steps at λ_sep = +0.43) and weakens again by T=2
   (the high-T contraction side) — an instability band in the same place
   where the fixed-prefix maps are chaotic (T ≈ 0.6–1.9).

6. **The separation rate is radius-dependent at large r, but has a true
   r-independent floor inside the band.** Same reference orbit at T=1.0
   (margins verified equal across runs), r ∈ {0.003, 0.01, 0.03, 0.1}:
   λ_sep = +0.056, +0.049, +0.047, +0.144. For r ≤ 0.03 the rate is
   radius-independent (+0.05, IQR +0.035…+0.127 across directions) and the
   measured response sits 8–30× above the fp32 floor at start and 2–3
   orders above it during the measured growth: **a genuine linear-regime
   instability of the growing loop at T=1** (consistent with shrimp's
   7B growing-context twin growth). At r=0.1 an additional absolute
   straddle channel (the greedy flip condition |δm| ≥ m, smeared by the
   mixture) accelerates separation (λ apparent +0.144, t_sat 30 vs 68).
   At T=0.35, by contrast, there is *no* smooth growth in the linear
   regime (λ_sep ≈ 0 at every r ≤ 0.01): separation is *quantized* by rare
   gates — the saturation step jumps between gate events (t=45 or 160–197,
   nothing in between), and the gate gain at the same event spans ×300 to
   ×40,000 across perturbation directions. So the temperature phase
   structure of the growing loop is:
   - T ≲ 0.1: frozen (greedy regime; flux tubes; discrete flips at razor ties);
   - 0.1 ≲ T ≲ 0.5: **gate-schedule divergence** — threshold-like, quantized
     by the orbit's rare margin dips, no smooth rate;
   - 0.5 ≲ T ≲ 2: **dense-gate compounding** — a genuine r-independent
     separation rate (+0.05 nats/step at T=1.0), the smooth instability
     shrimp's fixed-prefix maps exhibit, now measured in the growing system;
   - T ≳ 2: weakening (orbit margins ≲ T, readout σ/T shrinking — the
     high-T contraction side of their T* bound).
   The reviewers' threshold explanation governs the cold edge; genuine
   smooth instability exists only inside the band; *both* are margin-gated.

7. **The window changes the loop's stability qualitatively.** With the true
   sliding window W=100 (departure from full attention verified at exactly
   t=35=W−P): (i) the network part becomes *contracting* — replay gives
   λ_net = −0.005…−0.015 at every T (the flush evicts the perturbation
   support from the cache; the flush law acts inside the loop); (ii) the
   windowed orbits are far more confident (median margins 4.6–9.2 vs 1.4
   at full attention), and the loop survives to 600 steps for all T ≤ 0.15
   (vs separation already at T=0.05 full attention); the onset of fast
   separation moves to T≈0.3. Less memory ⇒ higher onset — the "chaos needs
   memory" theme, measured, and a direct interaction between our window
   law and their feedback loop.

8. **Low-T loop orbits track the greedy orbit and inherit its gates.** At
   T = 0.05 the loop reproduces the greedy argmax for 40 steps, its margins
   agree with the greedy margins to ~0.05 nats, and the first gates (t = 2,
   14, 16) are the greedy orbit's own margin dips. (At T=0.05 the mixture
   even *sharpens* the t=16 near-tie from 0.12 to 0.022 nats.) In the soft
   loop, argmax flips are **not terminal**: after a flip at T=0.2 the
   response can decay back (observed 22 → 0.55 over 30 steps) before later
   gates separate it — re-synchronisation is possible at T>0, unlike
   greedy decoding where 0/75 runs ever re-synchronise.

9. **Orbit sampling across prompts: nothing above T≈0.2 survives.** Two
   further prompts (childhood, cyber; 6 temperatures, 8 directions each)
   give median horizons of 259/92/7/102/113/15 and 34/13/122/216/14/35
   steps for T = 0.1…1.0, with a flip fraction of 1.00 everywhere: at
   every T ≥ 0.2, no orbit on any prompt survived the 600-step window,
   with separation rates up to +0.54 nats/step inside the band (T=0.7).
   The stable zone is the frozen one: T→0 (greedy), where the flux-tube
   picture governs; every softening of the feedback destabilises
   generation on some orbit. The main prompt's T=0.1 survivor was the
   exception, not the rule — orbit heterogeneity, quantified.

Pending (would complete the programme): the hybrid Qwen3.5-2B loop (blocked
by a broken CUDA torch in venv311 on x11 — needs an env rebuild); the 14B
loop grid on the Spark (P6); multi-prompt replay decomposition; and the
packaging decisions (Q1–Q3 below).

**Data and code produced tonight** (all on x11 under
`results/softloop*/`, mirrored to `results/softloop*/` locally):
11-temperature loop+replay grids for DeepSeek-1.5B (full attention and
window-100), Mamba-130m, Qwen2.5-3B; two extra prompts; radius grids at
T=0.35/1.0; four 3000-step attractor runs with saved states; gate
extractions for five loop orbits and three A1 corpora; ~25 figures in
`results/figures/softloop/` plus `sl_numbers.json` (every number above).
Total GPU use: ≈ 4.5 h on the T4.

---

## 4. The programme (concrete, staged)

**P1 — finish the grid (mostly done).** The ds15 grid, the windowed grid,
the radius grids and the long attractor runs are complete; the σ-estimator
rerun (ds15v2, w100v2) is queued. Deliverables so far:
`sl_response_*.png` (the staircase gallery), `sl_decomp_*.png`
(λ_sep / λ_net / gate stats vs T), `sl_survival_*.png` (the horizon
picture), `sl_jumps_*.png` (bursts sit at m_t ≲ T), `sl_models.png`
(cross-architecture, pending mamba/qwen25), `sl_gate_collapse.png`
(after the σ rerun), plus `d2pipe_softloop_*.png` for P3.

**P2 — the one-pass prediction: validated.** From the T→0 loop pass alone
(the greedy orbit; margins m_t and top-1/top-2 ids), `run_gates.py` computes
the pair geometry G_t = ‖e_i−e_j‖‖w_i−w_j‖ (median ≈ 2.35, IQR ±40%).
The two-token law σ ≈ p(1−p)·G_t with p = sigmoid(−m_t/T) then predicts the
*measured* readout σ_max(Σ_p) of the T>0 loop at every step:
- T=0.1 (loop still on the greedy orbit for 16 steps): log-log correlation
  **0.988**, measured/predicted median **1.16** (IQR 0.82–1.54);
- T=0.05 (44 steps of agreement): correlation **0.91**, median ratio 1.33
  (IQR 0.94–2.85).
The ~1.2–1.3× offset is exactly the top-3+ tail contribution to Σ_p
(measured independently in the float64 check: full-64 σ = 0.115 vs
two-token 0.091). **The core object of the whole greedy corpus — the
margin/top-2 landscape of one teacher-forced pass — is the predictor of the
continuous system's amplifier at any temperature.** At higher T the loop
leaves the greedy orbit and its own margins must be used (recorded in every
run).

**P3 — the in-family positive control for our diagnostics: done, and it is a
definitive negative.** 3000-step loop runs (T ∈ {0.5, 1, 2, 1e−9}, states
saved, the validated D2 pipeline with Theiler windows, loop screens, TwoNN,
MLE): the T→0 and T=1 reference orbits are (drifting) repetition loops
(period-13 at 68% coverage, period-29 at 25% — correctly flagged by our
screens); the T=0.5 and T=2.0 orbits are aperiodic, with **no scaling region
(0.00 decades)** — the same verdict as every greedy state cloud we ever
measured, now on a genuinely unstable continuous system, and matching
shrimp's own attractor (their §VIII: "correlation sum has no scaling
region"). The correlation dimension is simply inapplicable to LM state
clouds at these horizons — this closes the D2 chapter (reviewer R1.4) with
an in-family control instead of only synthetic references.

**P4 — the window question (data already coming).** W=100 vs full attention
loops: does memory truncation stabilise or destabilise the continuous loop?
Their fixed-window result says longer memory ⇒ more chaos; our flush law
gives an exact clock for what the window removes. A clean interaction study.

**P5 — architectures they do not have: first results.**
- **Mamba-130m** (SSM, no attention, provably contracting under teacher
  forcing): the soft loop destabilizes *without attention*. λ_net ≈ 0 at
  every T (the SSM never destabilizes itself); the horizon is strongly
  bimodal per orbit — 7–29 steps on orbits that contain a razor tie,
  >600 (frozen) on confident orbits (the tiny model's greedy text is very
  confident, median margins ~8 nats) — t_sat = 19 at T=1.0 (λ_sep +0.23).
  The readout/embedding competition is the instability; attention only
  smooths its statistics. (Mamba stepping needed an HF bug workaround:
  the attention mask breaks `MambaCache.update_conv_state` in 4.50/4.57.)
- **Qwen2.5-3B** (fp16, r=0.1): separates within 5–6 steps at T ≥ 0.5,
  consistent at T→0 with the existing flip corpus (t_sat ~105 at r=0.1,
  vs the published 30.4% flip fraction), one frozen orbit at T=1.4 —
  orbit heterogeneity again.
- **Qwen3.5-2B hybrid: blocked.** The venv311 torch cannot load CUDA on
  the T4 host (import-time failure); needs an env rebuild before the
  hybrid-attention loop can be run. Not urgent: the Mamba result already
  breaks the "attention carries the instability" reading.

**P6 — the 14B on the Spark (the case for ~2–3 h of GPU).** The 14B's razor
margins (min 9.7e-5, 207× thinner than the 1.5B) predict its loop destabilises
at *lower* T than the 1.5B, in the same way its r* is 13× smaller. One
temperature grid (T ∈ {0.02…1}, 300 steps, 8 directions, fp32) ≈ 1–1.5 h on
the Spark + the gate extraction. Payoff: the cross-scale fragility law
(margins ⇒ both discrete r* and continuous T_c) verified at 10× parameters.

**P7 — deliberately NOT proposed** (overlap with shrimp, cite instead):
steering-vector bifurcation scans, periodic-orbit Newton/interval proofs,
passage-time normal forms, fixed-prefix one-token maps on Qwen2.5.
**Later/optional:** sampling as a random dynamical system (their gap #6/#7);
gate-based passage-time prediction (our gates replacing their folds).

---

## 5. How this reshapes the paper

- The reviewers' central objection ("is it chaos or a threshold?") gets a new,
  measured answer: **it is both, on one axis, and the same object controls
  both.** At T = 0 divergence is purely threshold-crossing (our established
  result); at intermediate T the same margins act through the measured gate
  law σ_max(Σ_p) ≈ G_eff·e^{−m/T} as continuous amplifiers with
  threshold-like, quantized separation; inside the band (T ≈ 0.5–2) a
  genuine r-independent separation rate appears; and the network itself is
  never unstable (λ_net ≈ 0 at all T, all architectures). The instability
  lives in the decision geometry, not in the weights' Jacobian —
  quantitatively, at every temperature.
- The margin landscape — already the paper's core object (flip thresholds,
  survival prediction, cross-scale fragility) — now *also* predicts the
  continuous system's amplifier via the gate law, validated from one
  teacher-forced pass (P2). One object, two dynamical regimes.
- The D2/RP story is closed by P3's in-family negative.
- Title options: "Margins, not Jacobians: token-decision geometry controls
  the stability of language-model feedback" / "From flux tubes to gates:
  the temperature phase of LLM generation" / keep the piecewise-smooth
  framing and add the temperature phase as a section.
- New citations: arXiv:2610.31692 (their loop, chaos, certificates, and the
  growing-context twin pilot we confirm and decompose), modern Hopfield
  (their no-network tied-embedding argument), the stable-chaos set (ours).

## 6. Risks

- λ_sep as "the" exponent is crude for a staircase; the horizon statistics,
  the gate law and the jump statistics are the robust objects. Do not claim
  a single Lyapunov exponent; claim the gate mechanism and the band's
  r-independent floor (+0.05 at T=1, r ≤ 0.03).
- Orbit heterogeneity at fixed T (batch-shape roundoff selects different
  attractors at T ≥ 1) is real physics but forces multi-orbit sampling for
  any per-T claim; we did 3 prompts, a paper claim should use the frozen
  10-prompt set.
- fp16 models (Qwen2.5-3B) have exact-tie margins and a response floor; use
  fp32 models for quantitative gates, fp16 only for horizons.
- Their growing-context pilot found 0.5B growth only at sharper T — our
  DeepSeek-1.5B destabilises from T ≈ 0.05; model- and orbit-dependent,
  exactly as the margin picture predicts. Cite them as consistent.
- The cov64 restriction: at T ≥ 1.4 the distribution's effective support
  exceeds 64 tokens, so recorded σ there is a lower bound; the gate law is
  quoted for T ≤ 0.3 where coverage is > 1−10⁻⁶.

## 7. Questions (for the next session)

1. Do we want the Spark for the 14B loop grid (~2–3 h; the prediction is
   T_c ∝ margin scale, i.e. a much colder instability than the 1.5B)?
2. Should the revised manuscript make the temperature phase a full section
   (reframing), a second paper, or a long appendix?
3. Should I fix the venv311 CUDA torch and run the Qwen3.5-2B hybrid loop
   (the Gated-DeltaNet case), or leave architecture coverage at
   Mamba + 3 attention models?
4. Do we freeze the 10-prompt set now and re-run the loop grid on it for
   the paper-grade statistics (≈ 2 h T4), as the old notebook §12.4 asked
   for the greedy side too?
