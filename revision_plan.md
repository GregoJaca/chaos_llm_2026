# Revision plan: "Chaos in reason" (Chaos, CHA26-AR-02086)

*Internal handout for the co-authors. Line numbers refer to the submitted `main.tex`. Reference numbers [n] point to the list at the end.*

---

## 0. Summary

- Both reviewers recommend "reconsider after major revision" and are positive about the topic.
- They raise the **same central objection**: we show sensitivity, boundedness and structure. Chaos requires all three, but they do not prove it. The jump-like divergence could simply come from the discrete token choice (argmax) and not from exponential amplification.
- **Our own figures partly support this objection.**
  - The caption of Fig. 4 says the hidden-state distance *decreases* (negative finite-time exponent) until the first token flip.
  - Fig. 7 shows that a perturbation grows in the first layers, then plateaus, then shrinks.
- Proposed plan:
  1. Run 2–3 decisive experiments first, and let them decide the framing.
  2. Add matched non-chaotic and chaotic control systems.
  3. Make the recurrence and dimension analyses quantitative and robust.
  4. Narrow the claims, and state the new contribution clearly.
- The most likely outcome, in our view, is a framing that is both safer and more novel. Under greedy decoding an LLM is a **piecewise-smooth system**: a smooth part that is marginal or contracting, combined with discrete switching between tokens. The unpredictability then comes from finite perturbations crossing decision boundaries. Spiking neural networks show the same behaviour, known as "stable chaos" or "flux tubes" [1–3].

---

## 1. What the reviewers ask, grouped by theme

| Theme | Comments | Core question |
|---|---|---|
| T1. Chaos or threshold? | R1.1, R1.2, R2.1, R2.2 | Component Jacobian norms > 1 do not give a positive Lyapunov exponent for the full map. Finite-time growth also happens in non-chaotic systems. Token flips can cause divergence without chaos. |
| T2. Specificity of the diagnostics | R2.3, R1.3, R1.4 | Recurrence plots (RPs) and correlation dimension D₂ can look similar for correlated but non-chaotic signals. The RR choice (0.03 / 0.05) and convergence of D₂ with N need to be shown. |
| T3. Scope | R1.5 | One model and a few prompts; the claims must be restricted or the experiments extended. |
| T4. Contribution | R1.6, R2.4, R2.5 | What does the chaos picture explain or predict beyond known Transformer properties? The link to chain-of-thought (CoT) is not shown. |

---

## 2. Problems the reviewers did not mention (fix before round 2)

1. **Perturbation size vs. numerical precision.**
   - L1023 states that r must exceed the FP16 noise floor (≈ 5×10⁻⁴). But the experiments use r = 2.5–5×10⁻⁴ (L474, L1016), and Fig. 9 uses 10⁻⁴.
   - It is unclear whether ‖Δ‖ = r is the norm of the whole L×D tensor or of each token. If it is the whole tensor, each entry is about r/√(L·D) ≈ 10⁻⁶. That is below the FP16 spacing of typical embedding entries (≈ 1.5×10⁻⁵ for |x| ≈ 0.02).
   - If Δ is added to FP16 embeddings, most entries round to zero. The real perturbation would then be a sparse set of 1-ulp changes with an unknown norm.
   - **Check:** compute ‖fp16(E₀+Δ) − fp16(E₀)‖ / r.
   - In any case, rerun the core experiments in FP32. This also allows scanning r over several decades (Sec. 5.1, A2).
2. **The prompt perturbation never leaves the state.**
   - The perturbed prompt stays in the KV cache for the whole generation. So it acts as a persistent, parameter-like perturbation, not a perturbation of the current state.
   - Its influence is diluted as the context grows. This probably explains the negative early exponent (Fig. 4) and the "stable" runs in the attention-window experiment (Fig. 9).
   - We should say this explicitly, and add perturbations of the *current* state (the residual stream at generation step s).
3. **Overclaims to remove or soften:**
   - Abstract: "demonstrate that LLMs exhibit hallmark signatures of chaos", "fractal structures".
   - L805, L857: "consistent with low-dimensional deterministic chaos".
   - L751: short diagonals in RPs as evidence of a positive Lyapunov exponent. Coloured noise also produces short diagonals.
   - Boundedness is enforced by normalization by construction, so it is not evidence for an attractor.
4. **Missing statistics.** Report the number of prompts, the ensemble size per prompt and confidence intervals. The "~90% latent-then-jump" figure (L642) needs an N.
5. **Minor:**
   - Redundant 1/Δt in the λ formula (L578).
   - "Context window 3096" (L472, L1015): typo for 4096?
   - "discrete-space chaotic systems" (L646) is unclear.
   - The Phi-4 replication is cited only to the thesis; move it to the supplementary material.
   - The RQA table is commented out (L783–800).
   - "Data available on request": release code and data (GitHub + Zenodo).
   - `\orange` is mapped to black (L80). For the resubmission, produce a latexdiff PDF.

---

## 3. Key concepts

### 3.1 Free-running vs. teacher-forced generation

**Free-running generation** is what our experiments do now. The model computes logits, picks the token with argmax, and feeds that token back as the next input:

```
h_t → logits_t → argmax → token_t → embedding → h_{t+1} → ...
 ▲                                                  │
 └────────────── closed feedback loop ──────────────┘
```

A perturbed and an unperturbed run each follow their own choices. After the first differing token they receive different inputs. Argmax is discontinuous, so derivatives of the full map are not defined across a token flip.

**Teacher forcing** is a term from training recurrent and language models (standard reference [8]): the model is fed a *given* token sequence instead of its own predictions. For our purpose:

1. Run the unperturbed model normally and record its tokens y₁ … y_T.
2. Run a second pass with the perturbed prompt followed by **the same tokens y₁ … y_T**.
3. The model still computes hidden states and logits at every position, but its own choice is never fed back.

Both runs therefore always see the same tokens, and **no token can flip**. The map from the perturbation to the hidden-state difference δh_t is a composition of smooth operations (attention, MLP, normalization), so derivatives exist. Any growth of δh_t comes only from the continuous dynamics.

**Physics analogy.** The token sequence acts as a fixed external drive, making the hidden-state dynamics a driven, non-autonomous system. Teacher forcing asks whether the driven system forgets or amplifies a perturbation when the drive is held fixed. This is the **conditional Lyapunov exponent** of a drive–response system, as in chaos synchronization [9] (the 1990 paper calls them "sub-Lyapunov exponents"):

- Negative: the hidden states "synchronize" to the token stream and forget the perturbation.
- Positive: they amplify it even with identical inputs.

**Practical advantages:**

- No generation loop is needed. The whole sequence goes through one forward pass, because the causal mask makes this equivalent to step-by-step processing.
- One forward-mode derivative (`torch.func.jvp`) gives the exact infinitesimal response at every position at once.
- It works with *any* token sequence, including human-written text, which we need for the control experiments (Sec. 5.4).

```python
# sketch (fp32, attn_implementation="eager" so forward-mode AD works)
y   = greedy_generate(prompt_ids)                    # unperturbed tokens y_1..y_T
E   = embed(cat(prompt_ids, y))                      # (L_p+T, D)
U   = zeros_like(E); U[:L_p] = random_unit_direction  # perturb prompt rows only
# finite perturbation:
dh  = model(E + r*U).hidden[l] - model(E).hidden[l]   # δh_t for all t
# infinitesimal perturbation (exact linear response):
_, dh_lin = torch.func.jvp(lambda X: model(X).hidden[l], (E,), (U,))
# where the free run WOULD have flipped: argmax(logits_perturbed[t]) != y_t
```

### 3.2 Splitting the divergence into a smooth part and a discrete part

Free-running divergence has two parts:

- **Smooth part:** does δh_t grow or decay along the trajectory? Teacher forcing measures this.
- **Discrete part:** let m_t ≥ 0 be the gap between the top-1 and top-2 logits of the unperturbed run, and δm_t the change of this gap caused by the perturbation. The free run flips at the first t where |δm_t| > m_t.

Write δm_t ≈ r·g(t) and assume small margins occur with finite probability density p(0). Then the flip probability per step is ≈ p(0)·r·g(t). Setting the accumulated hazard to ~1 gives the mean first-divergence time ⟨T⟩:

| Growth of the smooth part g(t) | Mean first-divergence time ⟨T⟩ | Interpretation |
|---|---|---|
| e^{λt} (chaos) | ≈ λ⁻¹ ln(1/r) | **logarithmic** in r: the textbook predictability time |
| constant | ∝ 1/r | no amplification; pure threshold crossing |
| t^{β} (power-law growth) | ∝ r^{−1/(1+β)} | sub-exponential amplification |
| t^{−γ}, γ < 1 (slow decay) | ∝ r^{−1/(1−γ)} | contracting smooth part |
| t^{−γ}, γ ≥ 1 | a finite fraction **never** diverges | matches our "stable" runs |

So **the scaling of ⟨T⟩ with r distinguishes chaos from a threshold effect**. Logarithmic scaling indicates chaos; power-law scaling indicates no exponential growth. Measuring it requires r over at least 3 decades, hence FP32.

### 3.3 Why component Jacobians are not enough, and finite-time growth without chaos

- The Lyapunov exponent is a property of the **product** of Jacobians along the trajectory: λ = lim (1/n) ln ‖J_n ⋯ J_1 u‖. Each factor can have norm > 1 while the product still shrinks, because the expanding directions of one factor may be contracted by the next (e.g. by the RMSNorm projection). This is R2's point.
- **Non-normal transient growth.** A linear system whose eigenvalues are all stable can still amplify perturbations strongly for a finite time if its matrix is non-normal (eigenvectors not orthogonal). This is the classic explanation of transition to turbulence without linear instability [17, 18]. It is the standard example of "finite-time expansion without chaos" (R1.2). To quantify non-normality, compare the spectral norm ‖J‖₂ with the spectral radius ρ(J).
- **Our Fig. 7 already measures the full product** along depth for a finite perturbation, and it shows a plateau and then contraction. It should be presented as the answer to R1.1/R2.2: the components expand, but the composition is close to marginal.
- **Finite-time and non-autonomous systems.** Each layer has different weights, and the generation is finite. So only *finite-time* exponents are meaningful. **Expansion entropy** [16] is a definition of chaos that also covers non-periodically forced (non-autonomous) systems and is defined on a restricted region of state space. It is computed from Jacobians, which we already have.

### 3.4 Stable chaos and flux tubes

In networks of spiking neurons, the dynamics can be exponentially stable to infinitesimal perturbations (all Lyapunov exponents negative) and still be unpredictable. A perturbation above a critical size changes the order of spikes, which is a discrete event, and the trajectories then separate. Names for this:

- "stable chaos" [1, 2]
- "dynamic flux tubes" [3]: each trajectory is surrounded by a tube of radius ε*. Perturbations inside the tube decay; perturbations beyond it diverge.

Tokens play the role of spikes. The "latent-then-jump" pattern we observe (distance shrinks, then a token flips, then fast separation) is what this picture predicts. The flux-tube radius, i.e. the perturbation size needed to flip a token, is a new measurable quantity for LLMs.

### 3.5 Contracting maps with switching become periodic

A map that is contracting on each piece, with discontinuities between pieces, is called a **piecewise contraction**. Rigorous results show that such maps are typically **asymptotically periodic**, but only under specific assumptions:

- Bruin & Deane [4]: piecewise affine contractions of the plane, for almost every choice of the fixed points. They also note that not every piecewise contraction is asymptotically periodic.
- Nogueira & Pires [5a] and Gaivão & Nogueira [5b]: piecewise contractions of an interval, with at most as many periodic orbits as pieces.
- Cessac [6]: a discrete-time spiking network model whose dynamics are generically periodic. Cessac notes that, because of the sharp firing threshold, the dynamics can still look chaotic in numerical experiments.
- In stable chaos [1], the irregular transient before settling can grow exponentially with system size.

None of these theorems covers an LLM directly, so they serve as guidance, not proof.

This gives a falsifiable prediction: if the smooth part of a greedy LLM is contracting, greedy decoding should eventually fall into cycles, possibly after long irregular transients. That is the well-known repetition degeneration of greedy decoding [7], and it fits the attractor cycles of Wang et al. that we already cite. Measuring transient lengths and cycle statistics is therefore informative.

### 3.6 Matched controls

A control system only proves something if it keeps the **same discrete argmax step** while its smooth part is **provably non-expanding**. The reviewers' alternative explanation is "the threshold alone causes divergence", so the control must keep the threshold and remove the expansion. It should be paired with a positive (chaotic) control run through the same pipeline:

|  | Smooth part contracting | Smooth part chaotic |
|---|---|---|
| Greedy argmax on top | negative control | positive control |
| Teacher-forced (no argmax) | perturbation must decay | must grow at the known λ |

If the negative control also shows latent-then-jump divergence, Lorenz-like RPs and low D₂, then those features are not evidence for chaos. We would state this openly and rely on the diagnostics that do discriminate. Either outcome strengthens the paper.

---

## 4. Proposed framing (decided after the experiments of Sec. 5.1)

- **Outcome A (most likely).** δh_t does not grow exponentially under teacher forcing, and ⟨T⟩ follows a power law in r. Framing: *"sensitivity without exponential instability"*. Greedy LLM generation is a piecewise-smooth system whose continuous part is close to marginal stability ("edge of chaos" in the strict sense), and whose unpredictability comes from finite perturbations crossing token decision boundaries, as in stable chaos.
  - This answers R2.4 (what is new): a quantitative separation of the continuous and discrete contributions, a scaling law for the predictability time, a flux-tube radius, and a link to repetition loops.
  - Possible title: *"Sensitivity without instability: decision-boundary-driven divergence in LLM generation"*.
- **Outcome B.** δh_t grows exponentially under teacher forcing and ⟨T⟩ ∝ ln(1/r). Then we can claim chaos in the continuous dynamics with much stronger evidence, and keep the current framing in a more careful form.

In both cases, either keep "chain-of-thought" in the title and support it with experiments (Sec. 5.5), or drop it.

---

## 5. Experiments

### 5.1 Decisive experiments (smooth vs. discrete)

- **A1. Teacher-forced linear response** (Sec. 3.1).
  - Measure δh_t and δlogits_t along the trajectory for prompt perturbations, and for perturbations of the current state (residual stream at step s).
  - One backward pass (VJP) from position t gives ∂h_t/∂h_s for all s < t, i.e. the two-time response R(t,s) = ‖∂h_t/∂h_s‖.
  - Plot R against t − s: exponential growth means chaos in the smooth part; decay or power law means not. **This one figure addresses both reviewers.**
- **A2. ⟨T⟩ vs. r** (Sec. 3.2).
  - Run in FP32 with r over ≥ 3 decades; logarithmic vs. power-law scaling discriminates.
  - Also measure the distribution of margins m_t near zero, since the prediction depends on it.
- **A3. Margins and survival analysis.**
  - Show that flips occur at small-margin "forking tokens" [10, 11].
  - Predict the distribution of first-divergence times from the measured m_t and δm_t (from A1), and compare with the free-running data.
  - Use Kaplan–Meier survival curves: runs that never diverge are treated as censored, not discarded.
- **A4. Finite-size Lyapunov exponent (FSLE)** [12] / scale-dependent Lyapunov exponent [13].
  - λ(δ) = ln ρ / ⟨τ_ρ(δ)⟩, where τ_ρ(δ) is the time for a separation of size δ to grow by a factor ρ.
  - It is designed for systems where separation saturates and several scales are involved. For chaos, λ(δ) has a plateau at small δ; for noise it behaves differently.
  - The flux-tube picture predicts λ(δ) < 0 below a critical δ* and > 0 above it. δ* is the flux-tube radius.
- **A5. Branching entropy of the perturbed ensemble** ((ε,τ)-entropy [14]).
  - Count the number N(t, r) of distinct token sequences produced by an ensemble of perturbations of size r.
  - For chaos, h = (1/t) ln N stays finite as r → 0. For stable chaos it vanishes below δ*.
- **A6. Forced single flip.**
  - Replace the token at step s by the second-best token, then continue greedily, and check whether the hidden-state and semantic distances heal (shrink back) or keep growing.
  - Contracting systems heal; chaotic systems spread.

### 5.2 Proper Lyapunov quantities (R1.1, R2.2)

- **Full tangent map** instead of per-component norms. Propagate k tangent vectors through complete layers (and, under teacher forcing, through generation steps), re-orthonormalizing with QR (the Benettin method [15]). This gives a finite-time Lyapunov spectrum.
- **Expansion entropy** [16] for the depth map (Sec. 3.3).
- **Non-normality:** report ‖J‖₂ against ρ(J) per block (Sec. 3.3).
- **Extend our mean-field formula** to the full residual block, including the RMSNorm projection, to get a gain factor χ₁ in the sense of [19]. Then compare the analytic prediction λ_depth ≈ ½ ln χ₁ with the measured value. Ref. [20] treats signal propagation in Transformers.
- **Delay systems.** Our "delay system" interpretation (Sec. 6 of the paper) has a classic reference for Lyapunov spectra of infinite-dimensional delay systems [21].

### 5.3 Control systems

1. **Mamba (negative control, pretrained, no training needed)** [37].
   - Per layer, the state update (as implemented in the reference code) is h_t = exp(Δ_t A) h_{t−1} + Δ_t B_t x_t, with A = −exp(A_log) < 0 and Δ_t = softplus(·) > 0. Every diagonal entry of ∂h_t/∂h_{t−1} therefore lies in (0, 1).
   - The short convolution state is a shift register, so its Jacobian is nilpotent.
   - Under teacher forcing, the one-step Jacobian is **block lower-triangular** in the layer index: a layer's state depends on lower layers at the same step, never on higher layers. The Lyapunov exponents are therefore those of the diagonal blocks, and all are negative, so the smooth dynamics provably cannot be chaotic.
   - Caveats:
     - Some exponents may be close to 0 (long-memory channels).
     - The off-diagonal coupling makes the Jacobian non-normal, so *transient growth without chaos* is possible. If we observe it, it is a direct concrete answer to R1.2.
   - Mamba (130M–2.8B) and Pythia [38] (160M–2.8B) were both trained on the Pile. This gives an attention vs. no-attention comparison with the same data, which also tests our claim that attention carries the perturbation.
   - Note the structural difference: in Mamba the prompt perturbation decays in the state, while in a Transformer it stays in the KV cache.
   - Use the pure-PyTorch path of HF `MambaForCausalLM`, which supports autograd, in FP32. Sizes up to ~1.4B fit on a T4.
2. **Random-initialized Transformer, ordered vs. chaotic phase.**
   - Mean-field theory predicts an order-to-chaos transition as a function of the weight scale σ_w [19, 20]. Small σ_w gives a known contracting regime and large σ_w an expanding one.
   - This calibrates the Jacobian and layer-wise analysis. The Pythia step-0 checkpoint is a free real-size example.
   - The generated text is meaningless, so this calibrates the depth ("vertical") analysis better than the generation ("horizontal") analysis.
3. **Toy Transformer trained on a system with a known Lyapunov exponent (positive + negative control).**
   - Train a ~10M-parameter Transformer on tokenized logistic-map orbits x_{t+1} = a x_t(1 − x_t): a = 4 is chaotic with λ = ln 2, a = 3.2 has an attracting 2-cycle.
   - Greedy generation then iterates the learned map. The pipeline must recover λ ≈ ln 2 in one case and healing/reconvergence in the other. This is the only option where the true exponent is known. Training takes hours on a T4.
4. **Provably contractive Transformer** (most expensive, optional).
   - Train a small model on TinyStories [41] with L2 attention [39], spectral normalization [40] and damped residuals x ← (1−α)x + αF(x) with Lip(F) < 1.
   - Plain residuals are not enough, since ‖I + J‖ can reach 1 + ‖J‖.
   - The result is a globally contracting map that still produces language. Only worth doing if the Mamba results are ambiguous.

**Not useful:** n-gram models (no continuous state, so the comparison is trivial), and isolated blocks such as RMSNorm alone (they do not generate trajectories).

Run the **same tests** on all systems: A1 (R(t,s)), A2 (⟨T⟩ vs. r), A6 (forced flip), RQA/K₂/D₂ (Sec. 5.4), and transient length and cycle statistics (Sec. 3.5).

### 5.4 Recurrence plots and correlation dimension: specificity and robustness (R2.3, R1.3, R1.4)

**Calibration panel.**
- Run Lorenz, a noisy limit cycle, AR(1) / Ornstein–Uhlenbeck, fractional Brownian motion and a high-dimensional random walk.
- Embed each in 1536 dimensions with a random projection plus noise, and use the same N and the same pipeline.
- Show which processes our diagnostics separate.

**LLM null models (via teacher forcing).**
- Compare hidden states of: human-written text of the same genre, sentence-shuffled text, token-shuffled text, and text from a random-weight model.
- If human text gives the same RPs and D₂ as LLM-generated text, then these diagnostics measure the *structure of text*, not the generation dynamics. We need to know this before the reviewers ask.

**Surrogate data.** Surrogates are synthetic series that keep selected linear properties (spectrum, amplitude distribution, cross-correlations) but are otherwise random. A statistic that does not differ from its surrogates is not evidence of nonlinear determinism.
- IAAFT surrogates [28] on PCA components.
- Multivariate surrogates [27], which keep cross-correlations.
- Twin surrogates for recurrence statistics [26].

**Quantitative RPs.**
- Restore the RQA table: DET, LAM, L_max and ENT [47] as functions of RR ∈ [0.01, 0.1], with surrogate bands (R1.3).
- Estimate the correlation entropy K₂ from the diagonal-line length distribution, P(l) ∝ e^{−K₂ l} [25]. K₂ is a lower bound on the sum of positive Lyapunov exponents. A plateau in ε indicates deterministic chaos; growth as ε → 0 indicates noise.

**Ordinal methods.**
- Permutation entropy [30] and the complexity–entropy plane [29], designed to separate chaos from coloured noise.
- Apply them to scalar series: the logit margin m_t, PCA components and distance series.

**Forecasting test.** Nonlinear prediction skill as a function of horizon [31]: for chaos, skill decays with horizon; for noise, it is flat.

**Correlation dimension, done carefully (R1.4):**
- D₂ against N, by subsampling. D₂ must converge.
- Theiler window [22]: exclude pairs of points that are close in time. Temporal correlation produces spuriously low D₂, and semantic correlation in text is exactly this kind of correlation.
- Local slope d ln C / d ln ε plotted against ε, with a visible plateau.
- Eckmann–Ruelle bound [23]: reliable D₂ estimates need D₂ ≲ 2 log₁₀ N. For N = 1000 this means D₂ ≲ 6.
- Coloured (power-law spectrum) noise gives a finite D₂ [24], so it must be ruled out with the calibration panel and surrogates.
- Cross-check with other estimators: TwoNN [32], GRIDE (scale-dependent) [33], and local dimension from extreme value theory [34], which suits short, high-dimensional data. A comparative review appeared in *Chaos* [35].
- Use the term "effective dimension" instead of "fractal" unless scaling is shown over a wide range. Compare with known intrinsic-dimension profiles of Transformer layers [36].

### 5.5 Scope, chain-of-thought and practical value (R1.5, R1.6, R2.4, R2.5)

**Models:**
- Qwen2.5-Math-1.5B, the base model of R1-Distill-Qwen-1.5B [42]: a clean reasoning vs. non-reasoning pair.
- Qwen3-1.7B with thinking mode on and off [43].
- Llama-3.2-1B/3B, Gemma-2-2B, Phi-4-mini (move from the thesis to the paper).
- R1-Distill-Qwen-7B for scale.
- Pythia vs. Mamba (Sec. 5.3), with Pythia training checkpoints to see when the behaviour appears during training.

**Prompts:** a few hundred, from GSM8K / MATH-500 plus open-ended tasks, with confidence intervals.

**Chain-of-thought:**
- Check whether divergence onsets cluster at reasoning-branching tokens ("Wait", "Alternatively", "However"); these are known high-entropy forking tokens in reasoning models [11].
- "Divergent paths, convergent answers": measure answer agreement across the perturbation ensemble against trajectory divergence. The answer acting as an attractor in answer space would be a meaningful dynamical picture of "how CoT LLMs look for an answer".
- If we do not do this, we drop CoT from the title.

**Practical value (R1.6):**
- **Reproducibility horizon.** Batch and kernel nondeterminism in GPU inference [48] acts as a natural perturbation. If it falls on our ⟨T⟩(r) curve, the curve predicts how many tokens a greedy generation stays reproducible.
- **Uncertainty estimation.** Use the sensitivity of the final answer to tiny perturbations to predict correctness. Report AUROC against self-consistency [45] and semantic entropy [44].
- **Attention window** as a design parameter that controls stability (Fig. 9).

---

## 6. Reviewer comments → actions

| Comment | Action | Section |
|---|---|---|
| R1.1, R2.2: component norms ≠ Lyapunov exponent of the full map | Full tangent map with QR, finite-time spectrum, expansion entropy; present Fig. 7 as "components expand, composition ≈ marginal" | 3.3, 5.2 |
| R1.2, R2.1: finite-time growth / token threshold vs. chaos | Teacher forcing, ⟨T⟩(r) scaling, margins, FSLE, forced flip, Mamba control; reframing | 3.1–3.4, 5.1, 5.3 |
| R1.3: RR choice | RQA against RR ∈ [0.01, 0.1] with surrogate bands | 5.4 |
| R1.4: D₂ convergence | D₂ vs. N, Theiler window, local slopes, Eckmann–Ruelle bound, other estimators | 5.4 |
| R2.3: diagnostics not specific to chaos | Calibration panel, LLM null models, surrogates, K₂, ordinal methods | 5.4 |
| R1.5: scope | More models and prompts; claims restricted to what is tested | 5.5 |
| R1.6, R2.4: what is new / what does it predict | New framing and contribution list; predictability scaling law; reproducibility horizon; uncertainty estimation | 4, 5.5 |
| R2.5: CoT link | Reasoning vs. non-reasoning models, forking tokens, answer convergence; or change the title | 5.5 |

---

## 7. Priorities (given a T4 GPU)

**Must do:**
- FP32 reruns and the perturbation-rounding check (Sec. 2).
- A1 (teacher-forced response) and A2 (⟨T⟩ vs. r).
- RR and D₂ robustness.
- Human-text and shuffled-text null models.
- Rewriting the claims.

**Should do:**
- A3, A4, A6.
- Mamba vs. Pythia, and the logistic-map toy model.
- The calibration panel.
- R1-Distill vs. Qwen2.5-Math, 3–4 more models, a prompt dataset.

**Optional:**
- Random-init phase scan, the contractive Transformer.
- Expansion entropy, full Lyapunov spectra, the AUROC experiment.

---

## 8. Response letter

- **Structure:**
  - A short thank-you, then a summary of the major changes (new experiments, reframing, new title if any).
  - Then point by point: quote each comment, give our response, and state where the manuscript changed (section, figure, line).
- **Tone:** agree explicitly where the reviewers are right ("We agree that ... We have therefore ..."). Both called the topic fascinating, so an open and non-defensive answer works best.
- **Submission files:**
  - A latexdiff-marked PDF alongside the clean version.
  - Supplementary material with the extra models and controls.
  - Public code and data.
- **Deadline:** check the deadline in the decision letter, and ask the editor for an extension early if the new experiments need it.

---

## 9. Software

| Task | Tools |
|---|---|
| Jacobians, JVP/VJP, teacher forcing | PyTorch `torch.func` (already used); HF `attn_implementation="eager"` |
| Lyapunov spectra, expansion entropy, D₂ (with Theiler window), Takens estimator, EVT dimensions, RQA | DynamicalSystems.jl [S1] (ChaosTools, FractalDimensions, RecurrenceAnalysis) |
| RQA, recurrence networks, twin surrogates | pyunicorn [S2], PyRQA [S3] (fast, GPU) |
| Permutation entropy, complexity–entropy plane | ordpy [S4] |
| Intrinsic dimension (TwoNN, GRIDE, MLE) | scikit-dimension [S5], DADApy [S6] |
| Classic surrogates, Kantz Lyapunov estimator [46] | TISEAN [S7] |
| Survival curves (Kaplan–Meier) | `lifelines` (Python) |

---

## References

Bibliographic details were checked against Crossref, arXiv and publisher pages (November 2026). Content summaries of papers we have not read in full should be checked before citing them in the manuscript.

**Stable chaos, flux tubes, piecewise contractions**
1. A. Politi, R. Livi, G.-L. Oppo, R. Kapral, "Unpredictable behaviour in stable systems," *Europhys. Lett.* **22**, 571 (1993). doi:10.1209/0295-5075/22/8/003
2. S. Jahnke, R.-M. Memmesheimer, M. Timme, "Stable irregular dynamics in complex neural networks," *Phys. Rev. Lett.* **100**, 048102 (2008). doi:10.1103/PhysRevLett.100.048102
3. M. Monteforte, F. Wolf, "Dynamic flux tubes form reservoirs of stability in neuronal circuits," *Phys. Rev. X* **2**, 041007 (2012). doi:10.1103/PhysRevX.2.041007
4. H. Bruin, J. H. B. Deane, "Piecewise contractions are asymptotically periodic," *Proc. Amer. Math. Soc.* **137**, 1389 (2009). doi:10.1090/S0002-9939-08-09633-0

5. Piecewise contractions of an interval:
   - 5a. A. Nogueira, B. Pires, "Dynamics of piecewise contractions of the interval," *Ergodic Theory Dynam. Systems* **35**, 2198 (2015). doi:10.1017/etds.2014.16
   - 5b. J. P. Gaivão, A. Nogueira, "Dynamics of piecewise increasing contractions," *Bull. London Math. Soc.* **54**, 482 (2022). doi:10.1112/blms.12577
6. B. Cessac, "A discrete time neural network model with spiking neurons. Rigorous results on the spontaneous dynamics," *J. Math. Biol.* **56**, 311 (2008). doi:10.1007/s00285-007-0117-3
7. A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi, "The curious case of neural text degeneration," ICLR 2020. arXiv:1904.09751

**Teacher forcing, drive–response, forking tokens**

8. R. J. Williams, D. Zipser, "A learning algorithm for continually running fully recurrent neural networks," *Neural Comput.* **1**, 270 (1989). doi:10.1162/neco.1989.1.2.270
9. L. M. Pecora, T. L. Carroll, "Synchronization in chaotic systems," *Phys. Rev. Lett.* **64**, 821 (1990). doi:10.1103/PhysRevLett.64.821
10. E. Bigelow, A. Holtzman, H. Tanaka, T. Ullman, "Forking paths in neural text generation," ICLR 2025. arXiv:2412.07961
11. S. Wang et al., "Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning," NeurIPS 2025. arXiv:2506.01939

**Lyapunov exponents and stability**

12. E. Aurell, G. Boffetta, A. Crisanti, G. Paladin, A. Vulpiani, "Predictability in the large: an extension of the concept of Lyapunov exponent," *J. Phys. A* **30**, 1 (1997). doi:10.1088/0305-4470/30/1/003
13. J. B. Gao, J. Hu, W. W. Tung, Y. H. Cao, "Distinguishing chaos from noise by scale-dependent Lyapunov exponent," *Phys. Rev. E* **74**, 066204 (2006). doi:10.1103/PhysRevE.74.066204
14. P. Gaspard, X.-J. Wang, "Noise, chaos, and (ε,τ)-entropy per unit time," *Phys. Rep.* **235**, 291 (1993). doi:10.1016/0370-1573(93)90012-3
15. G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn, "Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory," *Meccanica* **15**, 9 (1980). doi:10.1007/BF02128236
16. B. R. Hunt, E. Ott, "Defining chaos," *Chaos* **25**, 097618 (2015). doi:10.1063/1.4922973
17. L. N. Trefethen, A. E. Trefethen, S. C. Reddy, T. A. Driscoll, "Hydrodynamic stability without eigenvalues," *Science* **261**, 578 (1993). doi:10.1126/science.261.5121.578
18. B. F. Farrell, P. J. Ioannou, "Generalized stability theory. Part I: Autonomous operators," *J. Atmos. Sci.* **53**, 2025 (1996). doi:10.1175/1520-0469(1996)053<2025:GSTPIA>2.0.CO;2
19. B. Poole, S. Lahiri, M. Raghu, J. Sohl-Dickstein, S. Ganguli, "Exponential expressivity in deep neural networks through transient chaos," NeurIPS 2016. arXiv:1606.05340
20. L. Noci, S. Anagnostidis, L. Biggio, A. Orvieto, S. P. Singh, A. Lucchi, "Signal propagation in transformers: Theoretical perspectives and the role of rank collapse," NeurIPS 2022. arXiv:2206.03126
21. J. D. Farmer, "Chaotic attractors of an infinite-dimensional dynamical system," *Physica D* **4**, 366 (1982). doi:10.1016/0167-2789(82)90042-2

**Dimension estimation**

22. J. Theiler, "Spurious dimension from correlation algorithms applied to limited time-series data," *Phys. Rev. A* **34**, 2427 (1986). doi:10.1103/PhysRevA.34.2427
23. J.-P. Eckmann, D. Ruelle, "Fundamental limitations for estimating dimensions and Lyapunov exponents in dynamical systems," *Physica D* **56**, 185 (1992). doi:10.1016/0167-2789(92)90023-G
24. A. R. Osborne, A. Provenzale, "Finite correlation dimension for stochastic systems with power-law spectra," *Physica D* **35**, 357 (1989). doi:10.1016/0167-2789(89)90075-4

**Recurrence analysis and surrogates**

25. M. Thiel, M. C. Romano, P. L. Read, J. Kurths, "Estimation of dynamical invariants without embedding by recurrence plots," *Chaos* **14**, 234 (2004). doi:10.1063/1.1667633
26. M. Thiel, M. C. Romano, J. Kurths, M. Rolfs, R. Kliegl, "Twin surrogates to test for complex synchronisation," *Europhys. Lett.* **75**, 535 (2006). doi:10.1209/epl/i2006-10147-0
27. D. Prichard, J. Theiler, "Generating surrogate data for time series with several simultaneously measured variables," *Phys. Rev. Lett.* **73**, 951 (1994). doi:10.1103/PhysRevLett.73.951
28. T. Schreiber, A. Schmitz, "Improved surrogate data for nonlinearity tests," *Phys. Rev. Lett.* **77**, 635 (1996). doi:10.1103/PhysRevLett.77.635

**Distinguishing chaos from noise**

29. O. A. Rosso, H. A. Larrondo, M. T. Martin, A. Plastino, M. A. Fuentes, "Distinguishing noise from chaos," *Phys. Rev. Lett.* **99**, 154102 (2007). doi:10.1103/PhysRevLett.99.154102
30. C. Bandt, B. Pompe, "Permutation entropy: a natural complexity measure for time series," *Phys. Rev. Lett.* **88**, 174102 (2002). doi:10.1103/PhysRevLett.88.174102
31. G. Sugihara, R. M. May, "Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series," *Nature* **344**, 734 (1990). doi:10.1038/344734a0

**Intrinsic and local dimension**

32. E. Facco, M. d'Errico, A. Rodriguez, A. Laio, "Estimating the intrinsic dimension of datasets by a minimal neighborhood information," *Sci. Rep.* **7**, 12140 (2017). doi:10.1038/s41598-017-11873-y
33. F. Denti, D. Doimo, A. Laio, A. Mira, "The generalized ratios intrinsic dimension estimator," *Sci. Rep.* **12**, 20005 (2022). doi:10.1038/s41598-022-20991-1

34. Local dimension from extreme value theory:
   - D. Faranda, G. Messori, P. Yiou, "Dynamical proxies of North Atlantic predictability and extremes," *Sci. Rep.* **7**, 41278 (2017). doi:10.1038/srep41278
   - V. Lucarini et al., *Extremes and Recurrence in Dynamical Systems* (Wiley, 2016). doi:10.1002/9781118632321
35. G. Datseris, I. Kottlarz, A. P. Braun, U. Parlitz, "Estimating fractal dimensions: A comparative review and open source implementations," *Chaos* **33**, 102101 (2023). doi:10.1063/5.0160394
36. L. Valeriani et al., "The geometry of hidden representations of large transformer models," NeurIPS 2023. arXiv:2302.00294

**Models, datasets and control architectures**

37. A. Gu, T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," COLM 2024. arXiv:2312.00752
38. S. Biderman et al., "Pythia: A suite for analyzing large language models across training and scaling," ICML 2023, PMLR 202:2397. arXiv:2304.01373
39. H. Kim, G. Papamakarios, A. Mnih, "The Lipschitz constant of self-attention," ICML 2021, PMLR 139:5562. arXiv:2006.04710
40. T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida, "Spectral normalization for generative adversarial networks," ICLR 2018. arXiv:1802.05957
41. R. Eldan, Y. Li, "TinyStories: How small can language models be and still speak coherent English?" arXiv:2305.07759 (2023)
42. DeepSeek-AI, "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning," *Nature* **645**, 633 (2025). arXiv:2501.12948. R1-Distill-Qwen-1.5B is distilled into the base model Qwen2.5-Math-1.5B.
43. Qwen Team, "Qwen3 technical report," arXiv:2505.09388 (2025). Thinking mode is switched with `enable_thinking`.

**Uncertainty and self-consistency**

44. S. Farquhar, J. Kossen, L. Kuhn, Y. Gal, "Detecting hallucinations in large language models using semantic entropy," *Nature* **630**, 625 (2024). doi:10.1038/s41586-024-07421-0
45. X. Wang et al., "Self-consistency improves chain of thought reasoning in language models," ICLR 2023. arXiv:2203.11171

**Other**

46. H. Kantz, "A robust method to estimate the maximal Lyapunov exponent of a time series," *Phys. Lett. A* **185**, 77 (1994). doi:10.1016/0375-9601(94)90991-1
47. N. Marwan, M. C. Romano, M. Thiel, J. Kurths, "Recurrence plots for the analysis of complex systems," *Phys. Rep.* **438**, 237 (2007). Already cited in the manuscript.
48. GPU inference nondeterminism: He et al. (2025), already cited in the manuscript as `he2025nondeterminism`.

**Software**

- S1. G. Datseris, "DynamicalSystems.jl: A Julia software library for chaos and nonlinear dynamics," *J. Open Source Softw.* **3**, 598 (2018). doi:10.21105/joss.00598
- S2. J. F. Donges et al., "Unified functional network and nonlinear time series analysis for complex systems science: The pyunicorn package," *Chaos* **25**, 113101 (2015). doi:10.1063/1.4934554
- S3. T. Rawald, M. Sips, N. Marwan, "PyRQA — Conducting recurrence quantification analysis on very long time series efficiently," *Comput. Geosci.* **104**, 101 (2017). doi:10.1016/j.cageo.2016.11.016
- S4. A. A. B. Pessa, H. V. Ribeiro, "ordpy: A Python package for data analysis with permutation entropy and ordinal network methods," *Chaos* **31**, 063110 (2021). doi:10.1063/5.0049901
- S5. J. Bac et al., "Scikit-Dimension: A Python package for intrinsic dimension estimation," *Entropy* **23**, 1368 (2021). doi:10.3390/e23101368
- S6. A. Glielmo et al., "DADApy: Distance-based analysis of data-manifolds in Python," *Patterns* **3**, 100589 (2022). doi:10.1016/j.patter.2022.100589
- S7. R. Hegger, H. Kantz, T. Schreiber, "Practical implementation of nonlinear time series methods: The TISEAN package," *Chaos* **9**, 413 (1999). doi:10.1063/1.166424
