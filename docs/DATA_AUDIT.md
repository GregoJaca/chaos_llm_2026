# Data audit and baseline protocol

Status: verified on 2026-11-17 on `x11` (Tesla T4, 15.6 GB; torch 2.5.1+cu121; transformers 4.50.0).
Every statement below is reproducible from the listed path/command unless marked **UNVERIFIED**.
This note is about *measurement validity*, not about the scientific interpretation of the results.

---

## 0. Models and setting

- `DeepSeek-R1-Distill-Qwen-1.5B`: Qwen2, 28 layers, `d=1536`, vocab 151936, 12 heads / 2 KV heads.
- `Phi-4-mini-instruct`: 32 layers, `d=3072`, vocab 200064, 24 heads / 8 KV heads.
- Prompt in all token-level runs: `interstellar_travel` (`prompts.txt`), prompt length 65 (DeepSeek) / 64 (Phi).
- `outputs_microsoft*` = Phi; `outputs_0.0001`, `outputs_0.0004`, `outputs_0.0006`, `outputs_OLD*` = DeepSeek.
- The token-level pipeline is `chaos_llm_2026/src/chaos_llm/run_experiment.py`. It generates the baseline with **batch size 1** and the perturbed runs with **batch size N** (N=16..64). The baseline is not a row in the perturbed batch.

---

## 1. Verified finding: the `r = 0` control fails for Phi in bfloat16

**Data.** `chaos_llm_2026/outputs_microsoft/run_*_0_interstellar_travel_seed_69`.
Config: Phi-4-mini, `bfloat16`, `num_conditions=100`, `batch_size=26`, magnitude list `[5e-06, 0]`. These are the zero-perturbation runs.

**What the saved files show (zero perturbation).**

| window | baseline len | perturbed rows | first token mismatch | count |
|---:|---:|---:|---:|---:|
| 10 | 1064 | 100 × 1064 | generation step **20** | 100/100 |
| 25 | 1064 | 100 × 1064 | step **11** | 100/100 |
| 50 | 1064 | 100 × 1064 | step **2** | 100/100 |
| 65 | 1064 | 100 × 1064 | step **1** | 100/100 |
| 80 | 1064 | 100 × 1064 | step **1** | 100/100 |
| 150 | 637 | 100 × 1064 | step **1** | 100/100 |
| 1000 | 1064 | 100 × 1064 | step **1** | 100/100 |

- The prompt token hashes are identical between baseline and every row.
- These are **different token IDs**, and the decoded text differs materially:
  - window 1000, around the mismatch: baseline `...The review should be structured as a scientific paper, with an abs...` vs perturbed `...The response should be structured as a detailed report, with each...`.
  - window 10: baseline degenerates into `positively positively...` while perturbed degenerates into `531531531...`.
- So this is a hard token-level divergence, not just hidden-state numerical noise.
- Window 150 additionally has a baseline/perturbed length mismatch (637 vs 1064), so it is doubly unusable.

**The `r=5e-6` “perturbation” is the same as `r=0`.** At every window with both an `r=0` and an `r=5e-6` run, compare the per-direction first-flip time `T`:

| window | identical `T` per direction | `r=0` modal `T` | `r=5e-6` modal `T` |
|---:|---:|---:|---:|
| 10 | 100/100 | 20 | 20 |
| 25 | 99/100 | 11 | 11 (one at 12) |
| 50 | 100/100 | 2 | 2 |
| 65 | 100/100 | 1 | 1 |
| 80 | 100/100 | 1 | 1 |
| 150 | 100/100 | 1 | 1 |
| 1000 | 100/100 | 1 | 1 |

So the “small perturbation” result is indistinguishable from the zero-perturbation numerical result. This is the strongest single demonstration that the Phi bf16 divergence is numerical, not dynamical.

**Direct reproduction (same prompt, same window, no perturbation).**

| model | dtype | attention | batch 1 vs batch 26 |
|---|---|---|---|
| Phi-4-mini | bfloat16 | sdpa | first mismatch at step **20** (matches the saved run) |
| Phi-4-mini | bfloat16 | eager | first mismatch at step **7** |
| Phi-4-mini | float16 | sdpa | **identical** |
| Phi-4-mini | float16 | eager | **identical** |
| DeepSeek-1.5B | bfloat16 | sdpa | identical (window 100, 120 tokens) |
| DeepSeek-1.5B | float16 | sdpa | identical (window 100/200, 120–250 tokens) |

- Within the Phi bf16 batch, all 26 rows are identical to row 0. The difference is **batch shape**, not row randomness.
- The effect is model/dtype-specific: DeepSeek bf16 is stable at window 100 for 120 tokens, so this is **not** a universal “batch size is broken” claim.

**Consequence.** Every Phi bf16 token-level divergence and attention-window result in `outputs_microsoft*` is contaminated by a numerical divergence that happens with **zero perturbation**. It cannot be used as evidence of perturbation sensitivity. The same class of failure must be checked per model/dtype/window before any new run is trusted.

---

## 2. Verified finding: DeepSeek low-precision baselines are near-tie fragile

The FP16/bf16 baseline was recomputed in FP32 and compared token-by-token. “Disagreement” means the FP32 argmax differs from the saved baseline token.

| window | saved dtype | disagreements | first disagreement | margin there | smallest margin (step) |
|---:|---|---:|---:|---:|---:|
| 10 | bf16 | 1 | 1 | 0.1093 | 0.1093 (1) |
| 50 | fp16 | 0 | — | — | 0.1673 (13) |
| 100 | fp16 | **1** | **123** | **0.0201** | **0.0201 (123)** |
| 200 | fp16 | 0 | — | — | 0.0031 (44) |
| 500 | fp16 | 0 | — | — | 0.0031 (44) |
| 1000 | bf16 | **14** | **55** | 0.0089 | 0.0017 (647) |
| 4096 | fp16 | 0 | — | — | 0.0031 (44) |

`margin = top1 logit − top2 logit` at the same position.

**Window 100, the important case.**
- The saved FP16 baseline chose token `1128` at generation step 123. In FP32 the top-1 is `1817` (margin 0.0201); `1128` is the second-best.
- In the saved data, **84/100** perturbed rows first differ from the baseline at step **123**, for every `r` in `5e-5 … 8e-4`; **16/100 never** differ.
- Under FP32 teacher forcing (fixed baseline tokens), 10 directions × `r ∈ {5e-5, 2e-4, 8e-4}` do **not** flip the FP32 top-1 at step 123. The perturbation changes the margin by only ~1e-5.
- FP32 teacher-forced `||δh_t||` (last layer) grows from ~1e-4 to ~1e-3 over the first ~100 steps, then decays back to ~1e-4. No exponential growth over 1000 steps.
- So the “universal flip at 123” is a near-tie where the low-precision baseline picked the second-best token; it is not evidence that the perturbation amplified and crossed a margin.

**Window 200, a clean-baseline case.**
- Saved FP16 baseline matches FP32 exactly (0 disagreements).
- But a FP32 free-run with `r=2e-4` for three directions (including directions that flipped at 44/55 in the saved FP16 data) produced **zero** mismatches in 90 tokens.
- Teacher-forced `δm` at step 55 is ~1e-5, while the margin is 0.0088. The FP16 flips at 44/55 are precision-driven, not perturbation-driven.
- This test used three directions; it is **preliminary**, but it is a direct counterexample to “FP16 flips at r=2e-4 are real”.

**Consequence.** First-divergence times in low precision are not a clean dynamical observable. FP32 (or at minimum a verified-stable FP16 protocol) is required. The baseline must be generated in the same precision and batch structure as the perturbed rows, and `r=0` must be a row in the batch.

---

## 3. Verified finding: `T(r)` is a staircase, not a smooth scaling law

The DeepSeek FP16 data in `outputs_OLD` + `outputs_0.0004` reused the **same 100 perturbation directions** across `r = 5e-5, 2e-4, 4e-4, 6e-4, 8e-4` (`num_conditions=100`, `subspace_seed=69`). This permits a fixed-direction analysis.

| window | directions that never flip | modal first-flip step at smallest `r` | constant across `r` | variable across `r` |
|---:|---:|---|---:|---:|
| 50 | 100/100 | — | 0 | 0 |
| 100 | 0/100 (16 censored at smallest `r`) | **123** (84/100) | 40 | 60 |
| 200 | 8/100 | **55** (20), 44 (7), 216 (2); 63 censored | 0 | 92 |
| 500 | 1/100 | **220** (33), 55 (20), 165 (13), 44 (7); 20 censored | 0 | 99 |

“Constant” = same first-flip step for every `r`; “variable” = at least one `r` changes the step or censors the run.

**Interpretation.** For a fixed direction, the first flip occurs at the first step whose margin is smaller than the perturbation effect. The observed `T` takes a few discrete values (the near-tie locations), and `r` changes **whether** a given near-tie is crossed, not the location smoothly. Averaging `T` over directions therefore averages a staircase and is dominated by the smallest margin. This is exactly why previous aggregation attempts failed.

**Consequence.** Replace `⟨T⟩(r)` by:
- survival curves `P(T > t)` with censoring,
- the per-step flip probability / hazard,
- the distribution of the local threshold `r_c(t,u) = m_t / |δm_t(u)/r|`,
- 3 individual direction examples plus the aggregate distribution.

---

## 4. Preliminary finding: exact cycles exist, but are often degenerate

Cheap CPU check on saved greedy baselines:
- DeepSeek 2000 generated tokens: exact period **8** in the tail; 93 repeated 5-grams (max count 109).
- Phi 4000 generated tokens (bf16): one run has a run of **1260 identical tokens** and exact period 1; another has period 1 and longest identical run 37.

This is a preliminary result. It says cycle statistics are measurable and sometimes nontrivial; the long Phi loops are degenerate and may be precision/window artifacts. Do not build the paper on periodicity without a larger, precision-controlled sample.

---

## 5. Baseline protocol (recommended, validated)

**Rules.**
1. **FP32 for science.** DeepSeek 1.5B FP32 fits the T4 (~6 GB). Phi 3.8B FP32 does not; use FP16 with a batch-matched reference, and state the limitation.
2. **The unperturbed condition is a row in the same batch.** Row 0 = `r=0` reference; optionally row 1 = duplicate reference. Every batch has its own reference. Compare only within the batch.
3. **Same code path, dtype, window, attention implementation, and shape** for reference and perturbed rows.
4. **Fixed-length generation** (`eos_token_id=None`, `max_new_tokens` fixed) or explicit EOS handling; do not compare a baseline that stopped early with a perturbed run that did not.
5. **Save metadata**: model path/revision, dtype, attention implementation, window, batch size, seed, prompt, prompt length, torch/transformers/CUDA versions, GPU, git commit, timestamp.
6. **Save the perturbation recipe**: subspace indices, simplex vertices or direction vectors, magnitudes, and the row-to-(direction, magnitude) mapping.
7. **Verify before using**: duplicate `r=0` rows identical; solo baseline vs in-batch reference reported (do not assume they match); if the solo check fails, use the in-batch reference and record the discrepancy.
8. **Do not use bf16 for small-`r` science** unless the `r=0` control and a FP32 spot check pass on the exact model/window/prompt.

**Validated examples (2026-11-17).**

| model | dtype | window | batch | tokens | `r=0` identical | solo = batched | perturbations flipped? | time |
|---|---|---:|---:|---:|---|---|---|---:|
| DeepSeek-1.5B | fp32 | 100 | 8 | 120 | yes | yes | no (`r=1e-4`, `1e-3`) | 7.0 s |
| Phi-4-mini | fp16 | 10 | 4 | 120 | yes | yes | no (`r=5e-6`) | 5.8 s |

Standalone script: `~/LLM/chaos_llm_2026/_audit/baseline_protocol.py` (not yet integrated into the pipeline; it does not modify the existing code).
The script currently puts all rows in one batch; for large sweeps, chunk the rows and repeat the reference in every chunk.

---

## 6. What is contaminated, what survives

**Contaminated (do not use as evidence without re-running).**
- Phi bf16 token-level divergence and attention-window results (`outputs_microsoft*`).
- DeepSeek bf16 token-level results (`outputs_0.0001`, bf16 parts of `outputs_0.0006`).
- DeepSeek fp16 first-divergence times as a quantitative dynamical observable (near-tie/precision dominated).
- Any comparison of a batch-1 baseline with a batch-N perturbed run in bf16.
- Any `r` below the numerical noise floor of the dtype.

**Survives.**
- Layer/MLP/attention/normalization Jacobian analyses (exact derivatives; no generation loop). These should be recomputed in FP32 if quantitative claims are made.
- Recurrence plots and correlation dimension on a **single** trajectory (no baseline comparison), subject to convergence/surrogate checks.
- The qualitative “latent then jump” pattern: it is real, but the jump is a near-tie crossing and can be triggered by numerical noise.
- The existence of nontrivial margins and near-ties: a valid, measurable object.

**Needs re-run.**
- Attention-window dependence of divergence.
- `T` distributions and `T(r)` scaling.
- Any claim that a small designed perturbation causes divergence.
- Any claim tying divergence to chain-of-thought or reasoning.

---

## 7. Data-quality checklist (use for every new dataset)

- [ ] Reference (`r=0`) is a row in the same batch as the perturbed rows.
- [ ] Duplicate reference is bit-identical.
- [ ] Solo reference vs in-batch reference reported.
- [ ] Same dtype, attention implementation, window, and shape for all rows.
- [ ] FP32, or FP16 with a passing `r=0` control and a FP32 spot check.
- [ ] Fixed output length or explicit EOS handling.
- [ ] Same directions across `r` (if doing fixed-direction analysis).
- [ ] Seeds, subspace indices, and direction vectors saved.
- [ ] Model revision, torch/transformers/CUDA versions, GPU, git commit saved.
- [ ] Token sequences and hidden states aligned (same rows, same lengths).
- [ ] For hidden states: layer indices, shapes, dtype, and whether teacher-forced or free-running.

---

## 8. Exact reproduction commands

All commands run from `~/LLM/chaos_llm_2026`.

```bash
# 1. Phi bf16 zero-perturbation control (saved data)
python3 - <<'PY'
import numpy as np, glob, os
for d in sorted(glob.glob('outputs_microsoft/run_*_0_interstellar_travel_seed_69')):
    z=np.load(os.path.join(d,'tokens.npz')); b=z['baseline_ids']; P=int(z['prompt_len'])
    rows=z['perturbed_ids']; L=z['perturbed_lengths']
    T=[]
    for i in range(rows.shape[0]):
        l=int(L[i]); mm=np.where(rows[i,P:l]!=b[P:l])[0]
        T.append(int(mm[0]) if len(mm) else -1)
    print(os.path.basename(d), dict(zip(*np.unique(T,return_counts=True))))
PY

# 2. FP32 baseline audit (windows 10..4096)
python3 _audit/margin_audit.py

# 3. Fixed-direction staircase audit
python3 _audit/fixeddir_audit.py

# 4. Periodicity audit
python3 _audit/periodicity_audit.py

# 5. Baseline protocol validation (DeepSeek FP32)
python3 _audit/baseline_protocol.py \
  --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
  --prompt @prompts.txt --window 100 --max-new-tokens 120 \
  --dtype float32 --attn sdpa --batch-size 16 --num-directions 3 \
  --magnitudes 0,1e-4,1e-3 --seed 69 \
  --out analysis_audit/protocol_ds_fp32_w100 --solo-check

# 6. Baseline protocol validation (Phi FP16)
python3 _audit/baseline_protocol.py \
  --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
  --prompt @prompts.txt --window 10 --max-new-tokens 120 \
  --dtype float16 --attn sdpa --batch-size 8 --num-directions 2 \
  --magnitudes 0,5e-6 --seed 69 \
  --out analysis_audit/protocol_phi_fp16_w10 --solo-check
```

Audit outputs are in `~/LLM/chaos_llm_2026/analysis_audit/`:
`margin_audit.json`, `fixeddir_audit.json`, `periodicity_audit.json`, `ds_batch_audit.json`, `protocol_ds_fp32_w100/`, `protocol_phi_fp16_w10/`.

---

## 9. Open / unverified items

- **UNVERIFIED:** whether DeepSeek bf16 batch-shape affects longer generations or other windows. The 120-token window-100 test was identical; this does not prove stability everywhere.
- **UNVERIFIED:** a large-sample FP32 free-run check at window 200 (only 3 directions, 90 tokens were tested).
- **UNVERIFIED:** whether the saved `outputs_OLD` FP16 baselines are stable for prompts other than `interstellar_travel`. All token-level runs in this repo use one prompt.
- **UNVERIFIED:** the role of `repetition_penalty` in the old hidden-state datasets; `launch_sep` uses 1.1, the token-level pipeline uses 1.0.

---

## 10. Index conventions (use these everywhere)

- Prompt positions: `0 … P-1`. Generated positions: `t = 0 … T-1`, where `t=0` is the **first generated token** (absolute position `P`).
- When we say “the response grows over the first ~100 steps”, we mean the first ~100 **generated** tokens (`t=0…99`), **excluding** the prompt. The prompt response (`0…P-1`) is saved and reported separately.
- For a current-state perturbation injected at generation step `s`, report time relative to injection: `τ = t - s`; `τ=0` is the injection step.
- Always save all positions. In figures, label the axis explicitly (`generated token index` vs `absolute position`). Default reporting is generation-only; prompt response is a separate panel.
- A “direction” is defined **at a specific injection site** (the full prompt embedding, or one token/layer/step). Directions at different sites are not comparable and must not be pooled without saying so.

---

## 11. Pitfalls (do not repeat)

1. **Never compare a batch-1 baseline to a batch-N perturbed run.** Put `r=0` in the same batch.
2. **Never use bf16/fp16 for small-`r` science.** Use FP32 for DeepSeek. If FP32 does not fit, reduce batch/sequence length, not precision.
3. **Always verify the `r=0` control** before interpreting any divergence. If it fails, the run is invalid.
4. **Do not average first-flip times `T`.** `T(r)` is a staircase pinned by near-ties. Use survival curves, hazards, and threshold distributions.
5. **Do not treat teacher-forced response after the first flip as the actual free-run trajectory.** It is a counterfactual with the baseline tokens.
6. **Do not pool directions across tokens/sites.** A fixed direction at token A and token B means different things.
7. **Do not save full `N×N` distance matrices to disk.** Compute correlation sums in chunks and save only `C(ε)` curves.
8. **Do not mix prompts for a single D₂ estimate** without also reporting per-prompt values. A mixture of trajectories is not one attractor.
9. **Do not call a distance-curve FTLE a Lyapunov exponent.** The old `log(d(t)/d(0))/t` is not a tangent-map exponent.
10. **Do not generate hidden states step-by-step without a KV cache.** It is O(T²) and unnecessarily slow; use a batched KV-cache loop or one teacher-forced forward pass.
11. **Do not use `output_hidden_states=True` for all layers/tokens unless needed.** Save selected layers and norms; full tensors are large.
12. **Do not interpret a single `σ_max > 1` as chaos.** A component can amplify one step while the product contracts. Report `σ_max`, `ρ`, `κ`, and the product’s singular values.
13. **Do not choose one FSLE scale factor `ρ` and hide the choice.** Report `ρ ∈ {1.5, 2, 4}` (or show the curve is flat).
14. **Do not report a single number when the object is a distribution.** Show 3 individual cases plus an aggregate; label what is aggregated.
