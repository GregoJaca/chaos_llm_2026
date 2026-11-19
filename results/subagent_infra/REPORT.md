# Multi-model protocol audit — REPORT

**Scope:** `src/chaos_llm/protocol.py` + the 8 named scripts, prepared for running the
DeepSeek-R1-Distill-Qwen-1.5B perturbation protocol on
`Qwen/Qwen3.5-2B`, `Qwen/Qwen3-4B-Instruct-2507`, `microsoft/Phi-4-mini-instruct`.
**Method:** full read of the listed sources; hub configs/tokenizer configs fetched
2026-11-19 for all four models; masking mechanics verified line-by-line against
`transformers` 4.49.0 `modeling_phi3.py` and 4.51.3 `modeling_qwen3.py`; Qwen3.5
verified against the transformers v5.23 docs (`Qwen3_5ForCausalLM`,
`Qwen3_5TextConfig`). The audit machine has no torch/transformers, so everything
GPU-dependent is delivered as a *runtime check* (`selftest_cpu.py`, `smoke_gpu.py`)
rather than assumed. All delivered code was compiled and its stdlib core unit-tested
(`_local_test_*.py` in this folder; 41 + 21 checks pass).

Line numbers refer to the current repo files (`src/chaos_llm/...`).

---

## Part A — the known bugs, verified

### 1. Forwards outside `torch.inference_mode()` kept the autograd graph — OOM > 600 tokens
**Where:** mostly already fixed in the scripts (GPU_GUIDE rule 2). Remaining holes:
- `protocol.py:86-95` — `full_embeddings()` is **not** decorated; it is called outside
  any inference block at `run_a1.py:82`, `run_a1_long.py:38`, `run_state2.py:57`,
  `run_layer_probe.py:39`, so the embedding lookup is run with grad enabled
  (`load_model` never calls `requires_grad_(False)`).
- `protocol.py:31-38` — `load_model` leaves all parameters `requires_grad=True`.
**Why it breaks:** every script that re-uses the returned `E` outside an inference
block keeps a (small) graph slice alive; with bf16/fp32 activations on a 4B model this
is pure waste and on some paths (`E[j] += delta` on a non-inference tensor) silently
depends on autograd bookkeeping.
**Fix:** `protocol_general.full_embeddings` is `@torch.inference_mode()`; `load_model`
calls `model.requires_grad_(False)`. Call-site tensors were checked: `expand().clone()`
of an inference tensor outside inference mode is legal (clone materializes a normal
tensor), so the scripts' in-place perturbation code still works.

### 2. `make_directions(k=1)` → degenerate simplex → NaN
**Where:** `protocol.py:50-53` (`build_simplex`: for k=1, `1 - 1/1 = 0`, and
`0/||0|| = 0/0 = NaN`), `protocol.py:61-64` (no guard). Documented as GPU_GUIDE rule 5
but never enforced.
**Fix:** `build_simplex`/`make_directions` raise `ValueError` for `k < 2`
(`protocol_general.py`); `smoke_gpu.py --dirs < 2` refuses to run. The degeneracy
itself is pinned by a selftest check (norm of the k=1 vertex is exactly 0).

### 3. Hardcoded D=1536 / layer 27 / depth 28
**Where (exhaustive — the exact spots, including the ones the task list did not name):**
| file:line | hardcode |
|---|---|
| `run_state.py:15` | `D=1536` |
| `run_state_mid.py:16` | `D=1536` |
| `run_a1.py:39` | `--layers` default `"0,7,14,21,27"` (28-layer DeepSeek) |
| `run_a1_long.py:21` | `--layers` default `'7,14,21,27'` |
| `run_state2.py:37` | `--layers` default `'7,14,21,27'` |
| `collect_trajectories.py:12` | `--layers` default `'7,14,21,27'` |
| `run_state.py:11` | `--layer` default 27 |
| `run_state_mid.py:12` | `--measure-layer` default 27 |
| `run_layer_probe.py:17,43,48,65` | `range(28)` injection/measurement depth |
| `run_after_flip.py:63` | `hidden_states[26:28]` |
| `run_margins.py:12`, `run_attention.py:10`, `run_nonnorm.py:14-15` | layer defaults incl. 27 |
**Note on the task hint:** `run_state_freerun.py` has *no* D/layer hardcode (D comes
from `model.config.hidden_size` at line 33); the D=1536 hardcodes are in
`run_state.py`/`run_state_mid.py`.
**Why it breaks:** Qwen3-4B has **36** layers (index 27 is mid-network, the last block
35 is never measured), Phi-4-mini **32**, Qwen3.5-2B **24** — `run_layer_probe.py`
crashes with `IndexError` (hidden_states has only 25 entries, code takes `range(28)`);
`run_state2 --inject-layer` would silently probe the wrong depth on 36 layers.
**Fix:** patched scripts derive everything from `model_info()`: `--layers` defaults to
`[0, n/4, n/2, 3n/4, n-1]` which for n=28 reproduces the legacy `0,7,14,21,27`
**exactly** (unit-tested), and `run_layer_probe` covers all blocks + all `n+1`
hidden states, with explicit range validation.
(`run_state.py`, `run_state_mid.py`, `run_after_flip.py`, `run_margins.py`,
`run_attention.py`, `run_nonnorm.py` were NOT patched — not in the task's list; the
fix pattern is identical: take D from `model_info()["hidden_size"]` and validate layer
indices. Do not run them on new models as-is.)

### 4. `set_window` is Qwen2-specific; on other architectures it silently does nothing
**Where:** `protocol.py:41-43` sets `config.sliding_window` + `config.use_sliding_window`
blindly.
**Verified mechanics (transformers 4.49.0 / 4.51.3 sources):**
- *Qwen2/Qwen3* (sdpa/eager): the causal mask is rebuilt **at every forward** from
  `config.sliding_window` (`_prepare_4d_causal_attention_mask_with_cache_position`),
  so post-load mutation **does** take effect — this is why the DeepSeek flush-law runs
  worked. Flash-attention-2 instead uses per-attention `self.sliding_window`, captured
  in `__init__` and gated by `use_sliding_window && layer_idx >= max_window_layers`
  (4.51.3 `Qwen3Attention.__init__`), i.e. a config mutation alone is a **no-op** on FA2.
- *Phi3* (native): `Phi3Attention.forward` reads `getattr(self.config, "sliding_window",
  None)` at forward time → config mutation **works**; `use_sliding_window` is not a
  Phi3Config attribute, so `protocol.py:43` was silently writing a dead attribute.
  Phi-4-mini ships `sliding_window: 262144` (effectively full attention) and
  `full_attn_mod: 1`.
- *Qwen3-4B-Instruct-2507*: config has `sliding_window: null`, **no `layer_types`**
  (dense full attention). Setting `sliding_window=100` works through the sdpa mask path
  (same intervention semantics as DeepSeek, which also ships
  `use_sliding_window: false`); on FA2 it would be a no-op (the init-time gate
  `layer_idx >= max_window_layers=36` is never true). **Stay on sdpa for window runs.**
- *Qwen3.5-2B*: the text config has **no `sliding_window` attribute at all** —
  the architecture is a 3:1 Gated-DeltaNet/full-attention hybrid; there is no window
  mechanism. `set_window` on it is a silent no-op in the old code.
**Fix:** `protocol_general.set_window` only touches `sliding_window` when the config
actually has the attribute, propagates the value to every attention module that
captured it at init (covers FA2), logs the mechanism + Qwen3 legacy
`max_window_layers` gate, returns an info dict, and **warns loudly** for
unsupported/hybrid models. `window_is_effective()` (protocol_general) is an empirical
end-to-end check: two sequences differing only *before* the last W tokens must produce
bitwise-identical final hidden states if the window clips; it also runs a full-attention
control so an inconclusive test is reported as such. Wire it into the smoke step.

### 5. Tokenizer/prompt handling: no chat template, implicit special-token behavior
**Where:** `protocol.py:46-47` (`prompt_ids` calls the tokenizer with default
`add_special_tokens`), and `run_a1_long.py:26` (see bug 14).
**Decided protocol (documented in `protocol_general` docstring + every meta):**
**raw completion, no chat template, `add_special_tokens=False`**, greedy,
`eos_token_id=None`, fixed length. Rationale: the entire DeepSeek corpus was produced
this way (Qwen2 tokenizers add no BOS/EOS, so for DeepSeek the explicit `False` is
bit-identical to the old default); applying per-model chat templates now would make
every new run incomparable with the old ones and would put template control tokens in
the perturbed prefix.
**Verified per model:** Qwen3/Qwen2 tokenizers add no specials by default;
Phi-4-mini `tokenizer_config.json` has `add_bos_token: false, add_eos_token: false`
(class `GPT2Tokenizer`), eos `<|end|>`=200020 / ``=199999;
Qwen3.5 uses `Qwen3_5Tokenizer` (ChatML-style specials, 248320 vocab) — the selftest
prints the round-trip facts at runtime rather than assuming them.
**Consequences to accept & record:** trajectories are fixed-length even past the
model's own EOS (Phi-4-mini on raw text may emit `<|end|>` early — that is part of the
protocol; choose non-chat prompts so the continuation stays in completion mode).
`selftest_cpu.py` prints exactly whether each tokenizer's default adds specials, and
`generate()` (protocol_general) now *asserts* the returned length equals
`max_new_tokens` so a silent early stop can never pass unnoticed.

### 6. dtype selection: `load_model` always float32
**Where:** `protocol.py:34-37` (`torch_dtype=torch.float32`), `local_files_only=True`
always, plus `run_a1.py:150` records `dtype="float32"` in meta **unconditionally**
(for a bf16 run the metadata would lie).
**Fix:** `load_model(..., dtype="auto")`: fp32 if text-backbone params ≤ 2.5e9 else
bf16 (override `float32|bfloat16|float16`). Parameter count is estimated **without
loading weights**: safetensors headers (stdlib parser; excludes vision/mtp groups so
multimodal checkpoints count only the LM) or a meta-device instantiation from the
config. `model_info().dtype` is written into every meta. Result per model
(selftest prints it): DeepSeek 1.78B→fp32 (unchanged), Qwen3-4B 4.02B→bf16,
Phi-4-mini 3.8B→bf16, Qwen3.5-2B text≈2.2B→fp32.

### 7. Layer naming/count: `model.model.layers` hardwired
**Where:** `run_state2.py:91`, `run_layer_probe.py:60`, `run_a1_long.py:43`
(`model.model(...)` + `model.lm_head(...)`), `run_state_mid.py:30`.
**Verified:** Qwen2/Qwen3/Phi3/Qwen3_5 causal LMs all expose `lm.model.layers`
(candidate path `model.layers`) — *but* Qwen3.5 ships as
`Qwen3_5ForConditionalGeneration` (multimodal wrapper; text via
`Qwen3_5ForCausalLM`), whose text blocks live under a different nested path, and the
wrapper's `.lm_head` location is not guaranteed.
**Fix:** `protocol_general.model_layers()` resolves candidate paths
(`model.layers`, `model.language_model.layers`, …), cross-checks the length against
`config.num_hidden_layers` (text config when nested), and on failure **prints the
actual attribute layout and raises** (per the audit brief: no invented APIs).
`run_a1_long` no longer touches `model.model`/`model.lm_head`: it calls the wrapper
with `logits_to_keep=T+1`, which computes the lm_head exactly at the old `pos`
positions (`P-1..P+T-2`) without materializing `[L,V]` logits — identical values up
to fp rounding, valid for every wrapper class (requires transformers ≥ 4.50, which the
patched set already assumes).

### 8. `full_embeddings` / `generate`: pad/eos assumptions
**Where:** `protocol.py:67-75`, `86-95`.
**Verified:**
- `eos_token_id=None` is valid and means "never stop" for all four generation configs
  (DeepSeek/Qwen3 `eos [151645, 151643]`, Phi `[200020, 199999]`, Qwen3.5 text
  `248044`); with `inputs_embeds` generation the returned tensor contains **only** the
  new tokens, which is what the scripts' `T = gen.shape[1]` bookkeeping assumes.
  `generate()` now asserts `T == max_new_tokens`.
- `pad_token_id=0` is bookkeeping only (no row can finish early); id 0 is a valid
  vocab id for Qwen (id 0 exists in all four vocabs).
- `model.get_input_embeddings()` works for all four classes (tied checkpoints:
  Qwen3-4B, Phi-4-mini, Qwen3.5).
- One real dtype trap (see 11): `E[:, :P] += delta` with a bf16 model and fp32 delta
  raises a dtype-promotion `RuntimeError`.

---

## Part B — additional bugs found (not in the known list)

### 9. bf16 tensors crash `.cpu().numpy()` — numpy has no bfloat16
**Where:** `collect_trajectories.py:30,31`, `extend_trajectories.py:34`,
`run_a1.py:91,95,127,132,134,137`, `run_a1_long.py:51,75,77`,
`run_state2.py:63,101-107,114`, `run_layer_probe.py:66`.
`RuntimeError: Got unsupported ScalarType BFloat16` — this alone breaks every script
above on a bf16 model and is very likely "the previous attempt had errors" on
Phi-4-mini.
**Fix:** `to_np()` helper (`tensor.detach().float().cpu().numpy()`) used everywhere in
the patched set (remaining `.cpu().numpy()` calls operate on int64 token ids / fp32
directions only — audited).

### 10. fp32→bf16 in-place add crashes (`result type Float can't be cast to BFloat16`)
**Where:** the perturbation application sites: `run_a1.py:117-119`
(`E[j,:P] += delta`), `run_a1_long.py:68-70`, `run_thresholds.py:49-51`,
`run_state2.py:86-89` (embedding injection; the hook branch additionally replaced a
bf16 hidden state with an fp32 tensor, poisoning the next layer's matmul).
**Fix (uniform convention, documented in every patched script):** accumulate in fp32,
assign with a **single** rounding to the model dtype —
`E[j,:P] = (E[j,:P].float() + delta.view(P,D)).to(emb_dtype)`. No cross-dtype in-place
ops remain. For fp32 models this is bitwise-identical to the original.

### 11. `hard_window_mask` is fp32-only
**Where:** `run_a1.py:19-23` builds the 4D additive mask with
`torch.finfo(torch.float32).min` in fp32. `transformers` passes a user 4D mask
through to `F.scaled_dot_product_attention`, which requires the mask dtype to match
the query dtype → **crash on bf16 models** using `--hard-window`.
**Fix:** `protocol_general.hard_window_mask(L, W, dtype=...)` uses the embedding dtype
(passed by patched `run_a1`).

### 12. `run_a1_long.py:26` — the `--prompt @file` argument is never read
`text = args.prompt[1:]` stores the literal string `"prompts_long2.txt"` into
`meta["prompt"]`. Downstream, `run_thresholds.py:32-33` re-tokenizes
`meta["prompt"]` as the **free-run baseline prompt** → free runs launched from an
`a1_long`-produced directory would validate against the *wrong trajectory*.
**Audit of existing data:** I scanned all 75 local `results/**/meta.json` for
`"prompt": "prompts*.txt"` — **no contaminated file in the local copy**. Run the same
one-liner on the GPU host before reusing any a1_long outputs:
`grep -r '"prompt": "prompts' results/` (a hit ⇒ that free-run comparison is invalid).
**Fix:** patched `run_a1_long` reads the file (tab-format like `run_a1`), and patched
`run_thresholds` **refuses** a meta whose prompt looks like an `@file` string.

### 13. `--fixed-tokens-from` has no cross-model guard
**Where:** `run_a1.py:61-71`, `run_a1_long.py:27-30`, and every script that reuses a
stored `a1.npz` (`run_thresholds`, `run_state2`, `run_state_freerun`,
`run_layer_probe`, `extend_trajectories`). Qwen token ids fed to Phi-4-mini
(200064 vocab) are either an `IndexError` or silently meaningless teacher-forcing;
the reverse (Phi→Qwen) is silently garbage.
**Fix:** patched scripts raise when source ids ≥ the model's vocab, and warn when the
source meta names a different model. Note `run_thresholds` additionally checks that
re-tokenizing `meta["prompt"]` reproduces `meta["prompt_len"]` (it won't, across
models).

### 14. `resolve_model_path` cache handling is incomplete and can pick a stale snapshot
**Where:** `protocol.py:20-27`: snapshot revisions are chosen by **lexicographic**
order of the commit hash (not recency), and only `~/.cache/huggingface/hub` is
searched — `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE` are ignored; a miss returns
a nonexistent path silently.
**Fix:** `model_plan.resolve_model_path` prefers `refs/main`, falls back to the sort
with a logged warning, honors all cache env vars, keeps the legacy `~/LLM/...` layouts
(dir with `config.json` at root, or bare `models--*` repo dir with snapshots), and
returns the input unchanged (HF id) when nothing is local — with
`local_files_only` auto-set only for local dirs, so the three new hub models can be
downloaded on first use without breaking the DeepSeek default.

### 15. `load_model` does not disable grad / does not verify the loaded class
`requires_grad` stays True (compounds bug 1); and for nested-config repos
(Qwen3.5) `AutoModelForCausalLM` may not resolve. `protocol_general.load_model`
disables grad and, on AutoModel failure, tries the explicit text-only class derived
from the config (`Qwen3_5ForCausalLM`), and only then raises an error that prints
model_type/architectures/transformers version and what was tried.

### 16. `.cuda()` hardwired everywhere
All scripts and `protocol.py` assume CUDA. Patched scripts use `DEVICE`
from `protocol_general` (`cuda` if available else `cpu`) so smoke tests and the
selftest logic can run CPU-only. GPU runs are unchanged.

### 17. `run_thresholds.py` maps requested radii to the nearest meta radius silently
`run_thresholds.py:63-66`: `argmin(|mradii - r|)` can silently pair 1e-4 with a meta
radius of 1e-3 (10× off). Kept as-is for compatibility (it only affects the *predicted*
column, and the mapping is recorded in `summary.json`), but flagged here: choose
`--radii` that are an exact subset of the a1 radii when possible.

### 18. `run_a1.py:160` `dh_last[:, :dh_last.shape[1]]` is a no-op slice
Harmless oddity (kept); noting it so nobody "fixes" it into a shape change that breaks
old consumers.

### 19. Meta provenance gaps
Old metas never record `num_layers`/`vocab_size`/`model_class`/true dtype (and
`run_a1.py:150` hardcodes `dtype="float32"`). Patched scripts record the full
`model_info` subset — with three models in one analysis folder this is what prevents
silent mixups.

### 20. Phi-4-mini longrope: cache invalidation above 4096 tokens
Native `Phi3ForCausalLM.prepare_inputs_for_generation` **drops the KV cache and
re-prefills** when `input_ids.shape[1] >= original_max_position_embeddings + 1`
(4097 for Phi-4-mini; longrope short/long factor switch). Trajectories that cross
4097 total tokens will hit one expensive re-prefill step (and any custom
past_key_values surgery in `run_state_freerun`/`extend_trajectories` happens across
that boundary at its own risk). **Cap prompt+generated < 4096 for Phi-4-mini**, or
verify the boundary explicitly.

### 21. Phi-4-mini loading: remote code vs native
The model card's snippets use `trust_remote_code=True` (repo ships
`modeling_phi3.py` written against 4.45), but the card also states Phi-4 was
integrated in transformers **4.49.0** natively. `protocol_general.load_model`
defaults to `trust_remote_code=False` (native code, version-portable; verified
native `Phi3Attention` handles the 4.49+ attention interface and reads
`config.sliding_window` at forward time). Do a one-off native-vs-remote greedy
comparison (CHECKLIST step 4) before trusting native for the paper numbers; pass
`load_model(..., trust_remote_code=True)` if bit-compatibility with the old remote
path is ever required. **Do not** run the old remote code on transformers 5.x.

### 22. `torch.cuda.empty_cache()` calls on CPU-only machines are harmless but `.cuda()` was not
Covered by 16; noting only that `empty_cache()` itself is a no-op without CUDA.

---

## Part C — what is model-specific (the summary requested)

| | DeepSeek-R1-Distill-Qwen-1.5B | Qwen3-4B-Instruct-2507 | Phi-4-mini-instruct | Qwen3.5-2B |
|---|---|---|---|---|
| class (native) | `Qwen2ForCausalLM` | `Qwen3ForCausalLM` | `Phi3ForCausalLM` (remote code optional) | `Qwen3_5ForConditionalGeneration` / text: `Qwen3_5ForCausalLM` |
| transformers | ≥4.44 (current env) | **≥4.51** | ≥4.49 native | **5.x** (docs pin v5.23; card says install from git) |
| hidden / layers | 1536 / 28 | 2560 / 36 | 3072 / 32 | 2048 / 24 (text) |
| hidden_states entries | 29 | 37 | 33 | 25 |
| vocab (config) | 151936 | 151936 | 200064 | 248320 |
| params → dtype rule | 1.78B → **fp32** (7.1 GB) | 4.02B → **bf16** (8.0 GB) | 3.8B → **bf16** (7.6 GB) | text≈2.2B → **fp32** (~8.8 GB) |
| **window config key** | `sliding_window=4096, use_sliding_window=false, max_window_layers=21`; set via config, read at forward (sdpa/eager) | `sliding_window=null`, **no layer_types**; same mechanism, **must verify** with `window_is_effective` (sdpa only; FA2 ignores it) | `sliding_window=262144` (vestigial), `full_attn_mod=1`; native code reads config at forward | **no `sliding_window` — no window mechanism** (3:1 linear/full hybrid); window studies invalid |
| **layer path** | `model.layers` | `model.layers` | `model.layers` | `model.language_model.layers` (wrapper) — `model_layers()` resolves it |
| tokenizer specials | none added | none added | none added (`add_bos_token:false`) | verify with selftest |
| chat template | ChatML (unused) | ChatML (unused) | `<|role|>…<|end|>` (unused) | ChatML-style (unused) |
| protocol dtype floor | fp32: fine at r≥1e-6 | bf16: perturbation rounds away below **r≈1e-2** (see below) | bf16: same floor | fp32 (mamba state is fp32 anyway) |
| quirks | baseline of all results | full attention; 262144 ctx; max_window_layers=36 (gate never true) | longrope re-prefill >4096; instruct-tuned (emits `<|end|>` early on raw text) | MTP weights + vision tower in checkpoint (unused, harmless); needs `causal_conv1d`+`fla` or slow PyTorch fallback; 6 full-attention layers only |

**bf16 perturbation floor (why the radii ladder must be re-chosen for bf16 models):**
a unit-norm prompt perturbation of radius r spreads ~`r/√(P·D)` per embedding entry;
it survives rounding only if that exceeds roughly `|E|·2^-9` (half-ulp of bf16). With
P≈30, D≈2560–3072, |E|≈0.02 → **r_floor ≈ 1e-2**. The smoke script sweeps
r=1e-3…1 and reports the linear band + a suggested ladder; expect (and accept) that
the 1e-6…1e-3 decade is unreachable in bf16. This is a *scientific* limitation of
bf16, not a code bug — record it in the paper's protocol section.

---

## Part D — the one structural finding worth its own number

### 23. Qwen3.5-2B is not "a bigger Qwen": the protocol must be re-scoped for it
Verified from the hub config + transformers docs (2026-11-19):
- `Qwen3_5ForConditionalGeneration` (multimodal wrapper, nested `text_config`,
  model_type `qwen3_5` / `qwen3_5_text`); text-only use goes through
  `Qwen3_5ForCausalLM` + `Qwen3_5TextConfig`.
- **3:1 hybrid stack**: `layer_types` = 3×`linear_attention` (Gated DeltaNet) +
  1×`full_attention`, repeating — 24 layers ⇒ only 6 attention layers (idx
  3,7,11,15,19,23). Linear layers are recurrent (unbounded state), so:
  the *window flush law is not defined*; `set_window` warns and no-ops;
  `--hard-window` masks clip only the 6 attention layers and are **not** an
  equivalent intervention.
- Requires **transformers 5.x** (repo `transformers_version` 4.57.0.dev0; docs
  v5.23.0; model card: install latest from git) — a different environment from the
  other three models; the DeltaNet fast path additionally wants `causal_conv1d` +
  `fla`, otherwise "silently falls back to slower and more memory hungry PyTorch ops"
  (docs) — on a T4 there is no kernel build, so budget extra memory/time.
- Checkpoint contains vision-tower and MTP weights (`mtp_num_hidden_layers=1`);
  loading text-only will log unexpected-key warnings (harmless) and the
  **parameter-count rule must exclude them** — the safetensors-header estimate in
  `model_plan` does (text≈2.2B → fp32, verified by test).
- What still transfers 1:1: prompt perturbation → margins/thresholds (`run_a1`),
  free-run flips (`run_thresholds`), state kicks (`run_state2` inject at embeddings;
  **hook injection into a linear-attention block is allowed mechanically — the hook
  adds to the block's hidden-state output — but interpret depth-wise results knowing
  blocks alternate between two different operators**), trajectories, and the r=0
  duplicate check. `mamba_ssm_dtype: float32` keeps the recurrent state fp32 even
  though weights are fp32 here anyway.
- Loading path is handled but **must be smoke-tested first** (runtime check):
  `AutoModelForCausalLM` may not map `qwen3_5`; `load_model` then tries
  `Qwen3_5ForCausalLM` explicitly, and `model_layers()` prints the live module
  layout rather than guessing if neither path matches.

### 24. Version drift risk for the existing DeepSeek results
The windowed-masking code paths changed across transformers versions (4.51-era
`_update_causal_mask` vs 4.53+ `masking_utils` vs 5.x). The DeepSeek flush-law numbers
were produced on the *current* host environment; do not upgrade that environment in
place. The CHECKLIST therefore creates **env4 (4.51.3) fresh** for the new models and
leaves the old env untouched for any DeepSeek reproduction; if you must reuse one env,
re-run one stored A1 config and diff margins first (`np.array_equal` → safe; else
record the drift and keep version-pinned comparisons).

---

## Part E — what still needs a runtime check (cannot be verified from here)
1. `window_is_effective` on Qwen3-4B-Instruct-2507 with the exact installed
   transformers (expect True on sdpa; if False, use `--hard-window` for window runs —
   the run_a1 hard-window path is dtype-correct now).
2. Qwen3.5-2B load path (Auto → explicit class) and `generate(inputs_embeds=...)`
   with the hybrid cache under transformers 5.x — the smoke script covers load,
   forward, r=0 duplicate, and length assert.
3. Phi-4-mini native vs `trust_remote_code=True` greedy equivalence (one-off, 2
   minutes).
4. Qwen3.5 tokenizer round-trip / specials (selftest prints it).
5. Exact bf16 response floor per model (smoke prints the table + suggested ladder).

## Files delivered (this folder)
- `model_plan.py` — stdlib-only planning core (dtype rule, layer paths, window
  classification, safetensors param counting, path resolution). Unit-tested.
- `protocol_general.py` — drop-in for `protocol.py` (same public names) plus
  `model_layers`, `model_info`, `set_window` (with warnings), `window_is_effective`,
  `hard_window_mask`, `to_np`, `--dtype` handling.
- `patched/run_a1.py`, `run_a1_long.py`, `run_thresholds.py`, `run_state2.py`,
  `run_state_freerun.py`, `run_layer_probe.py`, `collect_trajectories.py`,
  `extend_trajectories.py` — complete replacements importing `protocol_general`.
- `selftest_cpu.py` — CPU-only: imports, syntax pass over everything, tokenizer
  round-trips, per-model config plan (no weights).
- `smoke_gpu.py` — one-time per-model validation (load / forward / r=0 duplicate /
  window / r-linearity + radii recommendation).
- `CHECKLIST.md` — operator runbook.
- `_local_test_model_plan.py`, `_local_test_selftest_localdirs.py` — the audit
  machine's own test runs (keep or delete).

**Backward compatibility:** every patched script keeps the original CLI (new flags are
optional; `--layers` empty means auto = legacy values on DeepSeek), and on
DeepSeek-1.5B in fp32 the patched arithmetic is bitwise-identical (fp32 accumulate,
`.to(fp32)` no-ops). The original `protocol.py` is untouched; scripts not in the
patched set still import it and are unaffected.
