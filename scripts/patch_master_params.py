p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()

anchor = "---\n\n## 2. What is perturbed, exactly"
table = r"""### Figure parameters (one table, so nothing has to be guessed)

`W` = sliding-window size. "native" = the model's own window setting; "inert" = the model has no
window mechanism, so `W` has no effect. `dirs` = number of random perturbation directions. All
teacher-forced unless stated.

| figure | model(s) | prompt | W | r | dirs | tokens |
|---|---|---|---|---|---|---|
| `flux_models_all_layers` | DeepSeek, Qwen3-4B, Phi, Qwen3.5 | `prompts.txt` (65 tok) | 100 (inert except DeepSeek) | 0.1 / 0.3 (Phi) | 20 | 300 |
| `model_lambda_compare` | same four | `prompts.txt` | 100 | 0.3 | 20 | 300 |
| `long_horizon` | DeepSeek | `prompts_long2.txt` | **4096** | 0.1 | 4 | **4000** |
| `block_sigma_full` | DeepSeek, Qwen3-4B, Phi | `prompts.txt` | 100 | Jacobian (no r) | 1 (power iteration, 20 it) | 4 positions |
| `sigma_vs_real` | DeepSeek | `prompts.txt` | 100 | 0.1 | 8 (depth map) / 6 (probe) | 300 |
| `relative_response_models` | five models | `prompts.txt` | 100 | 0.1 / 0.3 | 10–20 | 300, $t=50$ |
| `kernel_deepseek` | DeepSeek | `prompts.txt` | 100 | 0.1 | 8 | 300, 65 positions |
| `superposition` | DeepSeek | `prompts.txt` | 100 | 0.003–0.1 | 3 trials, same $u$ | 300 |
| `margin_coordinate` | DeepSeek | `window_sweep` prompt (63 tok) | **200** | 0.03–0.3 | 10 | 300 |
| `state_kick_latent_validated` | DeepSeek | `prompts.txt` (65 tok) | **100** | 0.1 | 24 | 300, $s=16,50$ |
| `posscan_A_heatmap`, `peak_is_absolute` | five models | `prompts.txt` | 100 | 0.1 / 0.3 | 8 | 300, 22 positions |
| `window4d_models` | Qwen3-4B, Phi (+ DeepSeek) | `prompts.txt` | 60–4096 | 0.1 | 6 | 300 |
| `rstar_per_model` | four models | 5 prompts | 100 | 0.1 | **512** | 300 |

"""
assert anchor in s
s = s.replace(anchor, table + anchor, 1)

# parameter line for the latent figure
old = r"""**What is plotted.** (a) Predicted (teacher-forced rule) against observed (free run) first changed
token, five perturbations."""
new = r"""**Parameters.** DeepSeek-1.5B, prompt `prompts.txt` (65 tokens), **window $W=100$**, $r=0.1$, **24
directions**, free-running generation, 300 tokens, perturbation positions $s=16$ and $s=50$.

**What is plotted.** (a) Predicted (teacher-forced rule) against observed (free run) first changed
token, five perturbations."""
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("added parameters table and latent parameters")
