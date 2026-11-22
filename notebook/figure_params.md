# Figure parameters — controls (one table, so nothing has to be guessed)

**Conventions shared by all runs.** FP32 throughout, on the project's single
Tesla T4 (torch 2.5.1+cu121, transformers 4.57.6), through the same
`src/chaos_llm` pipeline as the LLM work. Every perturbed batch contains an
exact r=0 reference row (`--inbatch-ref`); all $\delta h$ and $\delta m$ are
in-batch differences against that row, so the baseline is bit-identical.
Prompt perturbations use the master's sparse simplex directions ($K$
orthogonal unit rows written to $K$ random entries of the flattened
$P\times d$ prompt-embedding tensor, seed 69); single-position kicks
(`run_state2`), the kernel/superposition runs, flip fractions, and the
SSM/KV-cache state kicks use **dense** Gaussian unit vectors instead — same
Euclidean norm $r$ in both cases. All runs are teacher-forced on the
unperturbed greedy continuation unless the row says *free*; margins $m_t$ are
top1−top2 logit gaps along that continuation. Control models run at $W=4096$
("native"; Mamba, Pythia and the random Qwen have no sliding-window mechanism,
so $W$ is inert); every DeepSeek comparison curve is the master run at
$W=100$. 300 generated tokens unless stated. All parameters below were read
from the run metas (`results/ctrl_local/**/meta*.json`) and the invoking shell
scripts, not from memory.

| figure | model(s) | prompt | perturbation & radius | dirs | tokens / other |
|---|---|---|---|---|---|
| `c2_resp_time` | Mamba-130m, DeepSeek-1.5B | `prompts.txt` (interstellar; 67 tok Mamba / 65 DS) | prompt emb., r=0.1 (Mamba also 1.0; grids 3e-4–5 / 1e-3–2) | 20 | 300, TF, last layer; DS = master W=100 data |
| `c2_layers` | same | `prompts.txt` | r=0.1 | 20 | 300, TF; depth profile at t=10, 100; layers 0,6,12,18,23 |
| `c2_lambda` | Mamba (txt, childhood, cyber), DeepSeek (txt) | `prompts.txt`, `prompts_childhood.txt`, `prompts_cyber.txt` | r=0.1 | 20 | 300, TF, $\lambda_{10}(t)$ last layer |
| `c2_linearity` | Mamba-130m, DeepSeek | `prompts.txt` | r sweep: 3e-4–5 (Mamba), 1e-3–2 (DS); slope of $a(t{=}50)/r$ | 20 | 300, TF |
| `c2_T1000` | Mamba-130m | `prompts_long2.txt` line 1 (ocean currents, 63 tok) | r ∈ {1e-3, 0.1, 1} | 8 | **1000**, TF on the saved free-run trajectory |
| `c2b_scale` | Mamba-130m/370m/790m/1.4b | `prompts.txt` | r=0.1 (grid 1e-3–5) | 20 | 300, TF; layers 0,6,12,18,23 (130m), 0,12,24,35,47 (48-layer) |
| `c3_margins` | Mamba-130m, DeepSeek | `prompts.txt` | none (margins along the greedy run) | — | 300 |
| `c3_ladder` | Mamba (txt, cyber), DeepSeek (txt, cyber) | `prompts.txt`, `prompts_cyber.txt` | signed $r^*(u)$ measured at r=1 (Mamba) / 0.1 (DS) | 20 | 300, TF |
| `c3_obspred` | Mamba-130m | 3 prompts | **free** runs, 18 radii: txt 0.03,0.1,0.3,1,1.4,2,3,5; childhood/cyber 0.03,0.1,0.3,1,3 | 20/run | 300 free |
| `c3_survival` | Mamba-130m | same 18 free runs | same | 20/run | 300 free; agreement after first flip |
| `c3_flipstruct` | Mamba-130m | same 18 free runs | same | 20/run | 300 free; margins + response level at flip steps |
| `c3b_tfp` | Mamba-130m | `prompts.txt` | TF pass at r=0.3, 1, 2, 3, 5 vs the free run | 20 | 300; 100 direction-runs |
| `c3b_dirs` | Mamba-130m, DeepSeek | `prompts.txt` | r=0.1; per-direction gains $g_u$, median over $t\in[1,50]$ | 20 | 300, TF, last layer |
| `c4_kick` | Mamba-130m, Mamba-370m | `prompts.txt` (token stream of the a1i_txt run) | full-state kick (ssm+conv, **all** layers), $\varepsilon=10^{-2}$ at $t_0$=80, 150, 220 | 8 dense | horizon 200 (147 at $t_0$=220), TF |
| `c4_diag` | Mamba-130m, Mamba-370m | same | one-step $J$ diagonal, $\varepsilon=3\times10^{-2}$, $t$=70,100,150,220,290 | 24 (layer, channel) samples × 16 ssm dims, 9 layers | 1 step, TF |
| `c4_sigma` | Mamba-130m, Mamba-370m | same | $\sigma_{\max}(J_t)$ by power iteration (16 it, mean of last 8) | 1 | same 5 steps |
| `c5_loops`, `c5_d2`, `c5_d2_summary`, `c5_rp` | Mamba-130m (6 traj) vs DeepSeek (16 traj) | `prompts_long2.txt` (Mamba) / master's 16 long prompts (DS) | none: geometry of **free** greedy trajectories, last-layer states | — | 1000 free, $W$=4096; Hénon + Lorenz (m=8, $\tau$=4) refs at N=3031; D2 on first 1000 pts |
| `c5b_all_models`, `c5b_rp_gallery` | 5 LLMs, Mamba-130/370/790m, Pythia-410m (+step0), randQwen g=0.35/1/4 | `prompts_long2.txt` lines 1–3 (or 1–4/1–6 per model); DS: 16 prompts | none | — | 1000 free, $W$=4096, last layer; RP at matched RR=2% |
| `c6_phase`, `c6_response`, `c6_obspred` | randQwen (exact DeepSeek arch, 28 layers; update-matrix init std $0.02\,g$, seed 1234) | `prompts.txt` | prompt emb., r=0.1 (grid 3e-5–1); depth probe: kick at s=20, r=0.05; free runs at 2–3 auto-picked radii per gain | 20 (8 in depth probe) | 300; $g$ ∈ {0.35, 0.5, 0.71, 1, 1.41, 2, 2.83, 4} |
| `c7_pythia_resp` | Pythia-410m, Pythia-1b, Pythia-410m step0, Mamba-130m, DeepSeek | `prompts.txt` | r=0.3 (410m), r=0.1 (others) | 20 | 300, TF; DS = master W=100 data |
| `c7_pythia_obspred` | Pythia-410m | `prompts.txt` | **free** runs r=0.19, 0.3, 0.7, 1.4, 3 vs signed rule | 20 | 300 |
| `c7_rstar` | Pythia-410m step0, Mamba-130m, Pythia-410m/1b, DeepSeek | `prompts.txt` | $r^*$ at sub-flip radius: 0.1 (most), 1.0 (Mamba), 0.19 (410m) | 20 | 300, TF |
| `c8_kernel` | Mamba-130m; DeepSeek, Phi-4-mini (master kernels) | `prompts.txt` (67/65/64 positions) | one prompt position, dense unit vector, r=0.1 | 8 | 300, TF, last layer, all positions; DS kernel at $W$=100 |
| `c8_superposition` | Mamba-130m | `prompts.txt` | same $u$ at every position, r=0.01/0.03/0.1 per position (total $r\sqrt P$=0.08–0.81), 3 trials | 1 per trial | 300, TF |
| `c8_lambda_all` | 5 LLMs (master values) + Mamba ×3 prompts, Mamba-790m, Pythia-410m, step0, randQwen g=1 | `prompts.txt` | r=0.1 (0.3 Pythia-410m) | 20 | 300, TF; late $\lambda$ = mean over $t\in[150,290]$ |
| `c8_s16_spread` | Mamba-130m (+ master table values) | `prompts.txt` | one-position kick at s=16, 50 (generated), inject layer 0, r=0.1, dense | 8 | 300, TF, layers 0,6,12,18,23; spread 5–95 pct |
| `c8_flipfrac` | Mamba-130m | 8 prompts: `prompts.txt`, `prompts_childhood.txt`, `prompts_cyber.txt`, `prompts_more.txt` | dense dirs, r=0.1, 0.3, 0.6, 1.0; tie rule $m\le0.02$ | **192** | 300 **free** (figure pending: run data not yet in repo) |
| `c8_kv_kick`, `c8_kv_layers` | DeepSeek-1.5B, **$W$=100** | `prompts.txt` | KV-cache kick, $\varepsilon$=0.01, 0.1, 1.0 at $t_0$=70, 150; layer modes all/0/14/27; **live-window (k,v) entries only** | 8 dense | horizon 150, TF (figure pending: run data not yet in repo) |

Free-run radii notes: the `fr_*` Mamba free runs reuse the *same* simplex
directions as the a1i run they are validated against; `c8_flipfrac` and the KV
kicks draw fresh dense directions. Master-data comparison curves (`c2`, `c3`,
`c3b`, `c7`, `c8` DeepSeek panels) come from `results/local_data_ds/w100_layers/a1.npz`
and `results/local_data_ds/a1_cyber_w100/a1.npz`: `prompts.txt`/`prompts_cyber.txt`,
$W$=100, 20 dirs, radii 1e-3–2, 300 tokens.
