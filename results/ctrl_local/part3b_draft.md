# Part 3b — draft: TF-exactness and direction anisotropy on the negative control

Status: measured and final (local CPU analysis). To be folded into
`notebook/controls.md` after review. Definitions follow that document;
nothing there is modified.

## Data

- **TF-exactness** (`results/ctrl_local/mamba130_tfp/tfp_r{0.3,1,2,3,5}.npz`):
  Mamba-130m, txt prompt, 20 simplex prompt-perturbation directions, 300
  tokens, both passes per direction-run:
  - `pred[0, u, t]`: token predicted at step $t$ by **one teacher-forced
    pass** over the baseline sequence with the perturbed prompt embedding;
  - `ar[0, u, t]`: token actually emitted at step $t$ by the **free run**
    (true autoregressive greedy decoding of the perturbed prompt);
  - `ref[t]`: the unperturbed baseline greedy sequence.
- **Direction gains**: the existing A1 response files
  `results/ctrl_local/mamba130/a1i_txt.npz` (Mamba-130m) and
  `results/local_data_ds/w100_layers/a1.npz` (DeepSeek-1.5B), same protocol
  (20 directions, 300 tokens, in-batch reference for Mamba).

**Definitions used here.** A *flip* at step $t$ for direction $u$ is
`pred != ref[t]` (TF prediction) or `ar != ref[t]` (free run). The *first
flip* is the earliest such $t$. After the first flip the free run has left
the baseline trajectory, so later agreement is not counted. A direction-run
is *bit-identical* if neither pass flips at any of the 300 steps. A *false
alarm* is a TF flip with no free-run flip; a *miss* is the converse. The
per-direction *gain* is
$g_u=\mathrm{median}_{t\in[1,50]}\|\delta h_L(t,u)\|/r$ at the last
measured layer (Mamba layer 23, DeepSeek layer 27), $r=0.1$; the *gain
spread* is $\max_u g_u/\min_u g_u$.

## Result 1 — one TF pass is an exact flip simulator for the control as well

![tfp](figures_controls/c3b_tfp.png)

Stacked bars per radius; each column is the 20 direction-runs of that
radius, split into bit-identical / exact / mismatched / false alarm / miss
(counts printed inside).

| radius $r$ | bit-identical | exact flip (step+token) | mismatched | false alarm | miss |
|---|---|---|---|---|---|
| 0.3 | 20 | 0 | 0 | 0 | 0 |
| 1 | 20 | 0 | 0 | 0 | 0 |
| 2 | 20 | 0 | 0 | 0 | 0 |
| 3 | 20 | 0 | 0 | 0 | 0 |
| 5 | 17 | 3 | 0 | 0 | 0 |
| **total** | **97** | **3** | **0** | **0** | **0** |

The three r=5 flips: direction 15 at $t=0$ (token 0), direction 10 at $t=3$
(token 1984), direction 17 at $t=21$ (token 310). In every case the TF pass
predicted exactly the observed step *and* the observed token.

Sanity checks pass: at $r\le3$ nothing flips anywhere (consistent with the
txt-prompt flip threshold $r^*=4.26$ from Part 3), and at $r=5$ the free
runs flip 3 times at steps $\{0,3,21\}$, matching the independent free-run
script.

**Verdict.** On the provably non-chaotic control, one teacher-forced pass
simulates the free run exactly up to the first token flip:
**3/3 flips exact, 0 false alarms, 0 misses, 97/100 direction-runs
bit-identical** (DeepSeek: 29/29 exact, 0 false alarms in 111 no-flip cases).
TF-exactness is a property of the decision boundary (the argmax layer), not
of chaos and not of attention. It can be used as a cheap simulator for the
control too — but it carries no information about the dynamics *between*
flips, which is where the models differ (Parts 2 and 4).

## Result 2 — the control's perturbation directions are anisotropic too

![dirs](figures_controls/c3b_dirs.png)

Sorted per-direction gains $g_u$ (log scale), both models, $r=0.1$, last
measured layer, median over $t\in[1,50]$.

| model | $\min_u g_u$ | $\max_u g_u$ | spread (t 1–50) | spread (t 1–299) | Spearman $\rho$, window [1,50] vs [150,299] |
|---|---|---|---|---|---|
| Mamba-130m (layer 23) | 0.27 | 1.78 | **6.5×** | 5.3× | **0.77** |
| DeepSeek-1.5B (layer 27) | 1.51 | 32.3 | **21×** | 11.5× | **0.92** |

- **The control has a real gain spread (≈6.5×)**: the same norm-$r$ kick to
  the prompt embedding produces a last-layer response that differs by up to
  ×6.5 between directions. Anisotropy of the prompt-response map exists
  without attention and without chaos (it is the non-normality measured in
  Part 4: $\sigma_{\max}\approx1$ while the median direction contracts).
- **But it is smaller and less ordered than the transformer's**: DeepSeek's
  spread is ×21 in the same early window (×11.5 over the full horizon,
  consistent with the established ×7–12), and its direction ordering is more
  stable in time ($\rho=0.92$ vs $0.77$).
- The spread answers the framing question: "does the non-attention control
  also have anisotropic directions?" — **yes, ×6.5; roughly a factor 2–3
  smaller than the transformer's and noticeably less time-stable.**

**Verdict.** Direction anisotropy by itself is *not* an attention or chaos
signature — a qualifier for any use of "danger directions" in the LLM
results. What discriminates is the *size* of the spread and its time
stability, both of which remain larger for the transformer.

## Data problems found

None. All five `tfp_r*.npz` files are complete (no unset `-1` entries in
either `pred` or `ar`; every free run emitted all 300 tokens), shapes match
the documented `[1, 20, 300]`, and the radii fields agree with the file
names. The r=5 flip steps reproduce the independent free-run measurement
exactly.
