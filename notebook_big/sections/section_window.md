# Part window program. The 14B death law is exact to the token; the cascade line is only an upper bound

## Summary

Six true sliding-window runs of `DeepSeek-R1-Distill-Qwen-14B` (FP32, same
65-token prompt `interstellar_travel`, the same 240 directions at radius
$r=0.1$, the same 500 teacher-forced baseline tokens, windows
$W=50,100,150,200,300,500$) show that a prompt perturbation disappears from
the first block at exactly the step the window law predicts. For every
direction whose perturbed prompt positions have maximum $p_{\rm last}$, the
block-0 response is nonzero while some perturbed prompt position is inside the
window and reaches the numerical floor at
$t^*=p_{\rm last}+(W-P)$. The equality is exact for all 240 directions at
$W=50$ and all 205 resolved directions at $W=100$; at larger $W$ the floor
sits closer to the pre-death signal and the measured crossing jitters by
$-3\ldots+2$ tokens, never more. A separate support scan (the same $W=100$,
the same prompt length, the same tokens and directions, only the perturbed
positions moved to $[0,22)$, $[22,44)$ or $[44,65)$) moves the death step by
exactly the change in $p_{\rm last}$: 720 of 720 directions lie within two
tokens of $p_{\rm last}+35$.
The literal test asked for in the brief — "the response becomes bitwise zero"
— **fails on the 14B for a numerical reason, not a physical one**: 0 of 1440
directions ever reach exactly zero or even $10^{-6}$: fp32 SDPA leaves a
round-off residue of about $1.0\times10^{-6}$ of the state norm
($2.9$–$3.0\times10^{-5}$ absolute at block 0, rising with depth), consistent
with the baseline being a single row while the perturbed rows run in batches
of 8. The same
test on the 1.5B, where the residue is absent, gives exact zeros at
$p_{\rm last}+(W-P)$ to the token for all five windows.
The cascade prediction $t^*_l=t^*_1+(l-1)(W-1)$ is confirmed as an **upper
bound**, not an equality, at the 14B's observable precision: every measured
death is at or before the bound, but the response reaches the floor 9–32
tokens early at state 2 and 30–129 tokens early at state 3 (medians; the gap
grows with depth). On the 1.5B, where the exact zero is observable, the
cascade is an exact equality: state 2 dies at 177, 277, 377 for
$W=100,150,200$ — exactly $t^*_1+(W-1)$. The 14B's smooth decay simply falls
below the fp32 floor before the last (already tiny) contribution disappears.
Finally, the window changes the decision landscape itself on identical
tokens: the two early razor ties are preserved bit-for-bit for every
$W\ge100$, while the number of margins below 0.5 nats nearly doubles from 62
($W=500$, approaching the no-clipping reference) to 116–120 at $W=100/50$;
at $W=50$ the prompt does not fit and the $t=21$ tie is destroyed. Every
window shows a flush surge one step after $t=W-P$: x2.19 at $W=100$ (the
1000-direction core run gave x2.20), x2.00 at $W=150$, x2.59 at $W=200$,
x3.95 at $W=300$, x2.41 at $W=500$ (the 1.5B reference was x1.73).
The prompt-length scan and the 1500-token cascade run are **pending** (not
yet local); their figures are placeholders and the script processes them
automatically if the files appear.

---

## 1. Runs, construction, and the fp32 floor

### 1.1 Setup

| item | value |
|---|---|
| model | `DeepSeek-R1-Distill-Qwen-14B` (48 blocks, hidden size 5120) |
| precision / decoding | FP32, TF32 off; greedy baseline, teacher-forced repeats |
| prompt | `interstellar_travel`, $P=65$ tokens |
| perturbation | unit direction $\times\,r=0.1$, added to prompt embeddings only |
| directions | $K=240$ per run, 12 disjoint simplex blocks of 20, same seed, same directions in all six runs |
| window | true sliding window, patched and verified; $W=50,100,150,200,300,500$ |
| continuation | the **same** 500 greedy tokens in every run (fixed from `a1_inter_w4096`) |
| saved states | 16 per run, `layers = [0,1,2,3,5,8,12,16,20,24,28,32,36,40,44,47]`; state 1 = block-0 output, state 0 = input embedding |
| reference | `a1_inter_w4096`, $W=4096$ (no clipping within 500 tokens), 32 directions |
| files | `results_big/ds14/a1_inter_w{W}_fixed/a1.npz` |

All six runs share the identical `gen_ids` array (checked), so there is no
text divergence anywhere in this section: every margin difference is caused
by the attention mask alone. The 240 directions are also identical across
runs; the perturbed positions of each direction are 20 random prompt
positions, and $p_{\rm last}(u)$ — the largest of them — takes values
$55\ldots64$ with median 59.5 in every run.

### 1.2 Window mechanics (the law being tested)

With a sliding window of width $W$, the generated position $P+t$ attends only
to positions $\max(0,P+t-W+1)\ldots P+t$. A perturbed prompt position $p$
(0-indexed) is therefore inside the window while
$p\ge P+t-W+1$, i.e. while $t\le p+(W-P)$; the first generated step at which
it is gone is
$$t^*(p)=p+(W-P).$$
For a direction $u$ whose perturbed positions have maximum $p_{\rm last}(u)$,
the block-0 response must fall to the floor at
$$t^*_1(u)=p_{\rm last}(u)+(W-P),$$
because before that at least one perturbed position is visible and after it
none is. Deeper blocks keep a response longer: block $\ell$ at position
$P+t$ reads block-$(\ell-1)$ states at previous positions **inside its own
window**, so the last possible path carries the information one window span
per block,
$$t^*_\ell=t^*_1+(\ell-1)(W-1).$$
This is a strict upper bound on the last nonzero contribution (a "front"),
not a claim about where the response becomes numerically invisible. When
$P>W$ the formula still applies: positions $p<P-W$ are already outside the
window at $t=0$ (the formula gives $t^*<0$, i.e. dead from the start), and
the surviving positions die at $t^*=p+(W-P)\in[0,W-1]$.

### 1.3 The literal zero test: 0 of 1440 directions reach zero on the 14B

The brief asks for the first $t$ with `dh_norm[1] <= 0` (state 1 = block-0
output), with $10^{-6}$ as a tolerance. Measured over all six runs:

| W | directions | reach exactly 0 | reach $<10^{-6}$ | smallest value over all $t$ | floor (5th pct of tail) |
|---:|---:|---:|---:|---:|---:|
| 50 | 240 | 0 | 0 | $2.19\times10^{-5}$ | $2.99\times10^{-5}$ |
| 100 | 240 | 0 | 0 | $2.14\times10^{-5}$ | $2.94\times10^{-5}$ |
| 150 | 240 | 0 | 0 | $2.02\times10^{-5}$ | $2.92\times10^{-5}$ |
| 200 | 240 | 0 | 0 | $2.09\times10^{-5}$ | $2.91\times10^{-5}$ |
| 300 | 240 | 0 | 0 | $2.12\times10^{-5}$ | $2.89\times10^{-5}$ |
| 500 | 240 | 0 | 0 | $2.49\times10^{-5}$ | $2.98\times10^{-5}$ |

So the literal test gives **no deaths at all**, and the reason is visible in
the last two columns: the response bottoms out at the same
$2$–$3\times10^{-5}$ in every window. This is not physics. The residue is present at every step,
before and after the expected death, and its size scales with the state norm:
it behaves like fp32 round-off from the different batch shapes (a single-row
baseline forward versus perturbed rows in batches of 8), not like a surviving
response. At block 0 the relative size is
$\approx1.0\times10^{-6}$ of the state norm (median norm 29.9,
hence $3\times10^{-5}$). At deeper states the residue scales with the larger
residual norm, reaching $2.3$–$3.5\times10^{-6}$ relative at layers 8–47
($1.7\times10^{-4}$ to $3.1\times10^{-3}$ absolute). The residue is
direction-independent in size and present at every step, including the
pre-death plateau, so it cannot be mistaken for a surviving response. At
$W=500$ the directions with the latest support do not reach the floor inside
the 500-token horizon (their predicted death is at 490–499); the 5th
percentile of the tail is then only a floor estimate, not evidence that all
directions died, which is why no $W=500$ direction is counted as resolved.
On the 1.5B the baseline and perturbed passes are numerically identical at
zero response, and exact zeros do occur (§8).

**Death is therefore defined in this section as the first step that starts
the final excursion below the round-off floor**: `dh_norm(t) <= 2 x floor`
and at least 90% of the next 30 steps stay below `4 x floor`, where the floor
is the 5th percentile of the last 60 steps (per state) and at least
$10^{-6}\times$ the median state norm. A direction is counted as *resolved*
only if its pre-death signal exceeds $10\times$ the floor and at least 11
steps remain in the run. On the 1.5B, where exact zeros exist, this rule
returns the same steps as the literal rule; on the 14B it is the only rule
that can be applied to all 240 directions. The raw arrays are unchanged; the
JSON reports the literal counts alongside the calibrated ones.

---

## 2. Figure `ds14win_departure.png` — state 1 dies exactly at $p_{\rm last}+(W-P)$

![departure](figures/ds14win_departure.png)

**Setup.** Each point is one direction: predicted death step
$p_{\rm last}(u)+(W-P)$ on $x$, measured death (first step at the floor, rule
above) on $y$; one colour per window. Panel (a) is state 1 (block-0 output),
which should follow the law exactly. Panels (b) and (c) are states 2 and 3,
which are tested against the cascade bound $t^*_\ell$; the line there is a
bound, so all points must be at or below it. Open black squares in (a) are the
1.5B's exact zeros (four directions per window, five windows); they are not
used in any 14B statistic.

**Axes.** $x$: predicted step, tokens since the first generated token.
$y$: measured death step. Both axes share the identity line; limits are
chosen per panel to contain every point. No logarithmic compression.

**Numbers.**

| W | directions resolved | exactly on the line | within ±2 tokens | median deviation | min | max |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 240 / 240 | 240 | 240 | 0 | 0 | 0 |
| 100 | 205 / 240 | 205 | 205 | 0 | 0 | 0 |
| 150 | 185 / 240 | 100 | 185 | 0 | 0 | +1 |
| 200 | 149 / 240 | 64 | 148 | +1 | −3 | +2 |
| 300 | 52 / 240 | 45 | 52 | 0 | −2 | +1 |
| 500 | 0 / 240 | — | — | — | — | — |

The "unresolved" directions are not failures of the law: at $W=500$ the
predicted death is $490$–$499$, leaving fewer than 11 steps, and at large $W$
the pre-death signal of directions with late $p_{\rm last}$ has decayed to
within a factor 3 of the floor, so the crossing step cannot be localised
better than a token or two. The maximum absolute deviation among all 831
resolved directions is 3 tokens, in one $W=200$ direction.

**What it shows.** The equality holds to the token wherever the measurement
is resolved: for $W=50$ and $W=100$ every single direction dies exactly at
$p_{\rm last}+(W-P)$, with no scatter at all (240/240 and 205/205). The
identity line is not a fit. The support of $p_{\rm last}$ is the same in
every run (55–64), so the five colours that have resolved points occupy
disjoint diagonal segments at the predicted offsets $W-65$: this is a scan
of the window width, not of the directions.

**What it does not show.** It cannot show the exact zero the brief asked
for, for the numerical reason in §1.3. It also cannot distinguish "the
response is exactly zero" from "the response is below $3\times10^{-5}$";
the 1.5B points show that where the exact zero is observable, the two
coincide. It says nothing about what happens at $W>500$ with 500 tokens,
because the predicted death then lies beyond the horizon.

---

## 3. Figure `ds14win_cascade.png` — the cascade bound is an upper bound at 14B precision

![cascade](figures/ds14win_cascade.png)

**Setup.** Panel (a): for each window, the median over directions of the
measured death at saved state $l$ (solid, markers) and of the predicted bound
$t^*_l$ computed from each direction's own $t^*_1$ (dashed, same colour).
$W=100$ and $W=200$ are drawn bold, the other windows thin for context.
Panel (b) is reserved for the 1500-token cascade run (8 directions, window
100, states 0–14), which is **pending**; it will be drawn automatically when
`results_big/ds14/a1_inter_w100_cascade/a1.npz` appears.

**Axes.** $x$: saved state index $l$ (1 = block-0 output; the saved-state
spacing in depth is not uniform, but the cascade prediction depends only on
the index). $y$: death step, median over directions. Linear axes; the dashed
bounds have slope $W-1$ by construction.

**Numbers.**

| W | l=1 | l=2 | l=3 | l=4 | l=6 | l=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 measured | 44.5 | 84.0 | 111.0 | 141.0 | 182.0 | 208.0 |
| 50 bound | 44.5 | 93.0 | 142.5 | 191.5 | 289.5 | 387.5 |
| 100 measured | 94.0 | 168.0 | 224.0 | 279.0 | — | — |
| 100 bound | 94.0 | 193.0 | 292.5 | 391.5 | — | — |
| 150 measured | 144.0 | 264.0 | 310.0 | — | — | — |
| 150 bound | 144.0 | 294.0 | 442.5 | — | — | — |
| 200 measured | 195.0 | 346.0 | — | — | — | — |
| 200 bound | 194.0 | 394.0 | — | — | — | — |

Median deviation (measured − bound), in tokens, with the 5th–95th percentile range:
$W=50$: state 2 −9 [−17,−1], state 3 −29.5 [−44,−11], state 4 −50 [−69,−17],
state 6 −107, state 8 −180; $W=100$: state 2 −27 [−42,−6], state 3 −70
[−96,−43], state 4 −111 [−138,−60]; $W=150$: state 2 −32, state 3 −129;
$W=200$: state 2 −50 [−77,−16]. For every cascade state ($l\ge2$), 100% of
resolved directions die at or before the bound except $W=50$ state 2, where
one direction is one token late ($98.7\%$ at or before). State 1 is the
equality of §2, so its ±1–3 token jitter puts a minority of points on either
side of the line.

**What it shows.** The bound is respected everywhere, but it is loose at the
observable level: the measured slope is 61 tokens per state at $W=100$
(against 99 for the bound) and 151 at $W=200$ (against 199), and at $W=50$ it
saturates: states 6 and 8 differ by only 26 tokens while the bound grows by
another 98. This is consistent with the raw traces: block $\ell$'s response
is a sum over block-$(\ell-1)$ states that are already decaying inside the
sliding window, so the last surviving path carries a signal that is far below
the fp32 floor before it finally disappears. The data establish the early
floor crossing; they do not by themselves separate "already below the floor"
from "the last path is weaker than the bound allows".

**What it does not show.** It does not measure the exact-zero cascade on the
14B (the floor hides the last contribution); it measures the time to reach
the floor, which is an observable upper bound on the true death but not a
physical constant (it depends on the $10^{-6}$-relative residue). The 1.5B
provides the exact-zero counterpart (§8). The pending 1500-token run is the
only way to follow states beyond $l=4$ at $W=100$ (or beyond $l=2$ at
$W=200$) without the horizon cutting the measurement.

---

## 4. Figure `ds14win_support.png` — moving the support moves the death step exactly

![support](figures/ds14win_support.png)

**Setup.** Three support-scan runs at $W=100$: 240 directions restricted to
prompt positions $[0,22)$, $[22,44)$ and $[44,65)$, 300 tokens, the same
prompt, the same fixed tokens, the same direction construction. Each
direction's $p_{\rm last}$ is computed from its own restricted coordinates;
prediction $p_{\rm last}+35$ (because $W-P=35$). Points: one per direction,
colour by support. Everything except the support is held fixed — this is the
controlled version of the law.

**Axes.** $x$: predicted death step $p_{\rm last}+35$. $y$: measured death
step (state 1, same floor rule). Identity line dashed. Equal aspect.

**Numbers.**

| support | $p_{\rm last}$ | predicted | resolved | exactly on line | within ±2 | max deviation |
|---|---|---:|---:|---:|---:|---:|
| $[0,22)$ | 18–21 | 53–56 | 240 / 240 | 80 | 240 | +2 |
| $[22,44)$ | 40–43 | 75–78 | 240 / 240 | 140 | 240 | +2 |
| $[44,65)$ | 61–64 | 96–99 | 240 / 240 | 240 | 240 | 0 |

**What it shows.** All 720 directions are resolved and all lie within two
tokens of $p_{\rm last}+35$; the latest-support group is exact for 240/240.
The three clusters are disjoint and separated by exactly the $p_{\rm last}$
shift (about 22 and 21 tokens). This rules out the
alternative explanation that the death step is set by the window width alone
or by the token index: the width and the prompt are identical; only the
support moved, and the death step moved with it one-for-one.

**What it does not show.** It is a single window width ($W=100$); it does not
test the $(W-P)$ term (that is the six-window scan in §2). The +1/+2 tokens
in the early groups are floor-crossing jitter, not a second mechanism.

---

## 5. Figure `ds14win_plen.png` — prompt-length scan: PENDING

![prompt length](figures/ds14win_plen.png)

**Setup.** The intended figure: for `short_prompt` ($P\approx6$) and
`long_prompt` ($P\approx250$) at each $W$, measured state-1 death versus
$p_{\rm last}+(W-P)$, with the $P<W$, $P\approx W$ and $P>W$ regimes
annotated. The 12 files
(`results_big/ds14/a1_short_prompt_w{W}_fixed/a1.npz` and
`a1_long_prompt_w{W}_fixed/a1.npz`, $W=50\ldots500$) are **not yet local**.
The figure is a placeholder; `figs_window.py` detects and processes them
automatically when they arrive.

**What the completed scan would test, and what is already known.** The
formula predicts three regimes. For $P<W$ every prompt position is visible at
$t=0$ and dies at $p+(W-P)$, so the deaths spread over $W-P$ steps and are
late. For $P\approx W$ the spread collapses to a few steps around $W-P\approx0$.
For $P>W$ the earliest $P-W$ prompt positions are already outside the window
at $t=0$: their formula value is negative, so they are dead from the first
generated step, while the remaining positions die at $p+(W-P)\in[0,W-1]$.
The already-available $W=50$ run is exactly the $P>W$ case ($P=65>50$): its
deaths span $40$–$49$, and the baseline margin trace differs from the
no-clipping reference from $t=0$ (not from $W-P$), because the first 16
prompt tokens are invisible from the start. This is the qualitative
prediction the pending short/long scan will quantify.

**What it does not show.** Everything quantitative: the actual short and long
$P$, the exact death distribution, and whether the flush survives when the
prompt is much shorter or much longer than the window.

---

## 6. Figure `ds14win_response_shape.png` — a flush surge one step after $t=W-P$

![response shape](figures/ds14win_response_shape.png)

**Setup.** For each window, $\lVert\delta h_t^{47}\rVert/r$ at the last saved
state (layer 47), median over the 240 directions at each step; the band is
the 25–75% range over directions, drawn for $W=100$ only to keep the panel
readable. The dashed grey curve is the $W=4096$ no-clipping reference on the
same tokens (32 directions). Vertical dotted lines mark $t=W-P$ for each
window. Log $y$.

**Axes.** $x$: generated step $t$, 0–500. $y$: response norm per unit radius
at layer 47 (median over directions, log). Free $y$-range $3\times10^{-2}$ to
$30$; the curves span three decades.

**Numbers.** Ratio of the median response in the ten steps after the first
clipped position to the ten steps before,
$\Phi_W=\mathrm{median}_{t\in[t^*+1,t^*+10]}\rho_t/
\mathrm{median}_{t\in[t^*-10,t^*-1]}\rho_t$:

| W | $t^*=W-P$ | $\Phi_W$ | per-direction median ratio |
|---:|---:|---:|---:|
| 50 | −15 (prompt does not fit) | — | — |
| 100 | 35 | 2.19 | 2.34 |
| 150 | 85 | 2.00 | 2.02 |
| 200 | 135 | 2.59 | 2.33 |
| 300 | 235 | 3.95 | 3.58 |
| 500 | 435 | 2.41 | 2.36 |
| 4096 (ref) | — | no kink | — |

The 1000-direction true-window-100 core run gives $\Phi=2.20$ by the same
pooled definition (2.28 by the per-direction median); the 1.5B window-100
value is 1.73 (quoted from the earlier 1.5B analysis).

**What it shows.** Each window produces a visible upward step exactly at its
own $t=W-P$, not at a fixed token: 35, 85, 135, 235, 435. The no-clipping
reference has no such step anywhere. The surge is not a small effect — it
doubles the last-layer response at $W=100$ and nearly quadruples it at
$W=300$; the six values ($W=100\ldots500$) range from 2.00 to 3.95 and are
not monotonic in $W$. The $W=50$ curve starts
high (median 4.6) because its window already excludes prompt positions
0–15 at $t=0$, so the baseline itself is different from the first step.

**What it does not show.** It does not establish the mechanism of the surge
(renormalisation after dropping one key versus a burst of the surviving
paths); both are consistent with these traces. It also does not show whether
the surge changes any decision — that is the margin panel, and there the
changes are at other steps. The absolute levels are not comparable across
windows before $t=W-P$: a smaller window changes the baseline forward pass
itself, so the same direction produces a different response even while no
perturbed prompt position has yet left.

---

## 7. Figure `ds14win_margins_by_w.png` — the window changes the decision landscape itself

![margins by W](figures/ds14win_margins_by_w.png)

**Setup.** The same 500 fixed tokens and the same 240 directions under each
window. Panel (a): median and minimum top-1 minus top-2 gap over the 500
steps versus $W$ (log $x$); dashed/dotted grey lines are the $W=4096$
reference median and minimum. Panel (b): the full margin trace of each run,
one light line per $W$, with the $t=35=W-P$ line for $W=100$.

**Axes.** (a) $x$: window width in tokens (log, labelled with the six
values); $y$: margin in nats (log). (b) $x$: generated step 0–500; $y$:
margin in nats (log).

**Numbers.**

| W | median margin | minimum margin (step) | steps < 0.1 nats | steps < 0.5 nats | first step differing from W=4096 |
|---:|---:|---:|---:|---:|---:|
| 50 | 1.135 | $8.20\times10^{-5}$ (122) | 28 | 120 | 0 |
| 100 | 1.357 | $1.030\times10^{-4}$ (21) | 31 | 116 | 36 |
| 150 | 1.507 | $1.030\times10^{-4}$ (21) | 16 | 113 | 86 |
| 200 | 1.471 | $1.030\times10^{-4}$ (21) | 25 | 108 | 136 |
| 300 | 1.936 | $1.030\times10^{-4}$ (21) | 20 | 86 | 236 |
| 500 | 2.463 | $1.030\times10^{-4}$ (21) | 16 | 62 | 436 |
| 4096 | 2.989 | $1.030\times10^{-4}$ (21) | 15 | 63 | — |

The early ties are preserved to the bit for every $W\ge100$: the two razor
gaps are $m_{21}=1.02997\times10^{-4}$ and $m_{19}=1.1293\times10^{-3}$,
identical in all five runs and in the reference (these are the same two ties
seen in the 1000-direction full/window comparison). For $W\ge100$ the margin
trace is bit-identical to the reference through $t=W-65$ and first differs at
$t=W-64$; for $W=50$, where $P>W$, it differs from $t=0$.

**What it shows.** Two separate effects. First, the mask changes the logits
on *identical* tokens: median margin rises overall from 1.135 at $W=50$ to
2.463 at $W=500$, approaching the reference 2.989, and the count below 0.5
nats falls from 120 to 62. Second, the early landscape is untouched for
$W\ge100$: the $t=21$ and $t=19$ ties survive to the bit, because both lie
before the first clipped position. At $W=50$ the prompt does not fit and the
first clipped position is $t=0$: both ties are gone ($m_{21}=0.382$,
$m_{19}=0.119$ nats), and the minimum margin moves to a new step, $t=122$.
So the window does not merely attenuate propagation of a perturbation; it
reshapes which decisions are close.

**What it does not show.** These are margins of a *teacher-forced* baseline
that is identical in all runs, so no text divergence contaminates the
comparison — but also no free-running consequence is measured. The margin
distribution of one 500-token continuation is not the margin distribution of
the model, and the count "below 0.5 nats" is not a flip count. The reference
has 32 directions, not 240, although margins do not depend on directions.

---

## 8. Comparison with the 1.5B law

The 1.5B flush sweep (`chaos_llm_2026/results/local_data/flush_sweep/w*`,
read-only) used the same 65-token prompt, $r=0.1$, 4 directions, 500 tokens,
and exact zeros are attained there (the baseline and perturbed passes are
numerically identical at zero response).

| quantity | 1.5B | 14B |
|---|---|---|
| state corresponding to block 0 | saved state 0 | saved state 1 |
| exact-zero death, $W=100/150/200/300/500$ | 78 / 128 / 178 / 278 / 478 | not observable (floor $3\times10^{-5}$) |
| predicted $p_{\rm last}+(W-P)$ | 78 / 128 / 178 / 278 / 478 | matched exactly at $W=50,100$; ±1–3 tokens at $W=150$–$300$ |
| exact-zero cascade, state 2, $W=100/150/200$ | 177 / 277 / 377 | floor crossing 27–50 tokens early (median) |
| exact-zero cascade, state 3, $W=100/150$ | 276 / 426 | floor crossing 70–129 tokens early (median) |
| flush ratio at $W=100$ | 1.73 | 2.19 (fixed 240 dirs), 2.20 (core 1000 dirs) |

The 1.5B numbers satisfy both formulas *exactly*: 78 = 43+35, 177 = 78+99,
276 = 78+198, 277 = 128+149, 426 = 128+298, 377 = 178+199, with
$p_{\rm last}=43$ and $P=65$. On the 14B the first formula is confirmed at
the floor-crossing level and the second is confirmed as an upper bound. The
two statements are consistent: the cascade's exact-zero time is the moment
the last, already tiny contribution vanishes; on the 14B that contribution
is below the fp32 batch residue, so the observable crossing must be earlier.
The 14B's *lower* numerical precision therefore costs the exact-zero test,
not the law.

---

## 9. What is pending

Two items from the brief are not yet available locally:

1. **Prompt-length scan** (`a1_short_prompt_w*_fixed`, `a1_long_prompt_w*_fixed`):
   figure placeholder `ds14win_plen.png`; the script will populate it and add
   the numbers to `ds14win_numbers.json` when the files appear. §5 states the
   three predicted regimes and the already-observed $P>W$ case ($W=50$).
2. **1500-token cascade** (`a1_inter_w100_cascade`): panel (b) of
   `ds14win_cascade.png` is a placeholder; it will show states 0–14 at
   $W=100$ with the same bound line. It is the only way to follow the
   cascade beyond the plotted range without the 500-token horizon; the
   present 14B cascade is plotted to $l=4$ for $W=100$ and $l=2$ for $W=200$.

Everything else in the brief is complete. The script `scripts_spark/figs_window.py`
detects each file independently, so a partial later arrival produces the
corresponding figure without touching the rest.
