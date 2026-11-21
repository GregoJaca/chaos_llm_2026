# Part window program. The 14B death law is exact to the token for prompts of 8 and 190 tokens; the cascade's bitwise-zero support is exact over 14 states while its 1e-6 crossing runs early with depth

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
A 1500-token cascade run (8 directions, $W=100$, 15 saved states) now follows
the law up the stack and sharpens both statements. The threshold death tracks
the law **only at $l=1$**: medians run 92, 169, 226, 269.5, … , 454.5 for
$l=1\ldots14$, a slope of 26 tokens per state against the predicted 99, and
the median gap grows monotonically from $-22$ at $l=2$ to $-924.5$ at $l=14$;
all 112 resolved direction-states die at or before their bound, so the
500-token early-death pattern is confirmed and extended. The same run also
makes the literal test observable: at `batch_dirs=2` the perturbed row becomes
bitwise identical to the baseline once the prompt is fully outside the window,
and the first exact zero lands on $t^*_l$ for all 8 directions at all 14
states ($92,191,\ldots,1379$; 112 of 112). The last nonzero path therefore
survives to the combinatorial bound exactly, but its amplitude crosses the
$10^{-6}$-relative floor hundreds of tokens earlier — the bound is attained
only by the support, not by the signal.
Finally, the window changes the decision landscape itself on identical
tokens: the two early razor ties are preserved bit-for-bit for every
$W\ge100$, while the number of margins below 0.5 nats nearly doubles from 62
($W=500$, approaching the no-clipping reference) to 116–120 at $W=100/50$;
at $W=50$ the prompt does not fit and the $t=21$ tie is destroyed. Every
window shows a flush surge one step after $t=W-P$: x2.19 at $W=100$ (the
1000-direction core run gave x2.20), x2.00 at $W=150$, x2.59 at $W=200$,
x3.95 at $W=300$, x2.41 at $W=500$ (the 1.5B reference was x1.73).
Two further scans close the remaining free parameters. A prompt-length scan
(the same construction with an 8-token prompt and a 190-token prompt, 120
directions each, $W=50,100,200,300,500$) gives the same equality:
$t^*=p_{\rm last}+(W-P)$ predicts the death step with $P$ read from the
tokenizer, not guessed. With $P=8$, all 240 directions over $W=50,100$ are
exact, and at $W=200$ 100 of the 112 resolved directions are exact with the
other 12 one step late. With $P=190$ and $P>W$ ($W=50,100$) the prompt does
not fit at $t=0$, yet 180 of 240 directions are exact and all 240 are within
two tokens; for $W=200,300$ ($P<W$) the deaths again sit on the line (all 120
at $W=300$ exactly on it). The only failures are at the 300-step horizon:
$W-P=292$ ($P=8,W=300$) and $W-P=310,492$ ($W=500$) put the predicted death
at or beyond the last step, and then the run, not the model, sets the limit —
the floor rule degenerates there and its "deaths" are crossings of a
contaminated floor. Margins of the fixed tokens follow the same onset rule as
the 65-token prompt (bit-identical through $t=W-P$, first change at $W-P+1$)
and, for the short prompt, the same monotone approach to the full-attention
value. The 1500-token cascade run is complete and supplies the $l\ge2$
literal zeros (§3).

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

## 3. Figure `ds14win_cascade.png` — the threshold crossing runs early with depth; the bitwise-zero support sits exactly on the bound

![cascade](figures/ds14win_cascade.png)

**Setup.** Panel (a): for each window, the median over directions of the
measured death at saved state $l$ (solid, markers) and of the predicted bound
$t^*_l$ computed from each direction's own $t^*_1$ (dashed, same colour).
$W=100$ and $W=200$ are drawn bold, the other windows thin for context.
Panel (b): the 1500-token cascade run
(`results_big/ds14/a1_inter_w100_cascade`), $W=100$, $P=65$, 8 directions,
15 saved states (0–14; states 1–14 used), tokens fixed from the 1500-token
$W=4096$ baseline. The 8 directions share one simplex block and therefore the
same 20 prompt coordinates: $p_{\rm last}=57$ and $t^*_1=92$ for every
direction, so the bound is a single line. The death rule is exactly the one of
§1.3. The run used `batch_dirs=2` rather than 16, and that matters: the last 60
steps are bitwise zero at every state, so the per-state floor is the rule's
$10^{-6}\times$median-norm cap ($2.95\times10^{-5}$ at $l=1$ rising to
$9.95\times10^{-5}$ at $l=14$), not a measured residue. Panel (b) therefore
shows two observables: the same threshold death as panel (a) (filled, with the
interquartile range over directions), and the first step at which `dh_norm` is
**bitwise zero** (open circles), which this run makes observable.

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

The 1500-token run (panel b), state by state. Bound $t^*_l=92+99(l-1)$;
"death" is the §1.3 threshold crossing; "zero" is the first bitwise-zero step:

| $l$ | bound | threshold death | deviation (median [min,max]) | first zero |
|---:|---:|---:|---:|---:|
| 1 | 92 | 92.0 | 0 [0,0] | 92 |
| 2 | 191 | 169.0 | −22 [−37,−11] | 191 |
| 3 | 290 | 226.0 | −64 [−93,−51] | 290 |
| 4 | 389 | 269.5 | −119.5 [−159,−110] | 389 |
| 5 | 488 | 282.5 | −205.5 [−238,−185] | 488 |
| 6 | 587 | 335.5 | −251.5 [−284,−225] | 587 |
| 7 | 686 | 388.0 | −298 [−337,−253] | 686 |
| 8 | 785 | 401.0 | −384 [−426,−351] | 785 |
| 9 | 884 | 415.5 | −468.5 [−516,−425] | 884 |
| 10 | 983 | 429.5 | −553.5 [−597,−519] | 983 |
| 11 | 1082 | 447.0 | −635 [−696,−588] | 1082 |
| 12 | 1181 | 447.0 | −734 [−795,−687] | 1181 |
| 13 | 1280 | 447.0 | −833 [−894,−786] | 1280 |
| 14 | 1379 | 454.5 | −924.5 [−970,−874] | 1379 |

All 8 directions are resolved at every state (112 of 112 direction-states),
and all 112 threshold deaths are at or before the bound.

**What it shows.** The threshold death is an amplitude landmark, not the
cascade. On the 1500-token run the measured medians move from 92 to 454.5 over
14 states, a slope of 26 tokens per state, while the bound moves by 99; the
median gap is already −22 at $l=2$, −64 at $l=3$, −120 at $l=4$, and −924.5 at
$l=14$. The 500-token early-death finding is therefore confirmed and extended
rather than revised: for $W=100$, state 2 is −22 here against −27 in the
500-token run, state 3 −64 against −70, state 4 −120 against −111 (different
runs and direction sets, same effect). From $l=7$ the threshold medians
saturate — 388, 401, 415.5, 429.5, 447, 447, 447, 454.5 — moving 67 tokens
while the bound moves 693, because deeper states start with a larger peak
($13\times$ the floor at $l=1$, $577\times$ at $l=14$) but their late-time
decay reaches the $10^{-6}$ cap at nearly the same step. That saturation is a
property of the threshold, not a physical time.

The literal test behaves the opposite way. At `batch_dirs=2` the perturbed row
becomes bitwise identical to the baseline once no prompt position is inside
any window, so `dh_norm` is exactly 0 from $t^*_l$ onward; the first zero is
$t^*_l$ for all 8 directions at all 14 states. This is partly structural — a
row with no prompt in its receptive field must compute the same function as
the baseline — but it is also a measurement: at every step before the bound
the response is nonzero, so the last path genuinely survives to the
combinatorial bound. As a statement about the support, the cascade is an
equality over the whole stack; as a statement about the signal, it is a loose
upper bound. The last nonzero values are far below the $10^{-6}$ cap (for
direction 0 at state 2 they run $8.4\times10^{-5}$ at $t=175$ down to
$1.7\times10^{-6}$ at $t=190$: that direction's support outlives its threshold
crossing by 12 steps; the median gap at $l=2$ is 22 and at $l=14$ it is 925).

**What it does not show.** The exact-zero equality is support geometry, not an
amplitude law: it says when the last path is cut, not how strong it is. The
residue issue is also batch-dependent, not resolved: the 500-token runs
(`batch_dirs=16`) carry a $\approx10^{-6}$-relative residue at the tail and
never reach a bitwise zero, while this `batch_dirs=2` run reaches exact zeros
at every state. The cause is presumably a kernel / reduction-order difference
between batch shapes, but it was not isolated here, so threshold death times
should not be compared across batch sizes to better than a token or two. The
run has one prompt, one window, and one simplex block (all directions share
$p_{\rm last}=57$), so it does not scan direction geometry. Finally it does
not follow states beyond $l=14$ (block 14 of 48); extrapolating the 26-token
slope would put the threshold crossing of the last block near $t\approx1550$,
beyond even this 1500-token horizon, but the saturation makes that
extrapolation unsafe.

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

## 5. Figure `ds14win_plen.png` — prompt length only shifts the death step

![prompt length](figures/ds14win_plen.png)

**Setup.** Two prompt lengths, each with its own no-clipping reference. The
short prompt is `Explain how tides work.`, **$P=8$ tokens**; the long prompt
is a review of climate physics, **$P=190$ tokens** (both read from
`meta.json`; the brief's estimates of $\approx6$ and $\approx250$ were not
exact). For each prompt, five true sliding-window runs
$W=50,100,200,300,500$ (no $W=150$), 120 directions at $r=0.1$, 300
teacher-forced tokens fixed from that prompt's own $W=4096$ run
(`a1_short_prompt_w4096`, `a1_long_prompt_w4096`), the same 16 saved states
and the same floor/death rule as §1.3. The five windows of a prompt share
identical `gen_ids` with their reference (checked), so there is no text
divergence. A direction's $p_{\rm last}$ is the largest perturbed prompt
position. Directions are built in blocks of 20 that share their 20 prompt
positions (only the coordinates inside them differ), so $p_{\rm last}$ takes
one value per block: 1 block at 6 and 5 blocks at 7 for the short prompt
(20 directions at 6, 100 at 7), and 164, 168, 173, 174, 182, 186 (20
directions each) for the long prompt. Panel (a) is the short
prompt, panel (b) the long one; colour is the window, and circle/triangle
distinguishes the prompt, but the panels never mix. Only runs with at least
one resolved direction are plotted.

**Axes.** $x$: predicted death step $p_{\rm last}+(W-P)$, with $P$ from
`meta.json`. $y$: measured state-1 death (first step at the floor). Dashed
identity line; equal aspect. Panel (b) marks the 300-step horizon
($t=299$): a predicted death beyond it cannot be observed in these runs.

**Numbers.**

| prompt | $P$ | $W$ | $W-P$ | $p_{\rm last}$ | $t^*=p_{\rm last}+(W-P)$ | died at $t^*$ | resolved | exact | within ±2 | max dev |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| short | 8 | 50 | +42 | 6–7 | 48–49 | 120 / 120 | 120 | 120 | 120 | 0 |
| short | 8 | 100 | +92 | 6–7 | 98–99 | 120 / 120 | 120 | 120 | 120 | 0 |
| short | 8 | 200 | +192 | 6–7 | 198–199 | 120 / 120 | 112 | 100 | 112 | +1 |
| short | 8 | 300 | +292 | 6–7 | 298–299 | at horizon | 0 | — | — | — |
| short | 8 | 500 | +492 | 6–7 | 498–499 | none | 0 | — | — | — |
| long | 190 | 50 | −140 | 164–186 | 24–46 | 120 / 120 | 120 | 100 | 120 | +1 |
| long | 190 | 100 | −90 | 164–186 | 74–96 | 120 / 120 | 120 | 80 | 120 | +2 |
| long | 190 | 200 | +10 | 164–186 | 174–196 | 120 / 120 | 27 | 26 | 27 | +1 |
| long | 190 | 300 | +110 | 164–186 | 274–296 | 120 / 120 | 5 | 5 | 5 | 0 |
| long | 190 | 500 | +310 | 164–186 | 474–496 | none | 0 | — | — | — |

`resolved`, `exact`, `within ±2` and `max dev` use the §1.3 resolution guard
(pre-death signal at least $10\times$ the floor and 11 steps left); `died at
$t^*$` counts all 120 directions regardless of the guard. The two rows with
0 resolved are horizon statements, not measured deaths.

*Short prompt.* With $P=8$, $W=50$ gives $t^*=p_{\rm last}+42=48$–$49$, not
$p_{\rm last}+44$: the window regime is indeed $W\gg P$, and the predicted
offset is larger than guessed. Every one of the 120 directions dies exactly
on the line at $W=50$ and at $W=100$ (240/240 exact, no scatter). At $W=200$
all 120 die at 199: 100 of the 112 resolved directions exactly and the other
12 one step late (their $t^*=198$). At $W=300$ the predicted death is
298–299 — the last
one or two steps of the run — so the 30-step excursion test cannot be
applied; the per-state floor estimate is also contaminated ($4.2\times
10^{-4}$ instead of $3.0\times10^{-5}$, because the response never reaches
the floor inside the run), and the 120 "deaths" the rule returns are not
usable. At $W=500$ the prompt would leave the window at $t=492$–$499$,
beyond the 300-step run: the state-1 response is still $1.1\times10^{-3}$ at
$t=299$ (literal minimum over $t$: $7.3\times10^{-4}$), so no death occurs;
the 102 crossings reported by the rule are crossings of the contaminated
floor ($7.7\times10^{-4}$). In both cases the horizon, not the law, sets the
limit.

*Long prompt, $P>W$ ($W=50,100$).* The prompt does not fit: at $t=0$ the
window already excludes prompt positions $0\ldots140$ ($W=50$) or
$0\ldots90$ ($W=100$), so the model sees only the last $W-1$ prompt
positions plus generated tokens. The law as stated nevertheless applies
without correction, because it refers only to $p_{\rm last}$: every
direction has $p_{\rm last}=164$–$186>P-W$, hence its last perturbed
position is visible at $t=0$ and $t^*=p_{\rm last}-140=24$–$46$ (or
$p_{\rm last}-90=74$–$96$) is positive. Measured: 100 of 120 directions
exact at $W=50$ (the other 20 are one token late) and 80 of 120 exact at
$W=100$ (20 one step late and 20 two steps late); all 120 are within two
tokens in both runs. There is **no systematic early, late, or absent death**:
the deaths sit on the line. What the non-fitting window changes is the amplitude
— the state-1 response starts at $3.4\times10^{-3}$ ($W=50$) or
$7.0\times10^{-3}$ ($W=100$) instead of $6.8\times10^{-3}$ ($W=500$) because
most of each direction's positions are already invisible — and the flush
kink: $t=W-P=-140$ or $-90$ lies before the first generated token, so the
flush happens inside the prompt and there is no surge in the generated trace
(the surge ratios are undefined here).

*Long prompt, $P<W$.* At $W=200$ ($W-P=10$) all 120 directions die at
175–196; 100 are exact and 20 (the $p_{\rm last}=164$ block, $t^*=174$) one
token late, and 26 of the 27 resolved are exact. Only 27 pass the resolution
guard, because the pre-death response
over $[t^*-10,t^*]$ is $2.3\times10^{-4}$, below the $10\times$floor
threshold ($3.0\times10^{-4}$): the surviving perturbed token is already
$\approx200$ positions behind the query. At $W=300$ ($W-P=110$) all 120
directions die exactly at $274$–$296$ — the death groups are precisely the
six $p_{\rm last}$ groups shifted by 110 — though only 5 are "resolved":
the group at 292/296 is within 11 steps of the end, and the pre-death
response is $1.9\times10^{-4}$ against a $4.3\times10^{-4}$ threshold. The
drop at $t^*$ itself is $1.6\times10^{-4}\to5.4\times10^{-5}$, so the death
is real; the strict guard rejects it, it is not a different mechanism.
$W=500$ is the no-op case: $W\ge P+T=490$, so no prompt position ever leaves
the window at any of the 300 steps. The state-1 response is still
$2.2\times10^{-3}$ at $t=299$, the floor estimate ($6.9\times10^{-4}$) is
signal, 42 directions never cross it and 78 cross only spuriously, and the
margin trace is bit-identical to the $W=4096$ reference.

**What it shows.** Prompt length only shifts $t^*$. The equality holds with
$P$ read from the tokenizer: 240/240 short-prompt directions exact at
$W=50,100$, 180/240 long-prompt directions exact over the same windows and
240/240 within two tokens. The regime that looked most likely to break the
law — the prompt not fitting the window — does not break it, because the law
concerns the last perturbed position, which is still inside the window when
$p_{\rm last}>P-W$; deaths are neither early nor late. The law fails only
where it cannot be measured: when $W-P$ puts $t^*$ at or beyond the horizon.

### 5.1 Margins of the fixed tokens

The same fixed-token comparison as §7, each prompt against its own $W=4096$
reference:

| prompt | $W$ | median margin | minimum (step) | steps $<0.1$ | steps $<0.5$ | first step differing from 4096 |
|---|---:|---:|---:|---:|---:|---:|
| short | 50 | 1.542 | 0.0113 (61) | 11 | 72 | 43 |
| short | 100 | 2.134 | 0.0198 (168) | 10 | 57 | 93 |
| short | 200 | 2.801 | 0.0060 (297) | 9 | 49 | 193 |
| short | 300 | 4.474 | 0.0087 (222) | 8 | 32 | 293 |
| short | 500 | 4.507 | 0.0087 (222) | 8 | 31 | none (identical) |
| short | 4096 | 4.507 | 0.0087 (222) | 8 | 31 | — |
| long | 50 | 1.635 | 0.0033 (71) | 12 | 58 | 0 |
| long | 100 | 1.197 | 0.0367 (94) | 9 | 66 | 0 |
| long | 200 | 0.722 | 0.0014 (167) | 30 | 114 | 11 |
| long | 300 | 1.346 | 0.0016 (270) | 17 | 72 | 111 |
| long | 500 | 2.471 | 0.0020 (165) | 9 | 45 | none (identical) |
| long | 4096 | 2.471 | 0.0020 (165) | 9 | 45 | — |

For the short prompt the $W$-dependence is the same in kind as for the
65-token interstellar prompt: the trace is bit-identical to the reference
through $t=W-P$ and first differs at $t=W-P+1$ (43, 93, 193, 293 for
$W=50,100,200,300$), the median margin rises monotonically toward the
full-attention value (1.542 → 2.134 → 2.801 → 4.474 → 4.507, reference
4.507), and the count below 0.5 nats falls (72 → 57 → 49 → 32 → 31, reference
31). At $W=500$, $W\ge P+T=308$, so the window is a no-op and the trace is
bit-identical to full attention. The difference from the interstellar scan is
only in level — that scan goes 1.135 → 2.463 with reference 2.989 and 120 → 62
near-ties against 63 — because the prompt and the continuation differ; the
onset rule and the direction of the change are identical.

The long prompt does **not** follow that pattern. $W=50$ and $W=100$ differ
from $t=0$ (the prompt does not fit), and the median margin is not monotone
in $W$: 1.635, 1.197, 0.722, 1.346, 2.471, with the minimum at $W=200$
(114 of 300 steps below 0.5 nats). So "a narrower window flattens the
decisions" transfers from the short and interstellar prompts to the long one
only in the onset rule (first change at $W-P+1$, or at $t=0$ when $P>W$), not
in the sense of the change. $W=500$ again reproduces the reference
bit-for-bit.

**What it does not show.** The scan has no window within a few tokens of $P$:
with $P=8$ and $P=190$ against $W\in\{50,100,200,300,500\}$, the $P\approx W$
regime is not sampled. It cannot observe the death of prompt positions with
$p<P-W$ when the prompt does not fit: those are already invisible at $t=0$,
the law assigns them $t^*<0$, and no measurement here can confirm it. It
cannot separate the floor crossing from a true zero (the same fp32 residue as
§1.3; 0 of 1200 plen directions reach exactly 0 or $10^{-6}$), and the
$W=300$/$W=500$ rows are horizon-limited statements about the floor rule, not
measurements of the law.

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
| exact-zero death, $W=100/150/200/300/500$ | 78 / 128 / 178 / 278 / 478 | not observable in the 500-token runs (floor $3\times10^{-5}$); 92 at $W=100$ in the 1500-token run |
| predicted $p_{\rm last}+(W-P)$ | 78 / 128 / 178 / 278 / 478 | matched exactly at $W=50,100$; ±1–3 tokens at $W=150$–$300$ |
| exact-zero cascade, state 2, $W=100/150/200$ | 177 / 277 / 377 | 191 at $W=100$ (1500-token run; 8/8 directions); threshold crossing 22–50 tokens early (median) |
| exact-zero cascade, state 3, $W=100/150$ | 276 / 426 | 290 at $W=100$ (1500-token run; 8/8); threshold crossing 64–129 tokens early (median) |
| exact-zero cascade, state 14, $W=100$ | — | 1379 (8/8 direction-states, exactly $92+13\cdot99$) |
| flush ratio at $W=100$ | 1.73 | 2.19 (fixed 240 dirs), 2.20 (core 1000 dirs) |

The 1.5B numbers satisfy both formulas *exactly*: 78 = 43+35, 177 = 78+99,
276 = 78+198, 277 = 128+149, 426 = 128+298, 377 = 178+199, with
$p_{\rm last}=43$ and $P=65$. On the 14B the first formula is confirmed at
the floor-crossing level and the second is confirmed as an upper bound at the
threshold level. The 1500-token run then produces exact zeros at every state:
its first zero lands on the bound for all 112 direction-states, from 92 to
1379, so the cascade is an equality for the *support*. The two statements are
consistent: the cascade's exact-zero time is the moment the last, already tiny
contribution vanishes; in the 500-token runs that contribution sits below the
fp32 batch residue, so only the earlier threshold crossing is observable,
while in the 1500-token run (`batch_dirs=2`) it is not hidden and the literal
zero appears at the bound. The 14B's *lower* numerical precision therefore
costs the exact-zero test only when the batch residue is present, not the law.

---

## 9. What remains

Nothing in the brief is now missing locally. The 1500-token cascade
(`a1_inter_w100_cascade`) is processed and fills panel (b) of
`ds14win_cascade.png`; it resolves the cascade to $l=14$ and supplies the
literal zeros that the 500-token runs could not observe (§3). It does not
change the $W=200$ picture, which still stops at $l=2$ because the run has
only 500 tokens.

The prompt-length scan of §5 is complete: all ten runs
($P=8$ and $P=190$; $W=50,100,200,300,500$) are local and processed. The scan
does not include $W=150$: with $P=8$ or $P=190$ it would only repeat the
$P<W$ or $P>W$ regime already sampled and would not reach $P\approx W$.

Everything else in the brief is complete. The script
`scripts_spark/figs_window.py` detects each file independently, so a partial
later arrival produces the corresponding figure without touching the rest.
