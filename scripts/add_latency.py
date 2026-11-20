p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()
anchor = "## 7. Result 4 — where the response comes from, and how linear it is"
new = r"""## 6b. The perturbation waits: latency before the flip

![latent](figures/state_kick_latent_validated.png)

**What is plotted.** (a) Predicted (teacher-forced rule) against observed (free run) first changed
token, five perturbations. (b) Token agreement with the unperturbed run along the continuation for one
direction. (c) The same for 24 directions at $s=16$.

**What it shows.** A perturbation at generated position $s=16$ changes the text at **$t=123$, 107
tokens later**; at $s=50$ it acts at $t=143$, 93 tokens later. All five predictions are exact. After
the flip the two continuations never re-converge (agreement falls to ~40% and stays there).

**Why this matters.** It is the operational meaning of "sensitivity without stretching": the
perturbation does not grow (§4), does not act immediately, and does not need to be large — it waits
until the model reaches a step where the top-two logit gap is small, and then it decides the token.
The waiting time is set by the *model*, not by the perturbation size (over the range tested).

**What is not shown.** The apparent decay of agreement after the flip is not a physical decay: it
saturates at a floor because continuations re-align on repetitive text. Do not fit it.

## 7. Result 4 — where the response comes from, and how linear it is"""
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
