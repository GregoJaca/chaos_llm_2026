p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
anchor = "## 8.6 Negative results and traps found in this analysis"
new = r"""## 8.5b How far does the boundary prediction hold? (large $r$, autoregressive)

![threshold large r](figures/threshold_largr.png)

**Data.** DeepSeek, two prompts, 20 directions, radii 0.5–2.0. $x$: first flip predicted by the
signed margin rule $m_t+\delta m_t<0$ from the teacher-forced run at the same $r$. $y$: first token
that actually differs in the autoregressive run.

**Reading.** On the robust prompt (right) the prediction is **exact 20/20, 19/20, 19/20, 20/20** even
at $r=2$ — the rule survives far outside the linear regime there. On the fragile prompt (left) it
degrades: 10/20 at $r=0.5$, 1–6/20 at $r\ge1.2$, with predictions clustered at $t\approx0$ and
observations scattered out to $t=50$.

**Why it fails exactly there.** At these radii the margin response has saturated (§N3), so the
teacher-forced $\delta m$ is no longer a faithful tangent; and on a fragile prompt a small error in
the predicted time moves the first flip by tens of tokens. The failure is therefore *informative*:
it marks the boundary of the linear theory, and it is the same prompt-dependence that makes the
stable-chaos statistics prompt-specific.

**One caveat on the stored `pred_T`:** it is computed from $|\\delta m|$, i.e. without the sign. It
therefore predicts a flip whenever the margin changes a lot, even if it grows. The signed rule above
is the correct one (and is what the 318/320 result used). Do not use `pred_T` directly.

## 8.6 Negative results and traps found in this analysis"""
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
