p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()

# add the kernel-across-models figure after the single-model kernel
anchor = "### 7.2 Is the kernel a valid superposition? (measured)"
new = r"""### 7.1b The kernel in all five models

![kernel models](figures/transport/kernel_models.png)

**Parameters.** Same protocol as above for each model: 8 prompts, all prompt positions, 8 directions,
$r=0.1$ ($r=0.3$ for Phi), $W=100$, teacher-forced, 300 tokens, read at the last block. Curves are the
median over prompts; `P` is the prompt length (60–65, so positions are shown as $p/P$).

**Panel (a).** The background $K(p\le5,t)$, i.e. the response to perturbing the first five prompt
positions, against the generated token. **Panel (b).** The peak over $t$ of $K(p,t)$ for each position.

**What it shows.** The background decays in DeepSeek (14 → 1.3, i.e. −0.74% per token, exponential,
$R^2=0.81$) and in Qwen2.5 (7 → 3); it is **flat** in Qwen3-4B and Phi; Phi's background even *rises*
around $t=40$–60 before decaying. The front amplitude separates the models more sharply than any other
measurement: **Phi ≈78 for every position**, DeepSeek 5–25, Qwen3-4B and Qwen2.5 15–45, Qwen3.5 ~3.

**Why it matters.** The two components of the kernel have different model signatures. Phi amplifies
*every* position equally at a fixed future time; DeepSeek's influence decays with distance. This is the
mechanism behind the "models differ" table, expressed as two numbers instead of a list.

""" + anchor
assert anchor in s
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
