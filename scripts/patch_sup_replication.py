p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()
anchor = "**A limit of the measurement, found while checking the other models.**"
add = r"""**Replication at $r=0.03$ (total norm 0.24):**

| run | $\|A-S\|/\|A\|$ (median) | $\cos(A,S)$ |
|---|---|---|
| DeepSeek, prompt 1 | 6.2% | 0.999 |
| DeepSeek, prompt 2 | 5.1% | 0.999 |
| DeepSeek, prompt 3 | 5.5% | 0.999 |
| Qwen3.5-2B, **fp32**, prompt 1 | 7.7% | 0.997 |

So the decomposition holds to 5–8% across three prompts and on a second model when the arithmetic is
fp32.

""" + anchor
assert anchor in s
s = s.replace(anchor, add, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
