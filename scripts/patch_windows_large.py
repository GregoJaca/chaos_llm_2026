p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()
old = "- **First departure** from the unwindowed run occurs at $t=W-P$: measured 36/136/0 for $W=100/200/60$\n  with $P=64$, all layers."
new = ("- **First departure** from the unwindowed run occurs at $t=W-P$: measured 36/136/0 for "
       "$W=100/200/60$ with $P=64$, all layers; and 237 = 300−63 for $W=300$ in three models "
       "(DeepSeek, Qwen3-4B, Phi, all seven saved layers) on a 900-token continuation.")
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")

p2 = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\veto.md"
t = open(p2, encoding="utf-8").read()
anchor = "### 11.7 Bottom line (updated)"
add = r"""### 11.6b Large windows: the departure law holds at W=300 and 500

Three models, the same fixed 900-token continuation, W=300 against W=500: the first difference between
the two runs is at **t=237 = W−P** (P=63) in every saved layer of every model. The window law therefore
holds at large windows too, where the effect appears late in the continuation. The flush at these
windows falls beyond the 900-token horizon, so only the departure side is tested here.

""" + anchor
assert anchor in t
t = t.replace(anchor, add, 1)
open(p2, "w", encoding="utf-8").write(t)
print("veto ok")
