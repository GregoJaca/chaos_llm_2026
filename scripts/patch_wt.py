p = r"C:\Users\hjaca\research\chaos_llm_2026\scripts\analysis_window_tokens.py"
s = open(p, encoding="utf-8").read()
s = s.replace('ax.plot(wv, first, "o", ms=8, color="tab:blue", label="measured")',
              'm = first > 0\nax.plot(wv[m], first[m], "o", ms=8, color="tab:blue", label="measured")\n'
              'for w in wv[~m]: ax.annotate("identical\\nfor 300", (w, 5), ha="center", fontsize=7)')
s = s.replace('ax.set_title("(c) The extra latency is 5–7 tokens, constant")',
              'ax.set_title("(c) Extra latency: 2–7 tokens (mean 5)")')
open(p, "w", encoding="utf-8").write(s)
print("ok")
