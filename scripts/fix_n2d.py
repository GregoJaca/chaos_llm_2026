p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\notebook.md"
s = open(p, encoding="utf-8").read()
s = s.replace("12 positions from 4 to 100", "22 positions from 4 to 100")
s = s.replace("| Qwen3-4B | 6.1 | 5.3\u20137.1 | 1.3 | flat |",
              "| Qwen3-4B | 6.1 | 5.3\u20137.4 | 1.4 | flat |")
s = s.replace("| Qwen2.5-3B | 4.6 | 4.5\u20134.8 | 1.07 | flat |",
              "| Qwen2.5-3B | 4.6 | 4.3\u20134.8 | 1.1 | flat |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
