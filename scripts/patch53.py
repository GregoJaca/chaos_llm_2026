p = r"C:\Users\hjaca\research\chaos_llm_2026\scripts\queue53.sh"
s = open(p, encoding="utf-8").read()
s = s.replace("--trials 3 --r 0.1 --layers 1,7,14,last", "--trials 3 --r 0.03 --layers 1,7,14,last")
s = s.replace("out=results/superposition/$m", "out=results/superposition/${m}_r003")
open(p, "w", encoding="utf-8").write(s)
print("patched queue53: superposition at r=0.03")
