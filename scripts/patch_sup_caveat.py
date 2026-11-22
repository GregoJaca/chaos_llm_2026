p = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\master.md"
s = open(p, encoding="utf-8").read()
old = """Qwen3-4B at $r=0.1$ per position shows a much larger error (13×), i.e. stronger saturation; the same
test at $r=0.03$ for the other four models is running."""
new = """**A limit of the measurement, found while checking the other models.** The same test is meaningless
for the fp16 models at these amplitudes: a hidden state is ~500 and fp16 resolves ~0.3 in absolute
value, while the single-position response at $r=0.03$ is ~0.05. The difference of two fp16 forward
passes is then below the format's resolution, and the measured "error" is 12–41× — a numerical
artefact, not saturation. **The superposition test is therefore only valid for DeepSeek (fp32).**"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
