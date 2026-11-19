import glob, re
files = glob.glob(r"C:\Users\hjaca\research\chaos_llm_2026\src\chaos_llm\*.py")
n = 0
for f in files:
    s = open(f, encoding="utf-8").read()
    orig = s
    s = s.replace(".float().float()", ".float()")
    s = re.sub(r"(?<!\.float\(\))\.cpu\(\)\.numpy\(\)", ".float().cpu().numpy()", s)
    s = s.replace(".float().float()", ".float()")
    if s != orig:
        open(f, "w", encoding="utf-8").write(s)
        n += 1
        print("patched", f)
print("files patched", n)
