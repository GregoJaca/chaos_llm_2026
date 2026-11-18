import json
d = json.load(open(r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_threshold2\numbers4figs.json"))
print("EXTRAP vs SAME-RADIUS vs OBS - signed rule:")
for r in d["extrap_rows"]:
    print(f"  {r['source'][:-4]:>16s} r={r['r']:<5g} obs={r['obs_flips']:2d} "
          f"same={r['same_flips']:2d} exact={r['same_exact']:2d} | "
          f"extrap={r['ext_flips']:2d} exact={r['ext_exact']:2d} "
          f"steperr={r['ext_steperr']:2d} max={r['ext_steperr_max']:3d}")
print("borderline:", d["borderline"])
