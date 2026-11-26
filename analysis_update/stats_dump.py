import numpy as np, sys, json
sys.path.insert(0, "analysis_update")
from common import *

print("ALPHA 0.03-0.3")
for n in ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Qwen3.5-2B"]:
    z = load_a1(n); r = z["radii"].astype(float); dh = z["dh_norm"].astype(float)
    t = min(50, dh.shape[3] - 1)
    y = np.array([np.median(dh[i, :, -1, t]) for i in range(len(r))])
    a = np.polyfit(np.log(r), np.log(y), 1)[0]
    print("  %-16s radii=%s alpha=%+.3f" % (n, np.round(r, 3), a))

print("REPLAY")
t = json.load(open("analysis_update/time_numbers.json"))
for n, v in t.items():
    if "a" in v and len(v["a"]) > 100:
        a = np.array(v["a"])
        print("  %-16s a1=%7.2f aend=%7.2f min=%7.2f max=%8.2f late=%+.4f" %
              (n, a[1], a[-1], a.min(), a.max(), v["late_slope"]))

print("RELATIVE DEPTH EXPONENT")
for n in ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini"]:
    z = np.load(DEPTH[n], allow_pickle=True); lam = z["lam"].astype(float); L = lam.shape[1]
    dl = load_a1(n); hn = dl["norm_h0"].astype(float); t = min(50, hn.shape[1] - 1)
    gstate = np.log(hn[-1, t] / hn[1, t]) / (L - 1)
    gtan = np.median(lam[:, -1, :])
    print("  %-16s gtan=%+.3f gstate=%+.3f relative=%+.3f" % (n, gtan, gstate, gtan - gstate))
