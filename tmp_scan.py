import glob, os, json, numpy as np

def scan(pattern):
    rows = []
    for p in glob.glob(pattern):
        try:
            z = np.load(p, allow_pickle=True)
            n = z["dh"].shape
            m = json.load(open(os.path.join(os.path.dirname(p), "meta.json")))
            rows.append((p, n, m.get("temp"), m.get("dirs"), m.get("steps"),
                         m.get("mode"), m.get("dtype"), m.get("prompt", "")[:20]))
        except Exception as e:
            rows.append((p, "ERR", str(e), "", "", "", "", ""))
    return rows

for pat in ["results/longloop/*/*/*/softloop.npz",
            "results/longloop/*/*/softloop.npz",
            "results/longloop_prompt/*/*/*/*/softloop.npz",
            "results/longtf/*/a1.npz",
            "results/long_a1/*/a1.npz"]:
    print("===", pat)
    for r in sorted(scan(pat)):
        print("  ", r[0].replace("results/", ""), r[1], "T", r[2], "K", r[3], "n", r[4], r[5], r[6])
