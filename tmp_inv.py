import json, glob, os

print("=== A1 (time/depth) runs: window, dtype, steps")
for d in sorted(glob.glob("results/rstar/*")):
    p = os.path.join(d, "p1", "meta.json")
    if os.path.exists(p):
        m = json.load(open(p))
        print("%-16s W=%-5s dtype=%-8s steps=%-5s dirs=%-5s r=%s" % (
            os.path.basename(d), m.get("window"), m.get("dtype"),
            m.get("max_new_tokens"), m.get("num_directions"), m.get("radii")))

print("\n=== other a1 runs used by D1/D3")
for p in ["results/local_data/w100_layers/a1.npz",
          "results/local_data/a1_4k_test.npz",
          "results/local_data/window4d/qwen3_4b/w100/a1.npz"]:
    if os.path.exists(p):
        z = __import__("numpy").load(p, allow_pickle=True)
        print(p, z["dh_norm"].shape, "layers", z["layers"][:6], "radii", z["radii"])
        d = os.path.dirname(p)
        m = os.path.join(d, "meta.json")
        if os.path.exists(m):
            j = json.load(open(m))
            print("   W", j.get("window"), "dtype", j.get("dtype"), "steps", j.get("max_new_tokens"))
