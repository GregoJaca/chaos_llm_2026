import json
d = json.load(open(r"C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\free_inter_w100\summary.json"))
print("radii:", d["radii"])
print("flip_frac:", d["flip_frac"])
print("mean_flip_step:", d["mean_flip_step"])
