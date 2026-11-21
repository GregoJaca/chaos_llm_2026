import json
d = json.load(open(r"C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\free_inter_full\summary.json"))
print("flip_frac:", d["flip_frac"])
print("mean_flip_step:", d["mean_flip_step"])
print("mean_postflip_agreement:", [round(x, 3) for x in d["mean_postflip_agreement"]])
