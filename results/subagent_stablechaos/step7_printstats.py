import json
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_stablechaos"
s = json.load(open(OUT + r"\stats.json"))
tv = s["T_vs_r"]
print("radii:", [round(r, 2) for r in tv["radii"]])
print("meanT:", [round(m, 1) for m in tv["meanT"]])
print("meanT_pred:", [round(m, 1) for m in tv["meanT_pred"]])
print("meanT_ci:", [[round(a, 1), round(b, 1)] for a, b in tv["meanT_ci"]])
print("log:", {k: round(v, 3) for k, v in tv["log_all"].items()}, "CI b:", [round(x, 1) for x in tv["log_b_ci"]])
print("pow:", {k: round(v, 3) for k, v in tv["pow_all"].items()}, "CI a:", [round(x, 3) for x in tv["pow_alpha_ci"]])
print("R2_threshold_means:", round(tv["R2_threshold_means"], 3))
print("log_fliponly:", {k: round(v, 2) for k, v in tv["log_fliponly"].items()})
print("pow_fliponly:", {k: round(v, 2) for k, v in tv["pow_fliponly"].items()})
print("flipfrac:", [round(x, 2) for x in tv["flipfrac"]])
print("flipfrac_pred:", [round(x, 2) for x in tv["flipfrac_pred"]])
print()
for key in ["r0.15", "r0.2", "r0.25", "r0.3", "r0.4", "r0.5", "r1", "r2"]:
    o = s["main"][key]
    f = o["fits"]
    print(f"{key}: flips {o['n_flip']}/20 pred {o['n_flip_pred']}, exactT {o['exact_pred_match']}, "
          f"MADthr {o['MAD_threshold']:.3f} MADexp {o['MAD_exp']:.3f}, "
          f"lam {f['exp']['lam']:.5f} R2exp {f['exp']['R2']:.2f}, "
          f"beta {f['stretched']['beta'] if f['stretched']['beta']==f['stretched']['beta'] else float('nan'):.3f} "
          f"R2str {f['stretched']['R2'] if f['stretched']['R2']==f['stretched']['R2'] else float('nan'):.3f}, "
          f"gamma {f['power']['gamma']:.3f} R2pow {f['power']['R2'] if f['power']['R2']==f['power']['R2'] else float('nan'):.3f}, "
          f"predonly {o['pred_only']} obsonly {o['obs_only']} later {o['later']} earlier {o['earlier']}")
print()
for p in ["cyber", "childhood"]:
    for key in ["r0.01", "r0.1", "r0.3"]:
        o = s[p][key]
        print(f"{p} {key}: flips {o['n_flip']}/20 pred {o['n_flip_pred']}, exactT {o['exact_pred_match']}, "
              f"MADthr {o['MAD_threshold']:.3f} MADexp {o['MAD_exp']:.3f}, predonly {o['pred_only']} obsonly {o['obs_only']}")
print()
print("discreteness:", s["discreteness"]["steps"], s["discreteness"]["counts"])
print("min_rc:", {k: round(v, 4) for k, v in s["discreteness"]["min_rc"].items()})
print("dh_norm:", json.dumps(s["dh_norm"], indent=0)[:400])
