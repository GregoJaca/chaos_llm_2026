"""Generate the flux/lambda/depth figure families for several runs."""
import os, sys, glob
sys.path.insert(0, "scripts")
import plots

RUNS = [
    ("deepseek_all", "results/big4/a1_all_layers/prompts"),
    ("deepseek_cyber", "results/big4/a1_all_layers/prompts_cyber"),
    ("qwen3_4b", "results/models/qwen3_4b/a1"),
    ("phi4mini", "results/models/phi4mini/a1"),
    ("qwen3_5", "results/models/qwen3_5/a1"),
]
OUT = "results/figs_models"
made = []
for tag, d in RUNS:
    if not os.path.exists(os.path.join(d, "a1.npz")):
        print("skip", tag); continue
    o = os.path.join(OUT, tag)
    os.makedirs(o, exist_ok=True)
    try:
        a1 = plots.load_a1(d)
        made += plots.flux_tube_family(a1, o, tag, light=True)
        made += plots.lambda_family(a1, o, tag, light=True)
        made += plots.depth_family(a1, o, tag, light=True)
        print("done", tag, flush=True)
    except Exception as e:
        print("FAILED", tag, type(e).__name__, str(e)[:200], flush=True)
print("TOTAL", len(made))
