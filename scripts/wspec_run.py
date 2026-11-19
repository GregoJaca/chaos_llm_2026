"""Window spectroscopy: load the model ONCE and sweep W, reusing run_a1 unchanged."""
import os
import sys

import chaos_llm.run_a1 as ra1

MODEL = os.path.expanduser("~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
ROOT = "results/big4/wspec"
LAYERS = "1,2,3,7,14,21,27"

print("loading model once", flush=True)
model, tok, path = ra1.load_model(MODEL)
ra1.load_model = lambda *a, **k: (model, tok, path)

todo = []
for w in list(range(60, 140)) + list(range(140, 401, 5)):
    if not os.path.exists(os.path.join(ROOT, f"w{w}", "a1.npz")):
        todo.append(w)
print("windows to do:", len(todo), flush=True)

for w in todo:
    argv = ["run_a1", "--prompt", "@prompts.txt", "--window", str(w), "--window4d",
            "--max-new-tokens", "200", "--num-directions", "4", "--radii", "0.1",
            "--layers", LAYERS, "--batch-dirs", "1", "--out", os.path.join(ROOT, f"w{w}")]
    sys.argv = argv
    try:
        ra1.main()
    except Exception as e:
        print("FAILED", w, type(e).__name__, str(e)[:120], flush=True)
print("WSPEC2 DONE", flush=True)
