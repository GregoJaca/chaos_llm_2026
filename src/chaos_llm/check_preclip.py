"""Does the attention window change anything BEFORE the context reaches W?

Same prompt, same forced tokens, no perturbation (r=0).  Compare hidden states
across windows position by position, bitwise and in norm.
"""
import os, json
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings

M = os.path.expanduser("~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
d = np.load("results/long_traj/interstellar_travel.npz")
pids = torch.as_tensor(d["prompt_ids"], device="cuda")[None]
gids = torch.as_tensor(d["gen_ids"][:200], device="cuda")[None]
P = int(pids.shape[1]); T = int(gids.shape[1])

model, tok, _ = load_model(M)
E = full_embeddings(model, pids, gids)

states = {}
for W in [50, 100, 200, 4096]:
    set_window(model, W)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device="cuda"),
                  output_hidden_states=True, use_cache=False)
    states[W] = {l: o.hidden_states[l][0].detach().clone() for l in [1, 7, 14, 27]}
    del o
    torch.cuda.empty_cache()

print(f"P={P}, T={T}")
np.savez("results/preclip.npz",
         **{f"W{W}_l{l}": states[W][l].cpu().numpy() for W in states for l in [1, 7, 14, 27]},
         P=P, T=T)
ref = states[4096]
for W in [50, 100, 200]:
    print(f"--- W={W} vs W=4096")
    for l in [1, 7, 14, 27]:
        diff = (states[W][l] - ref[l]).norm(dim=-1)          # [P+T]
        base = ref[l].norm(dim=-1)
        rel = diff / base.clamp_min(1e-9)
        # first position where they differ at all
        nz = torch.where(diff > 0)[0]
        first = int(nz[0]) if len(nz) else -1
        # first position with relative difference > 1e-5
        big = torch.where(rel > 1e-5)[0]
        firstbig = int(big[0]) if len(big) else -1
        print(f"  layer {l:2d}: first any-diff at pos {first:4d} (t={first-P:4d}), "
              f"first rel>1e-5 at pos {firstbig:4d} (t={firstbig-P:4d}), max rel diff={float(rel.max()):.2e}")
