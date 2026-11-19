"""Does set_window actually change the attention? Compare W=100 vs W=4096, no perturbation."""
import argparse
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--dtype', default='auto')
ap.add_argument('--tokens', type=int, default=120)
ap.add_argument('--layers', default='1,4,8,14,20')
ap.add_argument('--attn', default='sdpa')
args = ap.parse_args()

model, tok, path = load_model(args.model, attn=args.attn, dtype=args.dtype)
info = model._chaos_info
print('layers', info['num_layers'], 'dtype', next(model.parameters()).dtype, flush=True)

text = "Explain how ocean tides are produced by the Moon and the Sun, and why there are two high tides each day."
pids = tok(text, return_tensors="pt").input_ids.cuda()
P = int(pids.shape[1])
with torch.inference_mode():
    emb = model.get_input_embeddings()(pids)
    o = model(inputs_embeds=emb, use_cache=True)
    nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
    toks = [int(nxt.item())]
    for _ in range(args.tokens - 1):
        o = model(input_ids=nxt, past_key_values=o.past_key_values, use_cache=True)
        nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
        toks.append(int(nxt.item()))
gen = torch.as_tensor(toks, device='cuda')[None]
T = int(gen.shape[1])
print('P', P, 'T', T, flush=True)

E = full_embeddings(model, pids, gen)
layers = [int(x) for x in args.layers.split(',')]
states = {}
for W in [100, 4096]:
    print('window supported:', set_window(model, W), flush=True)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                  output_hidden_states=True, use_cache=False)
    states[W] = {l: o.hidden_states[l][0].detach().float() for l in layers}
    del o
    torch.cuda.empty_cache()

for l in layers:
    diff = (states[100][l] - states[4096][l]).norm(dim=-1)
    nz = torch.where(diff > 0)[0]
    first = int(nz[0]) if len(nz) else -1
    print(f'layer {l}: first difference at position {first} (t={first-P}), max diff {float(diff.max()):.3e}', flush=True)
