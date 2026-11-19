import argparse
import numpy as np, torch
from chaos_llm.protocol import load_model, full_embeddings
from chaos_llm.window_patch import install_sliding_window

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--dtype', default='auto')
ap.add_argument('--attn', default='eager')
ap.add_argument('--tokens', type=int, default=120)
ap.add_argument('--window', type=int, default=10)
args = ap.parse_args()

model, tok, path = load_model(args.model, attn=args.attn, dtype=args.dtype)
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
E = full_embeddings(model, pids, gen)

states = {}
for W in [args.window, 4096]:
    install_sliding_window(model, W) if W != 4096 else install_sliding_window(model, 100000)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                  output_hidden_states=True, use_cache=False)
    states[W] = o.hidden_states[-1][0].detach().float()
    del o
    torch.cuda.empty_cache()

d = (states[args.window] - states[4096]).norm(dim=-1)
nz = torch.where(d > 1e-6)[0]
print('P', P, 'T', T, 'window', args.window,
      'first difference at', int(nz[0]) if len(nz) else -1,
      '(t=', (int(nz[0]) - P) if len(nz) else None, ') max diff', float(d.max()))
