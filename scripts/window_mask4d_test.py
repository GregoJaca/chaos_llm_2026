import argparse
import numpy as np, torch
from chaos_llm.protocol import load_model, full_embeddings

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
L = int(E.shape[1])


def band_mask(W):
    idx = torch.arange(L, device='cuda')
    band = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - W)
    m = torch.zeros((L, L), device='cuda', dtype=torch.float32)
    m = m.masked_fill(~band, torch.finfo(torch.float32).min)
    return m[None, None]


states = {}
for W in [args.window, 4096]:
    am = band_mask(W)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=am, output_hidden_states=True, use_cache=False)
    states[W] = o.hidden_states[-1][0].detach().float()
    del o
    torch.cuda.empty_cache()

d = (states[args.window] - states[4096]).norm(dim=-1)
nz = torch.where(d > 1e-6)[0]
print('P', P, 'T', T, 'window', args.window,
      'first difference at', int(nz[0]) if len(nz) else -1,
      '(t=', (int(nz[0]) - P) if len(nz) else None, ') max diff', float(d.max()))
