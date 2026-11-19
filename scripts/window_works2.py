import argparse
import numpy as np, torch
from chaos_llm.protocol import load_model, full_embeddings

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--dtype', default='auto')
ap.add_argument('--attn', default='eager')
ap.add_argument('--tokens', type=int, default=120)
ap.add_argument('--window', type=int, default=100)
args = ap.parse_args()

model, tok, path = load_model(args.model, attn=args.attn, dtype=args.dtype)
cfg = model.config
print('model', type(model).__name__, 'layers', cfg.num_hidden_layers)
print('before: sliding_window', getattr(cfg, 'sliding_window', None),
      'use', getattr(cfg, 'use_sliding_window', None),
      'layer_types', str(getattr(cfg, 'layer_types', None))[:60], flush=True)


def enable_window(model, W):
    cfg = model.config
    cfg.sliding_window = int(W)
    if hasattr(cfg, 'use_sliding_window'):
        cfg.use_sliding_window = True
    lt = getattr(cfg, 'layer_types', None)
    if isinstance(lt, list) and lt:
        cfg.layer_types = ['sliding_attention'] * len(lt)
    # also patch inner configs if any
    for sub in ('text_config',):
        tc = getattr(cfg, sub, None)
        if tc is not None:
            setattr(tc, 'sliding_window', int(W))
            if hasattr(tc, 'layer_types') and isinstance(tc.layer_types, list):
                tc.layer_types = ['sliding_attention'] * len(tc.layer_types)


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
    enable_window(model, W)
    print('after set: sliding_window', model.config.sliding_window,
          'layer_types', str(getattr(model.config, 'layer_types', None))[:40], flush=True)
    with torch.inference_mode():
        o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                  output_hidden_states=True, use_cache=False)
    states[W] = o.hidden_states[-1][0].detach().float()
    del o
    torch.cuda.empty_cache()

d = (states[args.window] - states[4096]).norm(dim=-1)
nz = torch.where(d > 0)[0]
print('P', P, 'T', T, 'first difference at', int(nz[0]) if len(nz) else -1,
      '(t=', (int(nz[0]) - P) if len(nz) else None, ') max', float(d.max()))
