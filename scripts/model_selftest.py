"""Quick inference sanity check for a new model."""
import argparse, os, time
import numpy as np, torch
from chaos_llm.protocol import load_model, model_info, prompt_ids, generate, set_window, full_embeddings

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--window', type=int, default=100)
ap.add_argument('--tokens', type=int, default=40)
ap.add_argument('--dtype', default='auto')
args = ap.parse_args()

info = model_info(args.model)
print('INFO', {k: info[k] for k in ['params', 'dtype', 'hidden_size', 'num_layers', 'vocab_size', 'sliding_window']}, flush=True)
t0 = time.time()
model, tok, path = load_model(args.model, dtype=args.dtype)
print('loaded in %.1fs, dtype %s' % (time.time() - t0, next(model.parameters()).dtype), flush=True)
print('window supported:', set_window(model, args.window), flush=True)

text = "Explain how ocean tides are produced by the Moon and the Sun."
pids = prompt_ids(tok, text)
P = int(pids.shape[1])
D = model.config.hidden_size
print('prompt tokens', P, 'hidden', D, flush=True)

with torch.inference_mode():
    emb = model.get_input_embeddings()(pids)
    gen2 = generate(model, emb.expand(2, -1, -1), args.tokens)
same = torch.equal(gen2[0], gen2[1])
print('r=0 duplicate rows identical:', same, flush=True)
gen = gen2[0:1]
print('first tokens', gen[0, :6].tolist(), flush=True)

E = full_embeddings(model, pids, gen)
T = int(gen.shape[1])
with torch.inference_mode():
    o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
              output_hidden_states=True, use_cache=False)
h0 = o.hidden_states[-1][0, P:P + T].detach()
del o
torch.cuda.empty_cache()

rng = np.random.default_rng(0)
v = torch.as_tensor(rng.standard_normal(D), device='cuda', dtype=torch.float32)
v = v / v.norm()
print('r       a(0)     a(10)    a(39)', flush=True)
for r in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    E2 = E.clone()
    E2[0, :P] += (r * v).view(1, D)
    with torch.inference_mode():
        o = model(inputs_embeds=E2, attention_mask=torch.ones((1, E2.shape[1]), dtype=torch.long, device='cuda'),
                  output_hidden_states=True, use_cache=False)
    a = (o.hidden_states[-1][0, P:P + T] - h0).norm(dim=-1).float().cpu().numpy()
    print('%-7g %.4g  %.4g  %.4g' % (r, a[0], a[min(10, T - 1)], a[-1]), flush=True)
    del o, E2
    torch.cuda.empty_cache()
print('OK', flush=True)
