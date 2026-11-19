"""Pick a usable perturbation radius: report a(t)/r across r."""
import argparse
import numpy as np, torch
from chaos_llm.protocol import load_model, prompt_ids, generate, full_embeddings, set_window, build_simplex

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--dtype', default='auto')
ap.add_argument('--tokens', type=int, default=120)
ap.add_argument('--dirs', type=int, default=6)
args = ap.parse_args()

model, tok, path = load_model(args.model, dtype=args.dtype)
set_window(model, 100)
info = model._chaos_info
print('dtype', next(model.parameters()).dtype, 'layers', info['num_layers'], 'hidden', info['hidden_size'], flush=True)

text = "Explain how ocean tides are produced by the Moon and the Sun."
pids = prompt_ids(tok, text)
P = int(pids.shape[1])
with torch.inference_mode():
    emb = model.get_input_embeddings()(pids)
    gen = generate(model, emb, args.tokens)
T = int(gen.shape[1])
E = full_embeddings(model, pids, gen)
with torch.inference_mode():
    o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
              output_hidden_states=True, use_cache=False)
h0 = o.hidden_states[-1][0, P:P + T].detach()
del o
torch.cuda.empty_cache()

rng = np.random.default_rng(0)
V = torch.as_tensor(rng.standard_normal((args.dirs, info['hidden_size'])), device='cuda', dtype=torch.float32)
V = V / V.norm(dim=1, keepdim=True)
print('r       a0/r    a10/r   a50/r   a100/r', flush=True)
for r in [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]:
    vals = []
    for di in range(args.dirs):
        E2 = E.clone()
        E2[0, :P] += (r * V[di]).view(1, -1)
        with torch.inference_mode():
            o = model(inputs_embeds=E2, attention_mask=torch.ones((1, E2.shape[1]), dtype=torch.long, device='cuda'),
                      output_hidden_states=True, use_cache=False)
        a = (o.hidden_states[-1][0, P:P + T] - h0).norm(dim=-1).float().cpu().numpy()
        vals.append(a)
        del o, E2
    A = np.median(np.stack(vals), axis=0)
    idx = [0, min(10, T - 1), min(50, T - 1), min(100, T - 1)]
    print('%-6g %s' % (r, '  '.join('%.3f' % (A[i] / r) for i in idx)), flush=True)
    torch.cuda.empty_cache()
print('OK')
