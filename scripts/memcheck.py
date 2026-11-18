import os, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings
M = os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
model, tok, _ = load_model(M)
print('after load GB', torch.cuda.memory_allocated()/1e9, flush=True)
import numpy as np
d = np.load('results/long_traj/interstellar_travel.npz')
pids = torch.as_tensor(d['prompt_ids']).cuda()[None]
gids = torch.as_tensor(d['gen_ids'][:400]).cuda()[None]
print('params GB', sum(p.numel() for p in model.parameters())*4/1e9, 'dtype', next(model.parameters()).dtype)
E = full_embeddings(model, pids, gids)
print('E', E.shape, E.dtype, flush=True)
for w in [100, 1000]:
    set_window(model, w)
    for B in [1, 2, 3]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        Eb = E.expand(B, -1, -1).clone()
        try:
            with torch.inference_mode():
                o = model(inputs_embeds=Eb, attention_mask=torch.ones((B, Eb.shape[1]), dtype=torch.long, device='cuda'),
                          output_hidden_states=True, use_cache=False)
            print(f'w={w} B={B} ok alloc={torch.cuda.memory_allocated()/1e9:.2f} peak={torch.cuda.max_memory_allocated()/1e9:.2f}', flush=True)
            del o
        except RuntimeError as e:
            print(f'w={w} B={B} FAIL {str(e)[:80]}', flush=True)
        del Eb
        torch.cuda.empty_cache()
