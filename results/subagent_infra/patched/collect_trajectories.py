"""Collect long greedy trajectories with hidden states, regular settings.

Patched for multi-model use (protocol_general):
  * dtype auto (fp32 <= 2.5B params, bf16 above; --dtype overrides)
  * layers default = spread over the real depth (DeepSeek-1.5B reproduces 7,14,21,27)
  * to_np() instead of .cpu().numpy(): bf16 tensors CRASH .numpy() (numpy has no bfloat16)
  * device from protocol (cuda if available, else cpu -- CPU only for smoke tests)
  * set_window warns instead of silently no-op'ing on models without a window
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol_general import (
    DEVICE, default_layer_indices, load_model, model_info, prompt_ids,
    save_json, set_window, to_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--window', type=int, default=4096)
    ap.add_argument('--max-new-tokens', type=int, default=1000)
    ap.add_argument('--layers', default='')
    ap.add_argument('--dtype', default='auto')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    model, tok, model_path = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok, model_path=model_path)
    layers = ([int(x) for x in args.layers.split(',') if x.strip()]
              if args.layers.strip() else default_layer_indices(info['num_layers']))
    if max(layers) >= info['n_hidden_states']:
        raise ValueError(f'layer indices {layers} out of range '
                         f'(hidden_states has {info["n_hidden_states"]} entries)')
    set_window(model, args.window)
    lines = [l for l in open(args.prompts, encoding='utf-8').read().splitlines() if l.strip()]
    for line in lines:
        name, text = line.split('\t', 1) if '\t' in line else (f'prompt_{len(os.listdir(args.out))}', line)
        pids = prompt_ids(tok, text); P = pids.shape[1]
        with torch.inference_mode(): emb = model.get_input_embeddings()(pids)
        past = None; next_ids = None; gen = []; hs = {l: [] for l in layers}; full_hs = []
        for step in range(args.max_new_tokens):
            with torch.inference_mode():
                if past is None:
                    o = model(inputs_embeds=emb, use_cache=True, output_hidden_states=True, return_dict=True)
                else:
                    o = model(input_ids=next_ids, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            past = o.past_key_values
            for l in layers: hs[l].append(to_np(o.hidden_states[l][0, -1]).astype(np.float32))
            full_hs.append(to_np(o.hidden_states[-1][0, -1]).astype(np.float32))
            nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
            gen.append(int(nxt.item())); next_ids = nxt
            del o
        gen = np.asarray(gen, dtype=np.int32)
        np.savez(os.path.join(args.out, f'{name}.npz'), prompt_ids=pids[0].cpu().numpy(), gen_ids=gen,
                 h_last=np.stack(full_hs), **{f'h{l}': np.stack(hs[l]) for l in layers})
        save_json(os.path.join(args.out, f'{name}.json'), dict(name=name, text=text,
                 model=args.model, model_path=model_path, dtype=info['dtype'],
                 params=info['params'], hidden_size=info['hidden_size'],
                 num_layers=info['num_layers'], vocab_size=info['vocab_size'],
                 prompt_len=int(P), max_new_tokens=args.max_new_tokens,
                 window=args.window, layers=layers))
        print(name, 'done', len(gen), flush=True)


if __name__ == '__main__':
    main()
