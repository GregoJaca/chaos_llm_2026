"""Continue an existing greedy trajectory for N more tokens (re-prefill, then generate).

Patched for multi-model use (protocol_general):
  * to_np() instead of .cpu().numpy() on hidden states (bf16 crashes .numpy())
  * dtype auto, device from protocol, set_window warns on unsupported models
  * writes a sidecar json next to the output (provenance: model, dtype, window)

Note (Phi-4-mini): its longrope makes transformers re-prefill the cache when a
trajectory crosses original_max_position_embeddings=4096; keep prompt+generated
below that horizon for this model, or expect one slow re-prefill step.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol_general import (
    DEVICE, load_model, model_info, save_json, set_window, to_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--src', required=True, help='npz with prompt_ids, gen_ids, h_last')
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', type=int, default=2000)
    ap.add_argument('--window', type=int, default=4096)
    ap.add_argument('--dtype', default='auto')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    d = np.load(args.src)
    pids = torch.as_tensor(d['prompt_ids'], device=DEVICE, dtype=torch.long)[None]
    gids = torch.as_tensor(d['gen_ids'], device=DEVICE, dtype=torch.long)[None]
    h_last = d['h_last']
    model, tok, model_path = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok, model_path=model_path)
    if int(max(d['prompt_ids'].max(), d['gen_ids'].max())) >= info['vocab_size']:
        raise ValueError('source npz has token ids beyond this model\'s vocab (another model?)')
    set_window(model, args.window)

    with torch.inference_mode():
        emb = model.get_input_embeddings()(torch.cat([pids, gids], dim=1))
        o = model(inputs_embeds=emb, use_cache=True, output_hidden_states=False)
        past = o.past_key_values
        nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
        new_tokens = [int(nxt.item())]
        new_h = []
        for _ in range(args.extra):
            o = model(input_ids=nxt, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = o.past_key_values
            new_h.append(to_np(o.hidden_states[-1][0, -1]).astype(np.float32))
            nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
            new_tokens.append(int(nxt.item()))
            if len(new_tokens) >= args.extra:
                break

    gen = np.concatenate([gids[0].cpu().numpy(), np.asarray(new_tokens[:-1], dtype=np.int32)])
    h = np.concatenate([h_last, np.stack(new_h[:len(gen) - len(gids[0])])], axis=0)
    np.savez(args.out, prompt_ids=d['prompt_ids'], gen_ids=gen, h_last=h,
             extended_from=os.path.basename(args.src), window=args.window)
    save_json(args.out + '.json', dict(src=args.src, model=args.model,
             model_path=model_path, dtype=info['dtype'], params=info['params'],
             num_layers=info['num_layers'], hidden_size=info['hidden_size'],
             vocab_size=info['vocab_size'], extra=args.extra, window=args.window,
             src_len=int(gids.shape[1]), out_len=int(len(gen))))
    print('saved', args.out, 'length', len(gen))


if __name__ == '__main__':
    main()
