"""Continue an existing greedy trajectory for N more tokens (re-prefill, then generate)."""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--src', required=True, help='npz with prompt_ids, gen_ids, h_last')
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', type=int, default=2000)
    ap.add_argument('--window', type=int, default=4096)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    d = np.load(args.src)
    pids = torch.as_tensor(d['prompt_ids'], device='cuda')[None]
    gids = torch.as_tensor(d['gen_ids'], device='cuda')[None]
    h_last = d['h_last']
    model, tok, _ = load_model(args.model)
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
            new_h.append(o.hidden_states[-1][0, -1].detach().float().cpu().numpy().astype(np.float32))
            nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
            new_tokens.append(int(nxt.item()))
            if len(new_tokens) >= args.extra:
                break

    gen = np.concatenate([gids[0].float().cpu().numpy(), np.asarray(new_tokens[:-1], dtype=np.int32)])
    h = np.concatenate([h_last, np.stack(new_h[:len(gen) - len(gids[0])])], axis=0)
    np.savez(args.out, prompt_ids=d['prompt_ids'], gen_ids=gen, h_last=h,
             extended_from=os.path.basename(args.src), window=args.window)
    print('saved', args.out, 'length', len(gen))


if __name__ == '__main__':
    main()
