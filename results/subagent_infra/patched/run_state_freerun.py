"""Autoregressive validation of a current-state kick.

Feed the prompt plus the baseline tokens up to step s, add r*v to the embedding
of that position, then continue greedily.  Compare with the unperturbed tokens.

Patched for multi-model use (protocol_general): dtype auto, hidden size from
model_info(), device from protocol, bf16-safe perturbation arithmetic, and an
optional --check-r0 pass that verifies the KV-cache continuation reproduces the
stored baseline tokens before any perturbation is trusted.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol_general import (
    DEVICE, full_embeddings, load_model, model_info, save_json, set_window, to_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', default='16,50')
    ap.add_argument('--dirs', type=int, default=24)
    ap.add_argument('--r', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--dtype', default='auto')
    ap.add_argument('--check-r0', action='store_true',
                    help='first run one unperturbed continuation and verify it '
                         'reproduces the stored baseline tokens')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    pids = torch.as_tensor(d['prompt_ids'], device=DEVICE, dtype=torch.long)[None]
    gids = torch.as_tensor(d['gen_ids'], device=DEVICE, dtype=torch.long)[None]
    P = int(meta['prompt_len']); T = int(gids.shape[1])
    steps = [int(x) for x in args.steps.split(',')]
    model, tok, _ = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok)
    D = info['hidden_size']
    if int(d['gen_ids'].max()) >= info['vocab_size']:
        raise ValueError('source run has token ids beyond this model\'s vocab (another model?)')
    set_window(model, meta['window'])

    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    dirs = []
    for _ in range(args.dirs):
        v = torch.randn(D, device=DEVICE, generator=gen)
        dirs.append(v / v.norm())
    dirs = torch.stack(dirs)

    def continue_from(s, vec, radius):
        """Prefill prompt+gids[:s+1] (+ radius*vec at position P+s), continue greedily."""
        with torch.inference_mode():
            emb = model.get_input_embeddings()(torch.cat([pids, gids[:, :s + 1]], dim=1))
            if vec is not None:
                emb[0, P + s] = (emb[0, P + s].float() + radius * vec).to(emb.dtype)
            o = model(inputs_embeds=emb, use_cache=True, output_hidden_states=False)
            past = o.past_key_values
            nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
            toks = [int(nxt.item())]
            for t in range(s + 2, T):
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = torch.argmax(o.logits[0, -1]).view(1, 1)
                toks.append(int(nxt.item()))
            del o, past, emb
        torch.cuda.empty_cache()
        return np.asarray(toks, dtype=np.int64)

    if args.check_r0:
        ref = gids[0].cpu().numpy()
        for s in steps[:1]:
            toks = continue_from(s, None, 0.0)
            n = min(len(toks), T - s - 1)
            same = np.array_equal(toks[:n], ref[s + 1:s + 1 + n])
            mm = int(np.argmax(toks[:n] != ref[s + 1:s + 1 + n])) if not same else -1
            print(f'[r0 check] s={s}: continuation {"MATCHES" if same else f"diverges at +{mm}"} '
                  f'the stored baseline tokens', flush=True)

    out = np.zeros((len(steps), args.dirs, T), np.int64) - 1
    for si, s in enumerate(steps):
        for di in range(args.dirs):
            toks = continue_from(s, dirs[di], args.r)
            out[si, di, :] = -1
            out[si, di, s + 1:s + 1 + len(toks)] = toks
        print('step', s, 'done', flush=True)

    ref = gids[0].cpu().numpy()
    np.savez(os.path.join(args.out, 'state_freerun.npz'), pred=out, ref=ref,
             steps=np.asarray(steps), r=args.r, dirs=to_np(dirs))
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, model=args.model,
               dtype=info['dtype'], params=info['params'], steps=steps, dirs=args.dirs,
               r=args.r, seed=args.seed, hidden_size=D, num_layers=info['num_layers'],
               vocab_size=info['vocab_size'], prompt_len=P, T=T,
               window=meta['window']))
    print('saved', args.out)


if __name__ == '__main__':
    main()
