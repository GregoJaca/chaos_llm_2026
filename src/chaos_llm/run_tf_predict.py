"""Teacher-forced vs autoregressive: can one teacher-forced forward predict the
actual free-run next tokens after a perturbation?

For each (r, direction):
  - teacher-forced pass along the *baseline* token sequence -> argmax token at every step
  - free-run (autoregressive) tokens already collected in a free_run record
Compare predicted vs actual tokens: agreement on the flip step and afterwards.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--free-run', required=True, help='dir with free_run.npz + summary.json')
    ap.add_argument('--radii', default='0.1,0.3,0.5')
    ap.add_argument('--chunk', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    fr = np.load(os.path.join(args.free_run, 'free_run.npz'), allow_pickle=True)
    frs = json.load(open(os.path.join(args.free_run, 'summary.json')))
    simplex, sub = d['simplex'], d['subspace']
    P = int(meta['prompt_len']); K = int(simplex.shape[0])
    radii = [float(x) for x in args.radii.split(',')]
    fr_radii = [float(x) for x in frs['radii']]

    # reconstruct autoregressive token rows: each batch = [ref, dir...]
    gen_rows = list(fr['gen_ids'])
    ar = np.zeros((len(radii), K, int(meta['max_new_tokens'])), np.int64) - 1
    for ri, r in enumerate(radii):
        fri = int(np.argmin(np.abs(np.asarray(fr_radii) - r)))
        nb = int(np.ceil(K / args.chunk))
        for b in range(nb):
            rows = gen_rows[fri * nb + b]
            start = b * args.chunk
            for j in range(rows.shape[0] - 1):
                if start + j < K:
                    ar[ri, start + j] = rows[j + 1]

    model, tok, _ = load_model(args.model)
    set_window(model, meta['window'])
    D = model.config.hidden_size
    pids = torch.as_tensor(d['prompt_ids'], device='cuda')[None]
    gids = torch.as_tensor(d['gen_ids'], device='cuda')[None]
    T = gids.shape[1]
    sub_t = torch.as_tensor(sub, device='cuda', dtype=torch.long)
    Efull = full_embeddings(model, pids, gids)
    with torch.inference_mode():
        o0 = model(inputs_embeds=Efull, attention_mask=torch.ones((1, Efull.shape[1]), dtype=torch.long, device='cuda'),
                   use_cache=False)
    ref_tok = o0.logits[0, P - 1:P + T - 1].argmax(-1)
    del o0
    torch.cuda.empty_cache()

    pred = np.zeros((len(radii), K, T), np.int64)
    for ri, r in enumerate(radii):
        for di in range(K):
            E = Efull.clone()
            delta = torch.zeros(P * D, device='cuda')
            delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device='cuda') * r)
            E[0, :P] += delta.view(P, D)
            with torch.inference_mode():
                o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                          use_cache=False)
            pred[ri, di] = o.logits[0, P - 1:P + T - 1].argmax(-1).cpu().numpy()
            del o, E, delta
            torch.cuda.empty_cache()
        print('r', r, 'done', flush=True)

    ref = ref_tok.cpu().numpy()
    np.savez(os.path.join(args.out, 'tf_predict.npz'), pred=pred, ar=ar, ref=ref,
             radii=np.asarray(radii))
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, free_run=args.free_run,
               radii=radii, chunk=args.chunk, prompt_len=P, T=T))
    print('saved', args.out)


if __name__ == '__main__':
    main()
