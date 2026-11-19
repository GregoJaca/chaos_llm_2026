"""Does the state difference survive after the tokens converge?

Take a perturbed free-run sequence.  Teacher-force two runs on that SAME token
sequence: one with the perturbed prompt, one with the baseline prompt.  If the
hidden states agree, the divergence is carried entirely by the token sequence;
if they differ, the prompt perturbation leaves a persistent state signature.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import load_model, set_window, full_embeddings, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B')
    ap.add_argument('--run', required=True)
    ap.add_argument('--free-run', required=True)
    ap.add_argument('--radii', default='0.5,1.0')
    ap.add_argument('--dirs', default='0,1,2,5,10,11')
    ap.add_argument('--chunk', type=int, default=6)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.run, 'a1.npz'))
    meta = json.load(open(os.path.join(args.run, 'meta.json')))
    fr = np.load(os.path.join(args.free_run, 'free_run.npz'), allow_pickle=True)
    frs = json.load(open(os.path.join(args.free_run, 'summary.json')))
    simplex, sub = d['simplex'], d['subspace']
    P = int(meta['prompt_len']); K = simplex.shape[0]
    T = int(meta['max_new_tokens'])
    radii = [float(x) for x in args.radii.split(',')]
    dirs = [int(x) for x in args.dirs.split(',')]
    fr_radii = [float(x) for x in frs['radii']]

    # reconstruct the perturbed token rows
    rows_by = {}
    gen_rows = list(fr['gen_ids'])
    nb = int(np.ceil(K / args.chunk))
    for ri, r in enumerate(radii):
        fri = int(np.argmin(np.abs(np.asarray(fr_radii) - r)))
        for b in range(nb):
            rows = gen_rows[fri * nb + b]
            start = b * args.chunk
            for j in range(rows.shape[0] - 1):
                di = start + j
                if di < K:
                    rows_by[(r, di)] = rows[j + 1]

    model, tok, _ = load_model(args.model)
    set_window(model, meta['window'])
    D = model.config.hidden_size
    pids = torch.as_tensor(d['prompt_ids'], device='cuda')[None]
    base_emb = model.get_input_embeddings()(pids)[0]
    sub_t = torch.as_tensor(sub, device='cuda', dtype=torch.long)
    ref = torch.as_tensor(d['gen_ids'], device='cuda')[None]

    def tf_forward(prompt_emb, seq):
        E = torch.cat([prompt_emb[None], model.get_input_embeddings()(seq)], dim=1)
        with torch.inference_mode():
            o = model(inputs_embeds=E, attention_mask=torch.ones((1, E.shape[1]), dtype=torch.long, device='cuda'),
                      output_hidden_states=True, use_cache=False)
        return torch.stack([h[0, P:P + T] for h in o.hidden_states[26:28]], 0)  # layers 26, 27

    h_base_ref = tf_forward(base_emb, ref)
    out = {}
    for r in radii:
        for di in dirs:
            seq = torch.as_tensor(rows_by[(r, di)], device='cuda')[None]
            delta = torch.zeros(P * D, device='cuda')
            delta.index_copy_(0, sub_t, torch.as_tensor(simplex[di], device='cuda') * r)
            prompt_pert = base_emb + delta.view(P, D)
            h_pert = tf_forward(prompt_pert, seq)
            h_base_same = tf_forward(base_emb, seq)
            diff = (h_pert - h_base_same).norm(dim=-1)          # state difference with SAME tokens
            vs_ref = (h_pert - h_base_ref).norm(dim=-1)          # state difference vs baseline tokens
            out[f'{r}_{di}'] = dict(diff=diff.float().cpu().numpy(), vs_ref=vs_ref.float().cpu().numpy())
            print(f'r={r} dir={di}: max same-token state diff={float(diff.max()):.4f}, '
                  f'final={float(diff[:, -1].max()):.5f}', flush=True)

    np.savez(os.path.join(args.out, 'after_flip.npz'),
             **{k + '_diff': v['diff'] for k, v in out.items()},
             **{k + '_ref': v['vs_ref'] for k, v in out.items()},
             radii=np.asarray(radii), dirs=np.asarray(dirs))
    save_json(os.path.join(args.out, 'meta.json'), dict(run=args.run, free_run=args.free_run,
               radii=radii, dirs=dirs, prompt_len=P, T=T))
    print('saved', args.out)


if __name__ == '__main__':
    main()
