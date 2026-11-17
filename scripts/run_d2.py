import os, glob, json, argparse
import numpy as np
import torch
from safetensors import safe_open
from chaos_llm.analysis.dimension import d2_curve, rqa

torch.set_num_threads(8)
ROOT = os.path.expanduser('~/LLM')
OUT = os.path.join(ROOT, 'chaos_llm_2026/results/d2'); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(ROOT, 'chaos_llm_2026/results/figures'); os.makedirs(FIG, exist_ok=True)


def load_embedding():
    snap = sorted(glob.glob(os.path.expanduser('~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*/')))[0]
    idx = os.path.join(snap, 'model.safetensors.index.json')
    if os.path.exists(idx):
        wm = json.load(open(idx))['weight_map']
        shard = wm['model.embed_tokens.weight']
        f = safe_open(os.path.join(snap, shard), framework='pt')
        return f.get_tensor('model.embed_tokens.weight').float().numpy()
    f = safe_open(os.path.join(snap, 'model.safetensors'), framework='pt')
    return f.get_tensor('model.embed_tokens.weight').float().numpy()


def load_pt(p):
    return torch.load(p, map_location='cpu', weights_only=True).float().squeeze().numpy()


def prep(X, n=None, dedupe=False):
    if dedupe:
        X = np.unique(X, axis=0)
    if n is not None and len(X) > n:
        X = X[:n]
    X = X - X.mean(axis=0, keepdims=True)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-prompts', type=int, default=12)
    ap.add_argument('--n-main', type=int, default=2000)
    args = ap.parse_args()
    prompts = sorted([d for d in glob.glob(ROOT + '/results_pecs/*/') if os.path.exists(d + 'hidden_states_last.pt')])[:args.max_prompts]
    print('prompts', len(prompts), flush=True)
    results = {}
    for d in prompts:
        name = os.path.basename(d.rstrip('/'))
        first = load_pt(d + 'hidden_states_first.pt')
        last = load_pt(d + 'hidden_states_last.pt')
        res = {}
        for tag, X, ded in [('first', first, True), ('last', last, False)]:
            Xn = prep(X, n=args.n_main, dedupe=ded)
            c = d2_curve(Xn, n_eps=22, theiler=0)
            res[tag] = dict(n=int(len(Xn)), d2=c['d2'], r2=c['r2'], fit_range=c['fit_range'],
                            eps=c['eps'], C=c['C'])
            print(name, tag, 'D2', round(c['d2'], 3), 'R2', round(c['r2'], 3), flush=True)
        results[name] = res

    # convergence and Theiler for 3 prompts
    conv = {}
    for d in prompts[:3]:
        name = os.path.basename(d.rstrip('/')); conv[name] = {}
        for tag, fn, ded in [('first', 'hidden_states_first.pt', True), ('last', 'hidden_states_last.pt', False)]:
            X = load_pt(d + fn)
            rows = []
            for n in [500, 1000, 2000, 3000]:
                if len(X) < n: continue
                c = d2_curve(prep(X, n=n, dedupe=ded), n_eps=20, theiler=0)
                rows.append((n, c['d2'], c['r2']))
            conv[name][tag] = rows
            print('conv', name, tag, rows, flush=True)

    # vocab
    W = load_embedding(); print('embedding', W.shape, flush=True)
    rng = np.random.default_rng(0); idx = rng.choice(len(W), size=min(5000, len(W)), replace=False)
    cV = d2_curve(prep(W[idx]), n_eps=25, theiler=0)
    print('vocab D2', cV['d2'], 'R2', cV['r2'], flush=True)

    # RQA on 3 last-layer trajectories
    rq = {}
    for d in prompts[:3]:
        name = os.path.basename(d.rstrip('/')); X = prep(load_pt(d + 'hidden_states_last.pt'), n=2000)
        rq[name] = {f'W{W}': rqa(X, target_rr=0.02, theiler=W) for W in [0, 20]}
        print('rqa', name, rq[name], flush=True)

    json.dump(dict(prompts=results, conv=conv, vocab=dict(n=int(len(idx)), d2=cV['d2'], r2=cV['r2'],
                                                           eps=cV['eps'], C=cV['C']), rqa=rq),
              open(os.path.join(OUT, 'd2_rqa.json'), 'w'), indent=2)

    # plots
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d0 = prompts[0]; nm = os.path.basename(d0.rstrip('/'))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for tag in ['first', 'last']:
        r = results[nm][tag]
        axs[0].loglog(r['eps'], r['C'], marker='.', ms=3, lw=1, label=tag)
        e = np.asarray(r['eps']); C = np.asarray(r['C'])
        slope = np.gradient(np.log(C), np.log(e))
        axs[1].plot(np.log10(e), slope, marker='.', ms=3, lw=1, label=tag)
    axs[0].loglog(cV['eps'], cV['C'], marker='.', ms=3, lw=1, label='vocab')
    e = np.asarray(cV['eps']); C = np.asarray(cV['C']); slope = np.gradient(np.log(C), np.log(e))
    axs[1].plot(np.log10(e), slope, marker='.', ms=3, lw=1, label='vocab')
    axs[0].set_xlabel('epsilon'); axs[0].set_ylabel('C(epsilon)'); axs[0].legend(); axs[0].grid(alpha=.25)
    axs[1].set_xlabel('log10 epsilon'); axs[1].set_ylabel('local slope'); axs[1].legend(); axs[1].grid(alpha=.25)
    fig.suptitle(f'Correlation sum and local slope: {nm}'); fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'd2_curves.png'), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag in ['first', 'last']:
        for name, rows in conv.items():
            ns = [r[0] for r in rows[tag]]; ds = [r[1] for r in rows[tag]]
            ax.plot(ns, ds, marker='o', lw=1, alpha=.7, label=f'{tag} {name[:12]}')
    ax.set_xlabel('N'); ax.set_ylabel('D2'); ax.set_title('D2 vs N'); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'd2_vs_N.png'), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = {tag: [results[n][tag]['d2'] for n in results] for tag in ['first', 'last']}
    vals['vocab'] = [cV['d2']]
    ax.boxplot([vals[k] for k in ['first', 'last', 'vocab']], labels=['first', 'last', 'vocab'])
    ax.set_ylabel('D2'); ax.set_title('D2 distribution across prompts'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'd2_distribution.png'), dpi=200); plt.close(fig)
    print('saved', OUT)


if __name__ == '__main__':
    main()
