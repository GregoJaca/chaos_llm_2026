"""In-batch Jacobian measurements for Mamba (negative control, pure PyTorch path).

Every forward batch contains the exact reference in row 0 and perturbed copies
in rows 1..K.  Rows within a batch are bit-identical apart from the
perturbation, so finite differences here are exact to fp32 bit level.

State vector = concatenation over layers of (ssm_state [d_inner, d_state],
conv_state [d_inner, d_conv]).  The one-step map (teacher forced, fixed token)
is h_t = F(h_{t-1}, x_t) with Jacobian J_t.

Experiments
  A  state-perturbation growth: perturb the full state at token t0, teacher
     force H tokens, record the response of the state and the residual stream.
  B  one-step sigma_max of J_t by power iteration (renormalised chain).
  C  diagonal of J_t: sampled (layer, inner-channel, ssm-dim) entries, equal
     to exp(-dt*exp(A_log)); also the block-structure check (a perturbation in
     layer l must produce an exactly zero response in layers < l at step t+1
     through the state path, and the conv register must be nilpotent).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from transformers import AutoModelForCausalLM
try:
    from transformers import MambaCache
except ImportError:
    from transformers.models.mamba.modeling_mamba import MambaCache

from chaos_llm.protocol import resolve_model_path, save_json


def load_mamba(model):
    path = resolve_model_path(model)
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="eager", low_cpu_mem_usage=True).cuda().eval()
    return m


@torch.inference_mode()
def step_once(model, tok_id, cache, pos, B, record_logits=False):
    ids = torch.full((B, 1), int(tok_id), dtype=torch.long, device="cuda")
    o = model(input_ids=ids, cache_params=cache,
              cache_position=torch.tensor([pos], device="cuda"),
              use_cache=True, output_hidden_states=record_logits,
              return_dict=True)
    return o


class StateStore:
    """Baseline states of a token sequence, one token at a time (decode path)."""

    def __init__(self, model, ids):
        cfg = model.config
        L, B = cfg.num_hidden_layers, 1
        cache = MambaCache(cfg, max_batch_size=B, device="cuda", dtype=torch.float32)
        self.ssm = []   # [T][L][d_inner, d_state]
        self.conv = []  # [T][L][d_inner, d_conv]
        self.resid = []  # [T][D] final-layer residual stream
        self.logits = []
        for t, tok in enumerate(ids):
            o = step_once(model, tok, cache, t, B, record_logits=True)
            self.ssm.append([cache.ssm_states[l].clone() for l in range(L)])
            self.conv.append([cache.conv_states[l].clone() for l in range(L)])
            self.resid.append(o.hidden_states[-1][0, 0].detach().float().cpu().numpy())
            self.logits.append(o.logits[0, 0].detach())
        self.L = L
        self.d_inner = cfg.intermediate_size
        self.d_state = cfg.state_size
        self.d_conv = cfg.conv_kernel

    def fill_cache(self, cache, t, B, perturb=None):
        """Load states *after* token t-1 (i.e. entering token t) into a B-row cache."""
        assert 0 <= t < len(self.ssm) + 1
        if t == 0:
            for l in range(self.L):
                cache.ssm_states[l].zero_()
                cache.conv_states[l].zero_()
        else:
            for l in range(self.L):
                cache.ssm_states[l][:] = self.ssm[t - 1][l]
                cache.conv_states[l][:] = self.conv[t - 1][l]
        if perturb is not None:
            perturb(cache)


def state_norm(store, cache, row):
    s = 0.0
    for l in range(store.L):
        s += float(cache.ssm_states[l][row].float().pow(2).sum())
        s += float(cache.conv_states[l][row].float().pow(2).sum())
    return np.sqrt(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--state-spaces--mamba-130m-hf")
    ap.add_argument("--tokens-from", required=True, help="npz with prompt_ids + gen_ids (an a1.npz or traj npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0-list", default="80,150,220")  # absolute token index where the state is kicked (prompt included)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--diag-t-list", default="70,100,150,220,290")
    ap.add_argument("--diag-samples", type=int, default=24, help="(layer, channel) triples per batch row")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0all = time.time()

    model = load_mamba(args.model)
    cfg = model.config
    L = cfg.num_hidden_layers
    d = np.load(args.tokens_from)
    ids = list(d["prompt_ids"]) + list(d["gen_ids"])
    P = int(len(d["prompt_ids"]))
    T_total = len(ids)
    print("tokens: P=%d total=%d layers=%d d_inner=%d d_state=%d" % (
        P, T_total, L, cfg.intermediate_size, cfg.state_size), flush=True)

    store = StateStore(model, ids)
    state_scale = np.sqrt(np.mean([np.mean([float(s.float().pow(2).sum()) for s in ss]) for ss in store.ssm[:50]]))
    print("state store done %.1fs; typical ||state per layer|| ~ %.3g" % (time.time() - t0all, np.sqrt(state_scale)), flush=True)

    rng = np.random.default_rng(args.seed)
    g = torch.Generator(device="cuda").manual_seed(args.seed)

    def rand_dirs(k):
        v = torch.randn(L, cfg.intermediate_size, cfg.state_size + cfg.conv_kernel,
                        device="cuda", generator=g)
        v = v / v.norm(dim=(1, 2), keepdim=True)
        return v  # [k, L, d_inner, d_state+d_conv]

    # ---------------- Experiment A: state kick growth ----------------
    A_t0 = [int(x) for x in args.t0_list.split(",")]
    H = args.horizon
    K = args.dirs
    res_A = {}
    for t0 in A_t0:
        if t0 + H > T_total:
            H = T_total - t0
        cache = MambaCache(cfg, max_batch_size=K + 1, device="cuda", dtype=torch.float32)
        V = rand_dirs(K)
        def perturb(cache, V=V):
            for k in range(K):
                for l in range(L):
                    cache.ssm_states[l][k + 1] += (args.eps * V[k, l, :, :cfg.state_size])
                    cache.conv_states[l][k + 1] += (args.eps * V[k, l, :, cfg.state_size:])
        store.fill_cache(cache, t0, K + 1, perturb=perturb)
        # baseline states for comparison inside the loop are row 0 at each step
        dstate = np.zeros((K, H), np.float32)     # ||delta full state||
        dh_last = np.zeros((K, H), np.float32)    # ||delta final-layer residual||
        dh_layer = np.zeros((K, 4, H), np.float32)
        dm = np.zeros((K, H), np.float32)         # margin change (top1-top2 of ref row)
        for j, t in enumerate(range(t0, t0 + H)):
            o = step_once(model, ids[t], cache, t, K + 1, record_logits=True)
            # reference logits for margins
            zref = o.logits[0, 0]
            top2 = torch.topk(zref, k=2)
            m_ref = float(top2.values[0] - top2.values[1])
            i1, i2 = int(top2.indices[0]), int(top2.indices[1])
            for k in range(K):
                zk = o.logits[k + 1, 0]
                dm[k, j] = float((zk[i1] - zk[i2]) - m_ref)
                s = 0.0
                for l in range(L):
                    s += float((cache.ssm_states[l][k + 1] - cache.ssm_states[l][0]).pow(2).sum())
                    s += float((cache.conv_states[l][k + 1] - cache.conv_states[l][0]).pow(2).sum())
                dstate[k, j] = np.sqrt(s)
                hdiff = o.hidden_states[-1][k + 1, 0] - o.hidden_states[-1][0, 0]
                dh_last[k, j] = float(hdiff.norm())
                for li, l in enumerate([0, L // 3, 2 * L // 3, L]):
                    hd = o.hidden_states[l][k + 1, 0] - o.hidden_states[l][0, 0]
                    dh_layer[k, li, j] = float(hd.norm())
        res_A[t0] = dict(dstate=dstate, dh_last=dh_last, dh_layer=dh_layer, dm=dm,
                         t0=t0, H=H, eps=args.eps)
        med = np.median(dstate[:, 0])
        print("A t0=%d: ||dh_state(0)||=%.4g -> t+50 %.4g -> t+%d %.4g  (median over %d dirs)" % (
            t0, med, np.median(dstate[:, min(50, H - 1)]), H - 1, np.median(dstate[:, H - 1]), K), flush=True)
        del cache
        torch.cuda.empty_cache()

    # ---------------- Experiment C: diagonal of J_t ----------------
    ts = [int(x) for x in args.diag_t_list.split(",")]
    per_t = args.diag_samples  # (layer, channel) pairs per batch; d_state slice all 16
    diag_layers = sorted(set([0, L // 4, L // 2, 3 * L // 4, L - 1] + [l for l in range(0, L, max(1, L // 8))]))
    C_t = np.zeros((len(ts), len(diag_layers), per_t, cfg.state_size), np.float32)  # log J_ii
    C_below = np.zeros((len(ts), len(diag_layers)), np.float32)  # max |response| in layers < l
    eps_c = 3e-2
    for ti, t in enumerate(ts):
        for li, l in enumerate(diag_layers):
            chans = rng.choice(cfg.intermediate_size, size=per_t, replace=False)
            B = per_t + 1
            cache = MambaCache(cfg, max_batch_size=B, device="cuda", dtype=torch.float32)
            store.fill_cache(cache, t, B)
            for k, ch in enumerate(chans):
                cache.ssm_states[l][k + 1, ch, :] += eps_c
            o = step_once(model, ids[t], cache, t, B, record_logits=False)
            for k, ch in enumerate(chans):
                delta = cache.ssm_states[l][k + 1, ch, :] - cache.ssm_states[l][0, ch, :]
                C_t[ti, li, k, :] = np.log(np.maximum(delta.abs().cpu().numpy(), 1e-30) / eps_c)
            # block check: layers below l must be untouched
            below = 0.0
            for l2 in range(l):
                for k in range(per_t):
                    below = max(below, float((cache.ssm_states[l2][k + 1] - cache.ssm_states[l2][0]).abs().max()))
            C_below[ti, li] = below
            del cache, o
        print("C t=%d: median log J_ii=%.4f min=%.4f  max below-layer leak=%.3g" % (
            t, np.median(C_t[ti]), C_t[ti].min(), C_below[ti].max()), flush=True)
        torch.cuda.empty_cache()

    # ---------------- Experiment B: sigma_max of J_t (power iteration) ----------------
    B_sigma = np.zeros(len(ts), np.float32)
    B_chain = np.zeros((len(ts), 16), np.float32)  # log growth per iter of the chained product
    for ti, t in enumerate(ts):
        v = rand_dirs(1)[0]  # [L, d_inner, d_state+d_conv]
        v = v / v.norm()
        logs = []
        cache = MambaCache(cfg, max_batch_size=2, device="cuda", dtype=torch.float32)
        for it in range(16):
            def perturb(cache, v=v):
                for l in range(L):
                    cache.ssm_states[l][1] += v[l, :, :cfg.state_size]
                    cache.conv_states[l][1] += v[l, :, cfg.state_size:]
            store.fill_cache(cache, t, 2, perturb=perturb)
            step_once(model, ids[t], cache, t, 2, record_logits=False)
            dv = [(cache.ssm_states[l][1] - cache.ssm_states[l][0]) for l in range(L)]
            dc = [(cache.conv_states[l][1] - cache.conv_states[l][0]) for l in range(L)]
            n = float(np.sqrt(sum(float(x.pow(2).sum()) for x in dv) + sum(float(x.pow(2).sum()) for x in dc)))
            logs.append(np.log(n))
            v = torch.stack([torch.cat([dv[l].squeeze(0), dc[l].squeeze(0)], dim=-1) for l in range(L)]) / n
        B_sigma[ti] = np.exp(np.mean(logs[-8:]))
        B_chain[ti] = logs
        print("B t=%d: sigma_max(J_t)=%.4f  (logs: %s)" % (
            t, B_sigma[ti], " ".join("%.3f" % x for x in logs)), flush=True)
        del cache
        torch.cuda.empty_cache()

    out = os.path.join(args.out, "jac.npz")
    np.savez(out,
             A_dstate={t0: r["dstate"] for t0, r in res_A.items()},
             A_dh_last={t0: r["dh_last"] for t0, r in res_A.items()},
             A_dh_layer={t0: r["dh_layer"] for t0, r in res_A.items()},
             A_dm={t0: r["dm"] for t0, r in res_A.items()},
             A_t0=np.asarray(A_t0), A_H=np.asarray([r["H"] for r in res_A.values()]),
             A_eps=args.eps,
             C_logdiag=C_t, C_below=C_below, C_layers=np.asarray(diag_layers), C_t=np.asarray(ts),
             B_sigma=B_sigma, B_chain=B_chain,
             d_inner=cfg.intermediate_size, d_state=cfg.state_size)
    save_json(os.path.join(args.out, "meta.json"), dict(
        model=args.model, tokens_from=args.tokens_from, prompt_len=P,
        t0_list=A_t0, horizon=args.horizon, dirs=K, eps=args.eps,
        diag_t_list=ts, elapsed_sec=round(time.time() - t0all, 1)))
    print("saved", out, "elapsed %.1fs" % (time.time() - t0all), flush=True)


if __name__ == "__main__":
    main()
