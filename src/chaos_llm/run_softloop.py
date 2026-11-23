"""Soft-generation loop: feed back the expected embedding of the model's own
next-token distribution (a "soft token"), growing context, greedy otherwise.

System under study (new for this repo; complement of the greedy / teacher-forced
systems already measured):

    z_t   = model(prompt + e_0 ... e_{t-1})[-1]              (logits, V)
    p_t   = softmax(z_t / T)                                (temperature T)
    e_t   = E^T p_t = sum_v p_t(v) * E[v]                   (input embeddings)

T -> 0 recovers greedy decoding exactly (e_t -> embedding of the argmax token);
at tiny T the loop's argmax sequence is checked against model.generate.
Teacher forcing on the recorded e_t sequence (mode=replay) isolates the network
propagation from the feedback, so

    lambda_loop(T) - lambda_replay(T)  =  the feedback/readout contribution

which is the quantity the fixed-prefix soft-token literature attributes chaos to.

Rows per batch: 1 in-batch reference (r=0) + K perturbed rows, one fixed batch
shape for the whole run (batch-layout reproducibility).  The prompt
perturbation is the repo-standard sparse simplex (K nonzero entries of the
flattened P x d prompt-embedding tensor, total norm r).

Recorded per generated step t: reference margin / entropy / argmax, per-row
argmax (flip record), ||dh|| at saved layers (response at generated position t),
reference hidden norm, the reference soft token e_t (for replay), and the
readout factor sigma_max(CrossCov_p(e, w)) (from the top-k mass tokens).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chaos_llm.protocol import load_model, set_window, save_json


def read_prompt(spec):
    if spec.startswith("@"):
        line = open(spec[1:]).read().splitlines()[0]
        return line.split("\t", 1)[1] if "\t" in line else line
    return spec


def sparse_simplex(k, total, seed):
    """k rows, each unit norm, disjoint supports over `total` flattened coords."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=k, replace=False)
    eye = np.eye(k, dtype=np.float32)
    v = eye - np.ones((k, k), dtype=np.float32) / k
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return idx, v


def topk_crosscov_sigma(p, E, W, k=64, iters=30, return_M=False, sv_out=0):
    """sigma_max of sum_v p_v (E_v - ebar)(W_v - wbar)^T restricted to the
    top-k mass tokens (covers > 1-1e-6 of the distribution in practice).
    Clean power iteration on M^T M (fp32 accumulation, no big intermediate)."""
    pv, iv = torch.topk(p, k)
    if float(pv[-1]) <= 0.0:
        k = int((pv > 0).sum().item())
        pv, iv = pv[:k], iv[:k]
    e = E[iv]
    w = W[iv]
    ebar = (pv[:, None] * e).sum(0)
    wbar = (pv[:, None] * w).sum(0)
    # accumulate M = sum_i pv_i (e_i-ebar)(w_i-wbar)^T row by row
    M = torch.zeros(e.shape[1], w.shape[1], dtype=torch.float32, device=E.device)
    for i in range(e.shape[0]):
        M += pv[i] * torch.outer(e[i] - ebar, w[i] - wbar)
    v = torch.randn(M.shape[1], device=M.device)
    v = v / v.norm()
    dead = False
    for _ in range(iters):
        u = M @ v
        un = float(u.norm())
        if un == 0.0:
            dead = True
            break
        v = M.T @ u
        vn = float(v.norm())
        if vn == 0.0:
            dead = True
            break
        v = v / vn
    if sv_out > 0:
        sv = torch.zeros(sv_out, dtype=torch.float32, device=M.device)
        if not dead:
            Wk = M
            for j in range(sv_out):
                vv = torch.randn(M.shape[1], device=M.device)
                vv = vv / vv.norm()
                for _ in range(20):
                    uu = Wk @ vv
                    nn = float(uu.norm())
                    if nn == 0.0:
                        break
                    vv = Wk.T @ uu
                    nv = float(vv.norm())
                    if nv == 0.0:
                        break
                    vv = vv / nv
                sv[j] = float((Wk @ vv).norm()) if float(vv.norm()) > 0 else 0.0
                uu = Wk @ vv
                Wk = Wk - torch.outer(uu, vv) / max(float(vv.norm() ** 2), 1e-30)
        return (0.0 if dead else float((M @ v).norm())), sv
    if return_M:
        return float((M @ v).norm()), M
    return float((M @ v).norm())


def _cache_kwargs(o):
    """Return the right cache kwarg for the model family (Qwen: past_key_values;
    Mamba (transformers<4.49 style output): cache_params)."""
    pv = getattr(o, "past_key_values", None)
    if pv is not None:
        return {"past_key_values": pv}
    cp = getattr(o, "cache_params", None)
    if cp is not None:
        return {"cache_params": cp}
    return {}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", default="@prompts.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--temp", type=float, default=1.0, help="loop temperature T")
    ap.add_argument("--mode", default="loop", choices=["loop", "replay"])
    ap.add_argument("--replay-file", default="", help="npz with the recorded e_ref (mode=replay)")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--save-layers", default="")
    ap.add_argument("--save-states", action="store_true",
                    help="save the reference row's last-layer hidden state at every step")
    ap.add_argument("--feedback", default="soft",
                    choices=["soft", "argmax", "sample_crn", "sample_ind"],
                    help="what is fed back: soft=expected embedding (default); "
                         "argmax=embedding of the argmax token; "
                         "sample_crn=sampled token, SAME draw for every row "
                         "(common random numbers: isolates the state dynamics); "
                         "sample_ind=each row samples its own token")
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16"],
                    help="model precision (default: protocol auto rule)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    model, tok, path = load_model(args.model, dtype=args.dtype)
    if args.window:
        set_window(model, args.window)
    cfg = model.config
    D = getattr(cfg, "hidden_size", None)
    if D is None:
        D = getattr(getattr(cfg, "text_config", cfg), "hidden_size")
    del cfg
    Emb = model.get_input_embeddings().weight.detach().float()  # [V, d] fp32 view
    embed_dtype = model.get_input_embeddings().weight.dtype      # fp16 models feed fp16
    if hasattr(model, "lm_head") and getattr(model, "lm_head", None) is not None:
        Wout = model.lm_head.weight.detach()
        tied = Wout.shape == Emb.shape and bool(torch.equal(Wout, Emb))
        Wout = Wout.float()
    else:
        Wout, tied = Emb, True

    pids = tok(read_prompt(args.prompt), return_tensors="pt").input_ids.cuda()
    P = int(pids.shape[1])
    K = args.dirs
    B = 1 + K
    steps = args.steps

    idx, srows = sparse_simplex(K, P * D, args.seed)
    idx_t = torch.as_tensor(idx, device="cuda").long()
    srows_t = torch.as_tensor(srows, device="cuda")

    if args.save_layers:
        layers = [int(x) for x in args.save_layers.split(",")]
    else:
        nl = getattr(model.config, "num_hidden_layers", None)
        if nl is None:
            nl = getattr(getattr(model.config, "text_config", model.config),
                         "num_hidden_layers")
        layers = sorted(set([0, nl // 4, nl // 2, 3 * nl // 4, nl]))
    NL = len(layers)

    # baseline greedy run (token feeding), for the T->0 check
    greedy_ids = model.generate(pids, max_new_tokens=steps, do_sample=False,
                                eos_token_id=None, pad_token_id=0)[0, P:].cpu().numpy()

    # batch: 1 reference + K perturbed, prompt perturbation in the prompt rows
    Eb = model.get_input_embeddings()(pids)[0][None].expand(B, -1, -1).clone()
    flat = Eb[1:].reshape(K, -1)
    for k in range(K):
        flat[k, idx_t] += args.r * srows_t[k]

    if args.mode == "replay":
        d = np.load(args.replay_file)
        e_file = torch.as_tensor(d["e_ref"], device="cuda").float()  # [steps, D]
        assert e_file.shape[0] >= steps

    mask = torch.ones((B, P), dtype=torch.long, device="cuda")

    sgen = None
    if args.feedback in ("sample_crn", "sample_ind"):
        sgen = torch.Generator(device="cuda").manual_seed(args.sample_seed)

    dh = np.zeros((K, NL, steps), np.float32)
    h0n = np.zeros((NL, steps), np.float32)
    e_ref = np.zeros((steps, D), np.float32)
    m_ref = np.zeros(steps, np.float32)
    ent_ref = np.zeros(steps, np.float32)
    top1_ref = np.zeros(steps, np.int64)
    top2_ref = np.zeros(steps, np.int64)
    top1_rows = np.zeros((B, steps), np.int64)
    sig_read = np.zeros(steps, np.float32)
    cov64 = np.zeros(steps, np.float32)
    sv_read = np.zeros((16, steps), np.float32)  # top-16 singular values of Sigma_p
    flips = np.zeros((K, steps), np.int8)
    h_ref = np.zeros((steps, D), np.float32) if args.save_states else None
    fed_ids = np.zeros((B, steps), np.int64) - 1  # fed token ids (token modes)

    # initial forward over the (perturbed) prompt; logits predict token 0
    o = model(inputs_embeds=Eb, attention_mask=mask, use_cache=True,
              output_hidden_states=True)

    for t in range(steps):
        # (mamba needs the absolute position when a cache is passed)
        ck = _cache_kwargs(o)
        if "cache_params" in ck:
            ck["cache_position"] = torch.tensor([P + t], device="cuda")
        logits = o.logits[:, -1, :].float()

        # reference statistics (prediction for generated token t)
        zr = logits[0]
        top2 = torch.topk(zr, 2)
        m_ref[t] = float(top2.values[0] - top2.values[1])
        ent_ref[t] = float(-(torch.softmax(zr, -1) * torch.log_softmax(zr, -1)).sum())
        top1_ref[t] = int(top2.indices[0])
        top2_ref[t] = int(top2.indices[1])
        p = torch.softmax(zr / args.temp, -1)
        sig, sv = topk_crosscov_sigma(p, Emb, Wout, sv_out=16)
        sig_read[t] = sig
        sv_read[:, t] = sv.cpu().numpy()
        cov64[t] = float(torch.topk(p, 64).values.sum())
        if args.feedback == "soft":
            e_ref[t] = (p[:, None] * Emb).sum(0).cpu().numpy()
        # token modes: e_ref is recorded after the draw, from what was fed

        # argmax per row (flip record)
        am = torch.argmax(logits, -1)
        top1_rows[:, t] = am.cpu().numpy()
        flips[:, t] = (am[1:].cpu().numpy() != top1_ref[t]).astype(np.int8)

        # next input embedding
        if args.mode == "replay":
            e_next = e_file[t][None].expand(B, -1).contiguous().to(embed_dtype)
        elif args.feedback == "soft":
            pt = torch.softmax(logits / args.temp, -1)  # [B, V] fp32
            e_next = (pt @ Emb).to(embed_dtype)        # [B, D] in the model's dtype
        elif args.feedback == "argmax":
            ids = torch.argmax(logits, -1)
            e_next = Emb[ids].to(embed_dtype)
            fed_ids[:, t] = ids.cpu().numpy()
        elif args.feedback == "sample_crn":
            p_ref = torch.softmax(logits[0] / args.temp, -1)
            tok = int(torch.multinomial(p_ref, 1, generator=sgen).item())
            e_next = Emb[tok][None].expand(B, -1).contiguous().to(embed_dtype)
            e_ref[t] = e_next[0].float().cpu().numpy()
            fed_ids[:, t] = tok
        elif args.feedback == "sample_ind":
            pb = torch.softmax(logits / args.temp, -1)  # [B, V]
            ids = torch.multinomial(pb, 1, generator=sgen).squeeze(1)  # [B]
            e_next = Emb[ids].to(embed_dtype)
            e_ref[t] = e_next[0].float().cpu().numpy()
            fed_ids[:, t] = ids.cpu().numpy()

        # feed at position P+t, then record the response at that position
        mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long, device="cuda")], 1)
        feed = dict(inputs_embeds=e_next[:, None, :], use_cache=True,
                    output_hidden_states=True, **ck)
        if "cache_params" not in ck:      # mamba-style caches need no mask and
            feed["attention_mask"] = mask  # break (HF bug) if one is passed
        o = model(**feed)
        hs = o.hidden_states  # tuple(L+1), each [B, 1, D]
        if h_ref is not None:
            h_ref[t] = hs[-1][0, 0].float().cpu().numpy()
        for li, l in enumerate(layers):
            hl = hs[l]
            h0n[li, t] = float(hl[0, 0].norm())
            for k in range(K):
                dh[k, li, t] = float((hl[k + 1, 0] - hl[0, 0]).norm())

        del logits, hs
        if t % 50 == 0:
            torch.cuda.empty_cache()
            print(f"step {t}/{steps} m={m_ref[t]:.3f} H={ent_ref[t]:.2f} "
                  f"|dh|max={dh[:, -1, t].max():.4g} ({time.time()-t0:.0f}s)", flush=True)

    # T->0 check: the loop's argmax must reproduce token-feeding greedy
    if args.mode == "loop" and args.temp < 1e-3:
        am_match = int((top1_rows[0] == greedy_ids).sum())
        print(f"T->0 CHECK: loop argmax == greedy tokens on {am_match}/{steps} steps",
              flush=True)

    np.savez(os.path.join(args.out, "softloop.npz"),
             dh=dh, h0n=h0n, e_ref=e_ref, m_ref=m_ref, ent_ref=ent_ref,
             top1_ref=top1_ref, top2_ref=top2_ref, top1_rows=top1_rows,
             sig_read=sig_read, sv_read=sv_read, cov64=cov64, flips=flips,
             greedy_ids=greedy_ids, fed_ids=fed_ids,
             prompt_ids=pids.cpu().numpy(), idx=idx, srows=srows,
             **({"h_ref": h_ref} if h_ref is not None else {}))
    save_json(os.path.join(args.out, "meta.json"), dict(
        model=path, prompt_len=P, steps=steps, temp=args.temp, mode=args.mode,
        feedback=args.feedback, window=args.window, dirs=K, r=args.r,
        seed=args.seed, sample_seed=args.sample_seed, layers=layers,
        tied=tied, wall_s=round(time.time() - t0, 1)))
    print("saved", args.out, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
