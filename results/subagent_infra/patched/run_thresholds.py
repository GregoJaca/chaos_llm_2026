"""Free-run validation of predicted T(r,u), with in-batch r=0 reference.

Patched for multi-model use (protocol_general): dtype auto, bf16-safe perturbation
(cast to the embedding dtype before the in-place add), device from protocol, and a
cross-model guard when the source a1 run used another model/tokenizer.
"""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol_general import (
    DEVICE, generate, load_model, model_info, prompt_ids, save_json, set_window,
)


def first_mismatch(ref, row):
    n = min(len(ref), len(row))
    mm = np.where(row[:n] != ref[:n])[0]
    return int(mm[0]) if len(mm) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--radii", default="1e-4,1e-3,1e-2")
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    simplex = d["simplex"]; sub = d["subspace"]
    P = int(meta["prompt_len"])
    T = int(meta["max_new_tokens"]); K = simplex.shape[0]
    radii = [float(x) for x in args.radii.split(",")]

    model, tok, _ = load_model(args.model, dtype=args.dtype)
    info = model_info(model, tok=tok)
    D = info["hidden_size"]
    if int(d["prompt_ids"].max()) >= info["vocab_size"] or int(d["gen_ids"].max()) >= info["vocab_size"]:
        raise ValueError("the source run contains token ids beyond this model's vocab: "
                         "it was produced by another model -- free-run reuse is invalid")
    set_window(model, meta["window"])
    text = meta["prompt"]
    if not text or text.startswith("@"):
        raise ValueError(f"meta['prompt'] is not usable text ({text!r}). It is probably a "
                         "meta file written by the ORIGINAL run_a1_long.py, which stored the "
                         "@filename string instead of the prompt (fixed in the patched scripts).")
    pids = prompt_ids(tok, text)
    if pids.shape[1] != P:
        print(f"WARNING: tokenizing meta['prompt'] gives P={pids.shape[1]}, "
              f"meta says prompt_len={P}; the free-run baseline may not match the "
              f"teacher-forced run (different tokenizer/model?).", flush=True)
    with torch.inference_mode():
        base = model.get_input_embeddings()(pids)[0]
    sub_t = torch.as_tensor(sub, device=DEVICE, dtype=torch.long)

    obs = np.full((len(radii), K), -1, dtype=np.int32)
    all_ref = {}
    all_gen = {}
    for ri, r in enumerate(radii):
        for start in range(0, K, args.chunk):
            idx = list(range(start, min(start + args.chunk, K)))
            plan = [dict(kind="ref", di=-1)] + [dict(kind="pert", di=di) for di in idx]
            B = len(plan)
            E = base[None].expand(B, -1, -1).clone()
            for j, row in enumerate(plan):
                if row["kind"] == "pert":
                    delta = torch.zeros(P * D, device=DEVICE, dtype=torch.float32)
                    delta.index_copy_(0, sub_t,
                                      torch.as_tensor(simplex[row["di"]], device=DEVICE) * r)
                    # fp32 accumulate, round once to the model dtype (bf16-safe)
                    E[j] = (E[j].float() + delta.view(P, D)).to(E.dtype)
            gen = generate(model, E, T).cpu().numpy()
            ref = gen[0]
            all_ref[f"{ri}_{start}"] = ref
            all_gen[f"{ri}_{start}"] = gen
            for j, row in enumerate(plan):
                if row["kind"] == "pert":
                    obs[ri, row["di"]] = first_mismatch(ref, gen[j])
            print(f"r={r:g} dirs {start}-{start+len(idx)-1}: T={[obs[ri,di] for di in idx]}", flush=True)
            del E, gen
            torch.cuda.empty_cache()

    pred = d["pred_T"]
    # pred rows correspond to meta radii; map requested radii to nearest meta radius
    mradii = np.asarray(meta["radii"])
    pred_rows = np.array([int(np.argmin(np.abs(mradii - r))) for r in radii])
    np.savez(os.path.join(args.out, "free_run.npz"), obs=obs, pred=pred[pred_rows],
             radii=np.asarray(radii), ref_ids=np.array(list(all_ref.values()), dtype=object),
             gen_ids=np.array(list(all_gen.values()), dtype=object))
    save_json(os.path.join(args.out, "summary.json"),
              dict(run=args.run, model=args.model, dtype=info["dtype"],
                   radii=radii, obs=obs.tolist(), pred=pred[pred_rows].tolist(),
                   n_dirs=K, window=meta["window"], prompt_len=P, T=T))
    print("saved", args.out)


if __name__ == "__main__":
    main()
