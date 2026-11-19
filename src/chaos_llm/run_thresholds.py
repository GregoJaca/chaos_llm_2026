"""Free-run validation of predicted T(r,u), with in-batch r=0 reference."""
import argparse, json, os
import numpy as np, torch
from chaos_llm.protocol import generate, load_model, prompt_ids, save_json, set_window


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.run, "a1.npz"))
    meta = json.load(open(os.path.join(args.run, "meta.json")))
    simplex = d["simplex"]; sub = d["subspace"]
    P = int(meta["prompt_len"])
    T = int(meta["max_new_tokens"]); K = simplex.shape[0]
    radii = [float(x) for x in args.radii.split(",")]

    model, tok, _ = load_model(args.model)
    set_window(model, meta["window"])
    D = model.config.hidden_size
    text = meta["prompt"]
    pids = prompt_ids(tok, text)
    with torch.inference_mode():
        base = model.get_input_embeddings()(pids)[0]
    sub_t = torch.as_tensor(sub, device="cuda", dtype=torch.long)

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
                    delta = torch.zeros(P * D, device="cuda")
                    delta.index_copy_(0, sub_t, torch.as_tensor(simplex[row["di"]], device="cuda") * r)
                    E[j] += delta.view(P, D)
            gen = generate(model, E, T).float().cpu().numpy()
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
              dict(run=args.run, radii=radii, obs=obs.tolist(), pred=pred[pred_rows].tolist(),
                   n_dirs=K, window=meta["window"], prompt_len=P, T=T))
    print("saved", args.out)


if __name__ == "__main__":
    main()
