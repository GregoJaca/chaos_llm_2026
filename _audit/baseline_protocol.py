"""Reproducible baseline protocol for LLM divergence experiments.

Key rules:
- The unperturbed (r=0) condition is a ROW IN THE SAME BATCH as the perturbed rows.
- All rows in a batch have the same shape, dtype, window, and code path.
- Fixed-length generation (no EOS truncation) so sequences are comparable.
- Save enough metadata to reproduce and to audit.
- Verify: r=0 rows must be bit-identical to the in-batch reference.

This is deliberately standalone; it does not modify the existing pipeline.
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np
import torch


def resolve_model_path(model):
    from pathlib import Path
    p = Path(model)
    if p.is_file():
        return str(p.resolve())
    if p.is_dir() and (p / "config.json").exists():
        return str(p.resolve())
    if p.is_dir():
        snaps = sorted(glob.glob(str(p / "snapshots" / "*" / "config.json")))
        if snaps:
            return os.path.dirname(snaps[-1])
    # HF cache id
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    cand = cache / ("models--" + model.replace("/", "--"))
    snaps = sorted(glob.glob(str(cand / "snapshots" / "*" / "config.json")))
    if snaps:
        return os.path.dirname(snaps[-1])
    return model


def build_simplex(k):
    eye = np.eye(k, dtype=np.float32)
    v = eye - np.ones((k, k), dtype=np.float32) / k
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-directions", type=int, default=4)
    ap.add_argument("--magnitudes", default="0,1e-4,1e-3")
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solo-check", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prompt = args.prompt
    if prompt.startswith("@"):
        line = open(prompt[1:]).read().splitlines()[0]
        prompt = line.split("\t", 1)[1] if "\t" in line else line
    mpath = resolve_model_path(args.model)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        mpath, torch_dtype=dtype, local_files_only=True,
        attn_implementation=args.attn, low_cpu_mem_usage=True,
    ).cuda().eval()
    model.config.sliding_window = args.window
    model.config.use_sliding_window = True

    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    P = ids.shape[1]
    with torch.no_grad():
        E = model.get_input_embeddings()(ids)[0]  # [P,D]
    D = E.shape[1]

    mags = [float(x) for x in args.magnitudes.split(",") if x.strip() != ""]
    simplex = build_simplex(max(args.num_directions, 1))
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(P * D, size=max(args.num_directions, 1), replace=False).astype(np.int64)
    sub_t = torch.as_tensor(sub, device="cuda", dtype=torch.long)

    # Row plan: reference, duplicate reference, then (direction, magnitude) pairs.
    rows = [dict(kind="reference", r=0.0, direction=-1),
            dict(kind="reference_dup", r=0.0, direction=-1)]
    for d in range(args.num_directions):
        for r in mags:
            if r == 0.0:
                continue
            rows.append(dict(kind="perturbed", r=r, direction=d))

    def generate(row_plan):
        B = len(row_plan)
        Eb = E[None].expand(B, -1, -1).clone()
        for i, row in enumerate(row_plan):
            r = row["r"]
            if r == 0.0:
                continue
            delta = torch.zeros(P * D, device="cuda", dtype=Eb.dtype)
            delta.index_copy_(0, sub_t, torch.as_tensor(simplex[row["direction"]], device="cuda", dtype=Eb.dtype) * r)
            Eb[i] += delta.view(P, D)
        am = torch.ones((B, P), dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model.generate(inputs_embeds=Eb, attention_mask=am,
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=None, pad_token_id=0)
        return out.cpu().numpy()

    t0 = time.time()
    batched = generate(rows)
    elapsed = time.time() - t0

    ref = batched[0]
    checks = {}
    for i, row in enumerate(rows):
        if row["r"] == 0.0:
            checks[f"row{i}_{row['kind']}_equals_reference"] = bool(np.array_equal(batched[i], ref))
    pert_first = {}
    for i, row in enumerate(rows):
        if row["r"] == 0.0:
            continue
        mm = np.where(batched[i] != ref)[0]
        pert_first[f"dir{row['direction']}_r{row['r']:g}"] = int(mm[0]) if len(mm) else -1

    solo = None
    if args.solo_check:
        solo = generate([dict(kind="solo_reference", r=0.0, direction=-1)])[0]
        checks["solo_reference_equals_batched_reference"] = bool(np.array_equal(solo, ref))
        mm = np.where(solo != ref)[0]
        checks["solo_first_mismatch"] = int(mm[0]) if len(mm) else -1

    meta = dict(
        model=args.model, model_path=mpath, dtype=args.dtype, attn=args.attn,
        window=args.window, max_new_tokens=args.max_new_tokens,
        batch_size=len(rows), num_directions=args.num_directions, magnitudes=mags,
        seed=args.seed, prompt=prompt, prompt_len=int(P), hidden_dim=int(D),
        torch=torch.__version__, transformers=__import__("transformers").__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        elapsed_sec=round(elapsed, 2), checks=checks, first_mismatch=pert_first,
        git_commit=git_commit(os.path.expanduser("~/LLM/chaos_llm_2026")),
        time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    np.savez(os.path.join(args.out, "tokens.npz"),
             prompt_ids=ids[0].cpu().numpy(), reference_ids=ref,
             row_ids=batched, row_meta=np.array(json.dumps(rows)))
    with open(os.path.join(args.out, "protocol.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
