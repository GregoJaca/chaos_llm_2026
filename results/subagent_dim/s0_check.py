"""s0_check.py -- sanity checks on the raw npz data before analysis.

Verifies: shapes/dtypes, tied flag vs GE==GW, singular values consistent with
Gram eigenvalues, layer-0 acts == embeddings (norms match normE[gen_ids]),
logits ids == first 128 generated tokens, sample_E row indexing.
Writes results into numbers.json under 'check'.
"""
import os
import json
import numpy as np

from dimlib import DIMROOT, OUT, section_numbers, discover_tags


def normE2_sum(v):
    return float((v["normE"].astype(np.float64) ** 2).sum())


def normW2_sum(v):
    return float((v["normW"].astype(np.float64) ** 2).sum())

def main():
    tags = discover_tags()
    out = {"tags_found": tags, "per_tag": {}}
    for tag in tags:
        root = os.path.join(DIMROOT, tag)
        meta = json.load(open(os.path.join(root, "meta.json")))
        v = np.load(os.path.join(root, "vocab.npz"))
        a = np.load(os.path.join(root, "acts.npz"))
        lg = np.load(os.path.join(root, "logits.npz"))
        GE, GW = v["GE"], v["GW"]
        tied = bool(v["tied"][0])
        D, V, L = int(v["D"][0]), int(v["V"][0]), int(v["L"][0])
        rec = {
            "model": meta["model"], "D": D, "V": V, "L": L, "tied": tied,
            "GE_shape": list(GE.shape), "acts_shape": list(a["acts"].shape),
            "tied_flag_matches_GE_eq_GW": bool(np.array_equal(GE, GW)) == tied,
            "max_abs_GE_minus_GW": float(np.abs(GE - GW).max()),
            "sE_descending": bool(np.all(np.diff(v["sE"]) <= 0)),
            "sW_descending": bool(np.all(np.diff(v["sW"]) <= 0)),
        }
        # GE is the DxD FEATURE gram E^T E: trace must equal sum of squared
        # row norms (decisive against a VxV row-gram interpretation).
        rec["trGE_over_sum_normE2"] = float(np.trace(GE.astype(np.float64))
                                            / (normE2_sum(v)))
        rec["trGW_over_sum_normW2"] = float(np.trace(GW.astype(np.float64))
                                            / (normW2_sum(v)))
        # singular values consistent with Gram eigenvalues (spot-check via eigvalsh)
        ev = np.linalg.eigvalsh(GE)[::-1]
        rec["max_rel_err_eigGE_vs_sE2"] = float(
            np.abs(ev - v["sE"]**2).max() / ev[0])
        # layer 0 == embedding: compare norms
        norms, gen_ids = a["norms"], a["gen_ids"]
        rel = np.abs(norms[:, 0, :] - v["normE"][gen_ids]).max() / norms[:, 0].mean()
        rec["layer0_norm_vs_normE_maxrel"] = float(rel)
        rec["layers_key_in_acts"] = a["layers"].tolist()
        rec["acts_Lplus1"] = int(a["acts"].shape[1])
        # logits: ids == first 128 generated tokens of prompt 1?
        ids, pids, gids = lg["ids"], lg["prompt_ids"], a["gen_ids"]
        rec["logits_ids_eq_gen_ids_first128"] = bool(
            np.array_equal(ids, gids[0, : len(ids)]))
        rec["prompt_ids_len"] = int(len(pids))
        rec["logits_shape"] = list(lg["logits"].shape)
        # sample_E indexing: row j == E[j*(V//1024)]?  check via row norms
        step = V // 1024
        idx = np.arange(1024) * step
        sn = np.linalg.norm(v["sample_E"].astype(np.float32), axis=1)
        rec["sampleE_step"] = int(step)
        rec["sampleE_norm_match_maxrel"] = float(
            np.abs(sn - v["normE"][idx]).max() / sn.mean())
        snW = np.linalg.norm(v["sample_W"].astype(np.float32), axis=1)
        rec["sampleW_norm_match_maxrel"] = float(
            np.abs(snW - v["normW"][idx]).max() / snW.mean())
        # ---- forensics: the TOP acts index is the post-final-RMSNORM state.
        # If so, logits[t] = W_U h_top[t] exactly, so the predicted row norm
        # ||W_U h||^2 = h^T GW h must match the recorded ||logits||^2.
        h = a["acts"][0, L, :128].astype(np.float64)
        GWd = GW.astype(np.float64)
        pred = np.einsum("ti,ij,tj->t", h, GWd, h)
        rn2 = (np.linalg.norm(lg["logits"][:128].astype(np.float32),
                              axis=1) ** 2)
        rec["logits_eq_WU_htop_check"] = {
            "corr_pred_vs_rownorm2": float(np.corrcoef(pred, rn2)[0, 1]),
            "mean_ratio": float(rn2.mean() / pred.mean()),
            "interpretation": ("corr~1 and ratio~1 => acts index L is the "
                               "post-final-norm state (raw final block is "
                               "NOT saved; logits feed directly from it)")}
        out["per_tag"][tag] = rec
        print(f"--- {tag}:")
        for k, val in rec.items():
            print(f"    {k}: {val}")
    section_numbers("check", out)
    print("wrote numbers.json [check]")

if __name__ == "__main__":
    main()
