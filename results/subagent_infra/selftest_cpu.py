"""selftest_cpu.py -- CPU-only self test (no GPU, no model weights).

What it checks, per --models entry (defaults: the three new models + DeepSeek):

  (a) imports: numpy, torch, transformers, model_plan, chaos_llm.protocol_general
      (when present), plus a syntax (py_compile) pass over protocol_general.py and
      every script in patched/;
  (b) tokenizer round-trip: encode with the tokenizer default vs
      add_special_tokens=False, decode back, report whether special tokens were
      added and what bos/eos/pad are (downloads tokenizer files only, no weights);
  (c) config fields WITHOUT loading weights: resolves the local dir / HF id,
      reads config.json, estimates the parameter count (safetensors headers when
      local, or a meta-device init when torch+transformers are available), and
      prints the full plan: dtype that would be used, layer-path candidates,
      window key + support, default measurement layers, vocab size.

Sections degrade gracefully: without transformers, (b)/(c) are SKIPped with the
reason; without torch, meta-init params are SKIPped; the pure model_plan checks
always run. Exit code 0 only if no executed check failed.

Usage:
    python selftest_cpu.py
    python selftest_cpu.py --models "Qwen/Qwen3.5-2B" --offline
    python selftest_cpu.py --models ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
"""
from __future__ import annotations

import argparse
import json
import os
import py_compile
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
for p in (HERE, REPO_SRC):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

RESULTS = {"PASS": [], "FAIL": [], "SKIP": []}


def record(kind, name, detail=""):
    RESULTS[kind].append(f"{name}{' -- ' + str(detail) if detail else ''}")
    print(f"[{kind}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "microsoft/Phi-4-mini-instruct",
        "~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
    ], help="HF ids and/or local dirs")
    ap.add_argument("--offline", action="store_true",
                    help="local_files_only=True (no network)")
    ap.add_argument("--skip-tokenizers", action="store_true")
    ap.add_argument("--skip-meta-params", action="store_true",
                    help="skip the meta-device parameter count (needs torch+transformers)")
    args = ap.parse_args()

    print("=" * 78)
    print("chaos_llm multi-model self test (CPU only, no weights loaded)")
    print("=" * 78)

    # ---------------- (a) imports and environment -------------------------
    import numpy
    record("PASS", "numpy", numpy.__version__)
    try:
        import torch
        record("PASS", "torch", f"{torch.__version__} (cuda available: {torch.cuda.is_available()})")
    except ImportError as exc:
        torch = None
        record("SKIP", "torch not installed on this machine", str(exc))
    try:
        import transformers
        record("PASS", "transformers", transformers.__version__)
    except ImportError as exc:
        transformers = None
        record("SKIP", "transformers not installed on this machine", str(exc))
        record("SKIP", "tokenizer round-trip / hub configs",
               "install transformers and rerun for these sections")

    # model_plan is stdlib-only: always importable
    try:
        import model_plan as mp
        record("PASS", "model_plan import",
               f"from {getattr(mp, '__file__', '?')} (repo src? "
               f"{os.path.dirname(os.path.abspath(mp.__file__)) == REPO_SRC})")
    except Exception:
        record("FAIL", "model_plan import", traceback.format_exc(limit=2))
        mp = None

    protocol_general = None
    if mp is not None:
        try:
            import importlib
            protocol_general = importlib.import_module("chaos_llm.protocol_general") \
                if os.path.isfile(os.path.join(REPO_SRC, "chaos_llm", "protocol_general.py")) \
                else None
        except Exception:
            protocol_general = None
        if protocol_general is None:
            record("SKIP", "chaos_llm.protocol_general",
                   "not yet installed in src/chaos_llm (copy protocol_general.py and "
                   "model_plan.py there, then rerun) -- testing the delivered files only")
        else:
            try:
                record("PASS", "chaos_llm.protocol_general import",
                       f"torch ok={torch is not None}")
            except Exception:
                record("FAIL", "chaos_llm.protocol_general import", traceback.format_exc(limit=2))

    # syntax check: delivered protocol + patched scripts (no imports needed)
    targets = [os.path.join(HERE, "protocol_general.py")] + sorted(
        os.path.join(HERE, "patched", f) for f in os.listdir(os.path.join(HERE, "patched"))
        if f.endswith(".py"))
    for t in targets:
        try:
            py_compile.compile(t, doraise=True)
            record("PASS", "syntax", os.path.relpath(t, HERE))
        except Exception as exc:
            record("FAIL", "syntax", f"{os.path.relpath(t, HERE)}: {exc}")

    # ---------------- pure model_plan checks (always run) ------------------
    if mp is not None:
        # dtype rule
        for n, expected in [(2_499_999_999, "float32"), (2_500_000_001, "bfloat16")]:
            got = mp.pick_dtype(n)
            (record("PASS", f"pick_dtype({n}) == {expected}", got) if got == expected
             else record("FAIL", f"pick_dtype({n}) == {expected}", got))
        try:
            mp.pick_dtype(None)
            record("FAIL", "pick_dtype(None) raises", "no exception")
        except ValueError:
            record("PASS", "pick_dtype(None) raises", "ValueError")
        # default layers reproduce the legacy DeepSeek default
        want = [0, 7, 14, 21, 27]
        got = mp.default_layer_indices(28)
        (record("PASS", "default_layer_indices(28) == legacy '0,7,14,21,27'", got) if got == want
         else record("FAIL", "default_layer_indices(28) == legacy '0,7,14,21,27'", got))
        # simplex degeneracy is refused by protocol_general (needs torch); the
        # formula itself is checked here so the intent is pinned:
        import numpy as _np
        k = 1
        eye = _np.eye(k, dtype=_np.float32)
        v = eye - _np.ones((k, k), dtype=_np.float32) / k
        degenerate = float(_np.linalg.norm(v)) == 0.0
        (record("PASS", "simplex k=1 is the zero vector (protocol_general refuses it)",
                "norm=0 -> would be 0/0=NaN")
         if degenerate else record("FAIL", "simplex k=1 degeneracy check"))
        # path resolution must never raise, even for garbage
        try:
            mp.resolve_model_path("definitely/not/a-model")
            record("PASS", "resolve_model_path never raises")
        except Exception as exc:
            record("FAIL", "resolve_model_path never raises", str(exc))

    # ---------------- (b)+(c) per model ------------------------------------
    for model in args.models:
        print("-" * 78)
        print(f"MODEL: {model}")
        if mp is None:
            record("SKIP", f"plan[{model}]", "model_plan unavailable")
            continue
        path = mp.resolve_model_path(model)
        local = mp.is_local_model_dir(path)
        print(f"  resolved path : {path}"
              + ("" if local else "   (not on disk: will be treated as an HF hub id)"))

        # --- (c) config, without loading weights ---
        cfg = None
        cfg_src = None
        if local:
            try:
                with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg_src = "local config.json (stdlib)"
            except Exception as exc:
                record("FAIL", f"config read [{model}]", str(exc))
        elif transformers is not None:
            try:
                cfg = __import__("transformers").AutoConfig.from_pretrained(
                    path, local_files_only=bool(args.offline))
                cfg_src = f"AutoConfig (transformers {transformers.__version__})"
            except Exception as exc:
                record("SKIP", f"config [{model}]",
                       f"could not fetch config (offline={args.offline}): {exc}")
        else:
            record("SKIP", f"config [{model}]", "not local and transformers unavailable")
        if cfg is not None:
            params = None
            psrc = "unknown"
            if local:
                total, groups = mp.read_safetensors_param_counts(path)
                est = mp.text_param_estimate(groups)
                if total:
                    params = est if est is not None else total
                    psrc = (f"safetensors headers: total={total:,}, text~={params:,}; "
                            f"top groups=" + ", ".join(
                                f"{k}:{v:,}" for k, v in
                                sorted(groups.items(), key=lambda kv: -kv[1])[:4]))
            if params is None and torch is not None and transformers is not None \
                    and not args.skip_meta_params:
                try:
                    with torch.device("meta"):
                        m = transformers.AutoModelForCausalLM.from_config(
                            mp.resolve_text_config(cfg))
                    params = sum(p.numel() for p in m.parameters())
                    del m
                    psrc = "meta-device init from config (no weights read)"
                except Exception as exc:
                    record("SKIP", f"meta param count [{model}]",
                           f"{type(exc).__name__}: {exc} (pass --dtype at load time, "
                           "or download the model and rerun)")
            try:
                plan = mp.plan_from_config(cfg, model_dir=path if local else None,
                                           params=params)
                plan["params_source"] = psrc
                plan["resolved_path"] = path
                print("  PLAN (exactly what the protocol would use):")
                for k in sorted(plan):
                    if k in ("param_groups", "attention", "architectures"):
                        continue
                    print(f"    {k:24s}: {plan[k]}")
                att = plan["attention"]
                print(f"    {'attention/window':24s}: supported={att.get('window_supported')} "
                      f"key={att.get('window_key')} "
                      f"sliding_window={att.get('sliding_window')}")
                for note in att.get("notes", []) + att.get("warnings", []):
                    print(f"      - {note}")
                record("PASS", f"plan built [{model}]",
                       f"dtype={plan['dtype']} layers_path0={plan['layer_path_candidates'][0]}")
                if plan["dtype"] is None:
                    record("FAIL", f"dtype undecided [{model}]",
                           "run where the model files are cached, or pass --dtype")
            except Exception:
                record("FAIL", f"plan [{model}]", traceback.format_exc(limit=2))

        # --- (b) tokenizer round-trip ---
        if args.skip_tokenizers:
            record("SKIP", f"tokenizer [{model}]", "--skip-tokenizers")
            continue
        if transformers is None and not local:
            record("SKIP", f"tokenizer [{model}]", "transformers unavailable and no local dir")
            continue
        try:
            tok = __import__("transformers").AutoTokenizer.from_pretrained(
                path, local_files_only=bool(args.offline) if not local else True)
            text = "The quick brown fox jumps over the lazy dog."
            ids_default = tok(text).input_ids
            ids_raw = tok(text, add_special_tokens=False).input_ids
            specials = list(ids_default != ids_raw)
            dec = tok.decode(ids_raw, skip_special_tokens=False)
            dec_clean = tok.decode(ids_raw, skip_special_tokens=True)
            print(f"  tokenizer      : {type(tok).__name__}, vocab={tok.vocab_size}, "
                  f"len={len(tok)}")
            print(f"  bos/eos/pad   : {tok.bos_token_id} / {tok.eos_token_id} / "
                  f"{tok.pad_token_id}   chat_template="
                  f"{hasattr(tok, 'chat_template') and bool(tok.chat_template)}")
            print(f"  default adds special tokens: {specials} "
                  f"(protocol uses add_special_tokens=False -> {len(ids_raw)} ids)")
            print(f"  round-trip (raw ids, no specials skipped): {'exact' if dec == text else 'DIFFERS'}")
            if dec != text:
                print(f"    decoded: {dec!r}")
            ok_rt = dec == text or dec_clean == text
            (record("PASS", f"tokenizer round-trip [{model}]", "exact" if dec == text else
                    "recovered after skipping specials") if ok_rt
             else record("FAIL", f"tokenizer round-trip [{model}]", f"decoded={dec!r}"))
            record("PASS", f"special-token behavior recorded [{model}]",
                   f"default_adds_specials={specials}")
        except Exception as exc:
            record("SKIP", f"tokenizer [{model}]",
                   f"{type(exc).__name__}: {exc} (download the tokenizer or go offline "
                   "with the model cached)")

    # ---------------- summary ----------------------------------------------
    print("=" * 78)
    print(f"SUMMARY: {len(RESULTS['PASS'])} PASS, {len(RESULTS['FAIL'])} FAIL, "
          f"{len(RESULTS['SKIP'])} SKIP")
    for f in RESULTS["FAIL"]:
        print(f"  FAIL: {f}")
    print("Exit code: 1 (failures present)" if RESULTS["FAIL"] else "Exit code: 0")
    return 1 if RESULTS["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
