import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chaos_llm.config import load_config  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _resolve_run_dir(run_dir: str, cfg_path: str) -> str:
    if os.path.isdir(run_dir):
        return run_dir
    output_dir = None
    if cfg_path and os.path.isfile(cfg_path):
        cfg = load_config(cfg_path)
        output_dir = cfg["paths"]["output_dir"]
    if output_dir:
        candidate = os.path.join(output_dir, run_dir)
        if os.path.isdir(candidate):
            return candidate
    candidate = os.path.join("outputs", run_dir)
    if os.path.isdir(candidate):
        return candidate
    return run_dir


def _load_tokens(run_dir: str) -> dict:
    path = os.path.join(run_dir, "tokens.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing tokens.npz in {run_dir}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _format_tokens(token_ids: np.ndarray, tokenizer, decode: bool) -> str:
    if not decode:
        return "[" + ", ".join(str(x) for x in token_ids.tolist()) + "]"
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    return "[" + ", ".join(tokens) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print token sequences from a run folder")
    parser.add_argument("--run-dir", required=True, help="Path to run_* folder")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max tokens to print per sequence")
    parser.add_argument("--decode", action="store_true", help="Convert ids to token strings")
    parser.add_argument("--config", default=None, help="Path to config.yaml (for tokenizer)")
    parser.add_argument("--model-path", default=None, help="Model path for tokenizer")
    parser.add_argument("--no-baseline", action="store_true", help="Skip printing baseline")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.config)

    tokenizer = None
    if args.decode:
        model_path = args.model_path
        if not model_path and args.config:
            cfg = load_config(args.config)
            model_path = cfg["paths"]["model_path"]
        if not model_path:
            raise ValueError("Provide --model-path or --config when using --decode")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tokens = _load_tokens(run_dir)
    baseline_ids = tokens["baseline_ids"]
    perturbed_ids = tokens["perturbed_ids"]
    lengths = tokens["perturbed_lengths"]

    max_tokens = args.max_tokens

    if not args.no_baseline:
        base_slice = baseline_ids[:max_tokens]
        print("baseline:")
        print(_format_tokens(base_slice, tokenizer, args.decode))
        print("-")

    for i in range(perturbed_ids.shape[0]):
        seq_len = int(lengths[i])
        end = min(seq_len, max_tokens)
        seq = perturbed_ids[i, :end]
        print(f"sequence_{i}:")
        print(_format_tokens(seq, tokenizer, args.decode))
        print("-")


if __name__ == "__main__":
    main()
