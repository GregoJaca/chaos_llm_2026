import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode tokens.npz into text JSON")
    parser.add_argument("--run-dir", required=True, help="Path to run_* folder")
    parser.add_argument("--config", required=True, help="Path to config.yaml (for model path)")
    parser.add_argument("--output", default=None, help="Output JSON filename (default texts.json)")
    parser.add_argument("--skip-special", action="store_true", help="Skip special tokens")
    parser.add_argument("--clean-spaces", action="store_true", help="Clean up spaces")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = _resolve_run_dir(args.run_dir, args.config)

    model_path = cfg["paths"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tokens = _load_tokens(run_dir)
    baseline_ids = tokens["baseline_ids"].tolist()
    perturbed_ids = tokens["perturbed_ids"]
    lengths = tokens["perturbed_lengths"]

    skip_special = bool(args.skip_special or cfg["output"].get("text_skip_special_tokens", True))
    clean_spaces = bool(args.clean_spaces or cfg["output"].get("text_clean_up_spaces", True))

    baseline_text = tokenizer.decode(
        baseline_ids,
        skip_special_tokens=skip_special,
        clean_up_tokenization_spaces=clean_spaces,
    )

    perturbed_texts = []
    for i in range(perturbed_ids.shape[0]):
        seq_len = int(lengths[i])
        seq = perturbed_ids[i, :seq_len].tolist()
        text = tokenizer.decode(
            seq,
            skip_special_tokens=skip_special,
            clean_up_tokenization_spaces=clean_spaces,
        )
        perturbed_texts.append(text)

    output_name = args.output or cfg["output"].get("text_filename", "texts.json")
    output_path = os.path.join(run_dir, output_name)

    payload = {
        "baseline_text": baseline_text,
        "perturbed_texts": perturbed_texts,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
