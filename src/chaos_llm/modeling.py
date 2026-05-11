import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_file():
        return str(path)
    if path.is_dir():
        if (path / "config.json").exists():
            return str(path)
        snapshots = path / "snapshots"
        if snapshots.exists() and snapshots.is_dir():
            candidates = []
            for child in snapshots.iterdir():
                if (child / "config.json").exists():
                    candidates.append(child)
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                return str(latest)
    return model_path


def load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    resolved_path = _resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def apply_attention_overrides(model: Any, sliding_window: int, use_sliding_window: bool) -> None:
    if hasattr(model, "config"):
        if sliding_window is not None:
            if hasattr(model.config, "sliding_window"):
                model.config.sliding_window = int(sliding_window)
            if hasattr(model.config, "use_sliding_window"):
                model.config.use_sliding_window = bool(use_sliding_window)
    if hasattr(model, "generation_config") and sliding_window is not None:
        if hasattr(model.generation_config, "sliding_window"):
            model.generation_config.sliding_window = int(sliding_window)


def build_generation_kwargs(cfg: Dict[str, Any], tokenizer: Any) -> Dict[str, Any]:
    gen_cfg = cfg["generation"]
    kwargs = {
        "max_new_tokens": int(gen_cfg["max_new_tokens"]),
        "do_sample": bool(gen_cfg["do_sample"]),
        "temperature": float(gen_cfg["temperature"]),
        "top_k": int(gen_cfg["top_k"]),
        "top_p": float(gen_cfg["top_p"]),
        "num_beams": int(gen_cfg["num_beams"]),
    }

    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id

    return kwargs
