import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model_path(model_path: str) -> Tuple[str, Optional[str]]:
    path = Path(model_path)
    
    # 1. If it's a file, return it directly
    if path.is_file():
        return str(path), None
        
    # 2. If it's a directory containing config.json directly, load it directly
    if path.is_dir() and (path / "config.json").exists():
        return str(path), None
        
    # 3. If it's a HF cache folder itself (e.g. models--microsoft--Phi-4-mini-instruct)
    if path.is_dir() and path.name.startswith("models--"):
        parts = path.name.split("--")
        if len(parts) >= 2:
            repo_id = f"{parts[1]}/{'--'.join(parts[2:])}" if len(parts) >= 3 else parts[1]
            return repo_id, str(path.parent.resolve())
            
    # 4. If they wrote a relative path targeting a repo ID under a cache folder (e.g. "../microsoft/Phi-4-mini-instruct")
    parts = path.parts
    if len(parts) >= 2:
        namespace, model_name = parts[-2], parts[-1]
        cache_folder_name = f"models--{namespace}--{model_name}"
        prefix = Path(*parts[:-2])
        cache_dir_path = prefix / cache_folder_name
        if cache_dir_path.is_dir():
            return f"{namespace}/{model_name}", str(prefix.resolve())
            
    if len(parts) >= 1:
        model_name = parts[-1]
        cache_folder_name = f"models--{model_name}"
        prefix = Path(*parts[:-1])
        cache_dir_path = prefix / cache_folder_name
        if cache_dir_path.is_dir():
            return model_name, str(prefix.resolve())
            
    # 5. Otherwise, check standard snapshot directories or return the path as-is
    if path.is_dir():
        snapshots = path / "snapshots"
        if snapshots.exists() and snapshots.is_dir():
            candidates = []
            for child in snapshots.iterdir():
                if (child / "config.json").exists():
                    candidates.append(child)
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                return str(latest), None
                
    return model_path, None


def load_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    resolved_path, cache_dir = _resolve_model_path(model_path)
    
    kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
        kwargs["local_files_only"] = True
        
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs,
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
