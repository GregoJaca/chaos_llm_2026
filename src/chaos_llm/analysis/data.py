import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


_RUN_RE = re.compile(r"^run_(?P<window>[^_]+)_(?P<mag>[^_]+)_(?P<prompt>.+)$")


def discover_runs(input_dir: str, run_list: Optional[List[str]]) -> List[str]:
    if run_list:
        resolved = []
        for entry in run_list:
            if os.path.isabs(entry):
                resolved.append(entry)
            else:
                resolved.append(os.path.join(input_dir, entry))
        return [p for p in resolved if os.path.isdir(p)]

    runs = []
    if not os.path.isdir(input_dir):
        return runs
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isdir(path) and name.startswith("run_"):
            runs.append(path)
    runs.sort()
    return runs


def load_run_metadata(run_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    name = os.path.basename(run_dir)
    match = _RUN_RE.match(name)
    meta = {"runtime": {}, "prompt": {"name": "unknown"}}
    if match:
        meta["runtime"]["sliding_window"] = match.group("window")
        meta["runtime"]["perturbation_magnitude"] = match.group("mag")
        meta["prompt"]["name"] = match.group("prompt")
    return meta


def load_tokens(run_dir: str, mmap_mode: Optional[str]) -> Dict[str, np.ndarray]:
    path = os.path.join(run_dir, "tokens.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing tokens.npz in {run_dir}")
    with np.load(path, mmap_mode=mmap_mode, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
