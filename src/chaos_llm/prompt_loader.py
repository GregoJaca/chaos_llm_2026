from typing import Dict, List


def load_prompts(path: str, encoding: str = "utf-8") -> List[Dict[str, str]]:
    prompts = []
    with open(path, "r", encoding=encoding) as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                name, text = line.split("\t", 1)
                name = name.strip()
                text = text.strip()
            else:
                name = f"prompt_{idx}"
                text = line
            prompts.append({"name": name, "text": text})
    if not prompts:
        raise ValueError("No prompts found in prompts file")
    return prompts
