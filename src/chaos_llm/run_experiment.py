import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from chaos_llm.config import load_config, jsonable_cfg, resolve_device, resolve_dtype
from chaos_llm.generation import generate_baseline, generate_with_perturbation
from chaos_llm.modeling import apply_attention_overrides, build_generation_kwargs, load_model_and_tokenizer
from chaos_llm.output import make_run_dir, save_config_snapshot, save_tokens_npz
from chaos_llm.perturbations import build_simplex, iter_simplex_perturbations, select_subspace_indices
from chaos_llm.prompt_loader import load_prompts
from chaos_llm.utils import cleanup, set_seed


def build_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids)


def run_prompt(
    model: Any,
    tokenizer: Any,
    prompt: Dict[str, str],
    cfg: Dict[str, Any],
    device: str,
    dtype: torch.dtype,
) -> None:
    input_ids = tokenizer(prompt["text"], return_tensors="pt").input_ids.to(device)
    attention_mask = build_attention_mask(input_ids)

    with torch.no_grad():
        base_embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype)
    prompt_len = input_ids.shape[1]
    embed_dim = base_embeds.shape[2]
    total_dim = prompt_len * embed_dim

    token_limit = cfg["perturbation"]["apply_to_first_n_tokens"]
    if token_limit is not None:
        token_limit = min(int(token_limit), int(prompt_len))
        allowed_dim = token_limit * embed_dim
    else:
        allowed_dim = total_dim

    num_conditions = int(cfg["perturbation"]["num_conditions"])
    simplex = build_simplex(num_conditions)

    rng = np.random.default_rng(int(cfg["perturbation"]["subspace_seed"]))
    subspace_indices = select_subspace_indices(
        total_dim=allowed_dim,
        num_points=num_conditions,
        mode=str(cfg["perturbation"]["subspace_mode"]),
        rng=rng,
    )

    gen_kwargs = build_generation_kwargs(cfg, tokenizer)
    adaptive_stop = bool(cfg["generation"]["adaptive_stop"])

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for sliding_window in cfg["attention"]["sliding_window_list"]:
        apply_attention_overrides(
            model,
            sliding_window=int(sliding_window),
            use_sliding_window=bool(cfg["attention"]["use_sliding_window"]),
        )

        baseline_ids = generate_baseline(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_kwargs=gen_kwargs,
        )
        baseline_ids_cpu = baseline_ids[0].detach().cpu().numpy()
        baseline_ids_device = baseline_ids if adaptive_stop else None
        if not adaptive_stop:
            del baseline_ids
            cleanup()

        for magnitude in cfg["perturbation"]["magnitude_list"]:
            run_dir = make_run_dir(output_dir, int(sliding_window), float(magnitude), prompt["name"])

            perturbed_ids: List[np.ndarray] = []
            divergence_index: List[int] = []

            iterator = iter_simplex_perturbations(
                base_embeds=base_embeds,
                simplex=simplex,
                magnitude=float(magnitude),
                subspace_indices=subspace_indices,
            )

            for delta in tqdm(iterator, total=num_conditions, desc=run_dir):
                perturbed_embeds = base_embeds + delta
                output_ids, div_idx = generate_with_perturbation(
                    model,
                    inputs_embeds=perturbed_embeds,
                    attention_mask=attention_mask,
                    gen_kwargs=gen_kwargs,
                    baseline_ids=baseline_ids_device,
                    prompt_len=prompt_len,
                    adaptive_stop=adaptive_stop,
                )

                perturbed_ids.append(output_ids[0].detach().cpu().numpy())
                divergence_index.append(div_idx)

                del delta, perturbed_embeds, output_ids
                cleanup()

            config_snapshot = jsonable_cfg(cfg)
            config_snapshot["prompt"] = prompt
            config_snapshot["runtime"] = {
                "sliding_window": int(sliding_window),
                "perturbation_magnitude": float(magnitude),
                "prompt_len": int(prompt_len),
            }

            save_config_snapshot(os.path.join(run_dir, "config.json"), config_snapshot)
            save_tokens_npz(
                path=os.path.join(run_dir, "tokens.npz"),
                baseline_ids=baseline_ids_cpu,
                perturbed_ids=perturbed_ids,
                divergence_index=divergence_index,
                pad_token_id=int(cfg["output"]["pad_token_id"]),
                prompt_len=int(prompt_len),
            )

            del perturbed_ids, divergence_index
            cleanup()

        if baseline_ids_device is not None:
            del baseline_ids_device
        cleanup()

    del base_embeds, input_ids, attention_mask
    cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chaotic LLM experiment")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--prompt-name", default=None, help="Run only a single prompt by name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    device = resolve_device(str(cfg["project"]["device"]))
    dtype = resolve_dtype(str(cfg["project"]["dtype"]), device=device)

    model, tokenizer = load_model_and_tokenizer(
        model_path=cfg["paths"]["model_path"],
        dtype=dtype,
        device=device,
        trust_remote_code=True,
    )

    prompts = load_prompts(cfg["paths"]["prompts_path"], encoding=cfg["prompts"]["encoding"])
    if args.prompt_name:
        prompts = [p for p in prompts if p["name"] == args.prompt_name]
        if not prompts:
            raise ValueError("No prompt matches --prompt-name")

    for prompt in prompts:
        run_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )


if __name__ == "__main__":
    main()
