import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from chaos_llm.config import load_config, jsonable_cfg, resolve_device, resolve_dtype
from chaos_llm.generation import (
    generate_baseline,
    generate_baseline_topk,
    generate_with_perturbation,
    generate_with_perturbation_topk,
)
from chaos_llm.modeling import apply_attention_overrides, build_generation_kwargs, load_model_and_tokenizer
from chaos_llm.output import make_run_dir, save_config_snapshot, save_logits_npz, save_text_json, save_tokens_npz
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
    prompt_ids_cpu = input_ids[0].detach().cpu().numpy()

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

    include_prompt_tokens = bool(cfg["output"].get("include_prompt_tokens", True))
    save_text = bool(cfg["output"].get("save_text", False))
    text_filename = str(cfg["output"].get("text_filename", "texts.json"))
    text_skip_special = bool(cfg["output"].get("text_skip_special_tokens", True))
    text_clean_spaces = bool(cfg["output"].get("text_clean_up_spaces", True))
    prompt_len_saved = int(prompt_len if include_prompt_tokens else 0)
    logits_cfg = cfg.get("logits", {})
    logits_enabled = bool(logits_cfg.get("enabled", False))
    logits_top_k = int(logits_cfg.get("top_k", 10))
    logits_methods = [str(m) for m in logits_cfg.get("methods", [])]
    logits_max_steps = logits_cfg.get("max_steps", None)
    logits_filename = str(logits_cfg.get("filename", "logits_metrics.npz"))

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for sliding_window in cfg["attention"]["sliding_window_list"]:
        apply_attention_overrides(
            model,
            sliding_window=int(sliding_window),
            use_sliding_window=bool(cfg["attention"]["use_sliding_window"]),
        )

        baseline_topk_logits = None
        baseline_topk_indices = None
        if logits_enabled:
            baseline_generated, baseline_topk_logits, baseline_topk_indices = generate_baseline_topk(
                model,
                inputs_embeds=base_embeds,
                attention_mask=attention_mask,
                gen_kwargs=gen_kwargs,
                top_k=logits_top_k,
                max_steps=logits_max_steps,
            )
            baseline_ids_cpu = baseline_generated.detach().cpu().numpy()
        else:
            baseline_ids = generate_baseline(
                model,
                input_ids=None,
                inputs_embeds=base_embeds,
                attention_mask=attention_mask,
                gen_kwargs=gen_kwargs,
            )
            baseline_ids_cpu = baseline_ids[0].detach().cpu().numpy()
        if include_prompt_tokens:
            if baseline_ids_cpu.shape[0] < prompt_len or not np.array_equal(
                baseline_ids_cpu[:prompt_len], prompt_ids_cpu
            ):
                baseline_ids_cpu = np.concatenate([prompt_ids_cpu, baseline_ids_cpu])
        else:
            if baseline_ids_cpu.shape[0] >= prompt_len and np.array_equal(
                baseline_ids_cpu[:prompt_len], prompt_ids_cpu
            ):
                baseline_ids_cpu = baseline_ids_cpu[prompt_len:]
        baseline_text = None
        if save_text:
            baseline_text = tokenizer.decode(
                baseline_ids_cpu.tolist(),
                skip_special_tokens=text_skip_special,
                clean_up_tokenization_spaces=text_clean_spaces,
            )
        baseline_ids_device = None
        baseline_generated_only = None
        if adaptive_stop:
            if logits_enabled:
                baseline_generated_only = baseline_ids_cpu[prompt_len:] if include_prompt_tokens else baseline_ids_cpu
            else:
                baseline_ids_device = baseline_ids
        if not adaptive_stop and not logits_enabled:
            del baseline_ids
            cleanup()

        for magnitude in cfg["perturbation"]["magnitude_list"]:
            run_dir = make_run_dir(
                output_dir,
                int(sliding_window),
                float(magnitude),
                prompt["name"],
                int(cfg["project"]["seed"]),
            )

            perturbed_ids: List[np.ndarray] = []
            perturbed_texts: List[str] = []
            divergence_index: List[int] = []
            logit_metrics: Dict[str, List[np.ndarray]] = {m: [] for m in logits_methods}

            iterator = iter_simplex_perturbations(
                base_embeds=base_embeds,
                simplex=simplex,
                magnitude=float(magnitude),
                subspace_indices=subspace_indices,
            )

            # Determine batch size. Fallback to 1 if using adaptive_stop or logits (which require sequential logic)
            batch_size = int(cfg.get("generation", {}).get("batch_size", 64))
            if adaptive_stop or logits_enabled:
                batch_size = 1
                
            deltas = list(iterator)

            for i in tqdm(range(0, len(deltas), batch_size), desc=run_dir):
                batch_deltas = deltas[i:i+batch_size]
                delta_batch = torch.cat(batch_deltas, dim=0) # [B, seq_len, embed_dim]
                perturbed_embeds = base_embeds.expand(len(batch_deltas), -1, -1) + delta_batch
                batch_attention_mask = attention_mask.expand(len(batch_deltas), -1)

                if logits_enabled or adaptive_stop:
                    # Sequential fallback
                    for b in range(len(batch_deltas)):
                        if logits_enabled:
                            output_ids, div_idx, metrics = generate_with_perturbation_topk(
                                model,
                                inputs_embeds=perturbed_embeds[b:b+1],
                                attention_mask=attention_mask,
                                gen_kwargs=gen_kwargs,
                                baseline_topk_logits=baseline_topk_logits or [],
                                baseline_topk_indices=baseline_topk_indices or [],
                                methods=logits_methods,
                                prompt_len=prompt_len,
                                adaptive_stop=adaptive_stop,
                                baseline_ids=baseline_generated_only,
                                max_steps=logits_max_steps,
                            )
                            for name, values in metrics.items():
                                logit_metrics[name].append(np.array(values, dtype=np.float32))
                            seq = output_ids.detach().cpu().numpy()
                        else:
                            output_ids, div_idx = generate_with_perturbation(
                                model,
                                inputs_embeds=perturbed_embeds[b:b+1],
                                attention_mask=attention_mask,
                                gen_kwargs=gen_kwargs,
                                baseline_ids=baseline_ids_device,
                                prompt_len=prompt_len,
                                adaptive_stop=adaptive_stop,
                            )
                            seq = output_ids[0].detach().cpu().numpy()
                            
                        if include_prompt_tokens:
                            if seq.shape[0] < prompt_len or not np.array_equal(
                                seq[:prompt_len], prompt_ids_cpu
                            ):
                                seq = np.concatenate([prompt_ids_cpu, seq])
                        else:
                            if seq.shape[0] >= prompt_len and np.array_equal(
                                seq[:prompt_len], prompt_ids_cpu
                            ):
                                seq = seq[prompt_len:]
                        perturbed_ids.append(seq)
                        if save_text:
                            text = tokenizer.decode(
                                seq.tolist(),
                                skip_special_tokens=text_skip_special,
                                clean_up_tokenization_spaces=text_clean_spaces,
                            )
                            perturbed_texts.append(text)
                        divergence_index.append(div_idx)
                        del output_ids, seq
                    cleanup()
                else:
                    # Batched generation
                    output_ids, _ = generate_with_perturbation(
                        model,
                        inputs_embeds=perturbed_embeds,
                        attention_mask=batch_attention_mask,
                        gen_kwargs=gen_kwargs,
                        baseline_ids=None,
                        prompt_len=prompt_len,
                        adaptive_stop=False,
                    )
                    
                    for b in range(len(batch_deltas)):
                        seq = output_ids[b].detach().cpu().numpy()
                        if include_prompt_tokens:
                            if seq.shape[0] < prompt_len or not np.array_equal(
                                seq[:prompt_len], prompt_ids_cpu
                            ):
                                seq = np.concatenate([prompt_ids_cpu, seq])
                        else:
                            if seq.shape[0] >= prompt_len and np.array_equal(
                                seq[:prompt_len], prompt_ids_cpu
                            ):
                                seq = seq[prompt_len:]
                        perturbed_ids.append(seq)
                        if save_text:
                            text = tokenizer.decode(
                                seq.tolist(),
                                skip_special_tokens=text_skip_special,
                                clean_up_tokenization_spaces=text_clean_spaces,
                            )
                            perturbed_texts.append(text)
                        divergence_index.append(-1) # No divergence index tracked in batched mode
                    
                    del output_ids
                    cleanup()
                
                del batch_deltas, delta_batch, perturbed_embeds, batch_attention_mask
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
                prompt_len=int(prompt_len_saved),
            )
            if logits_enabled and logits_methods:
                save_logits_npz(
                    path=os.path.join(run_dir, logits_filename),
                    metrics=logit_metrics,
                )
            if save_text:
                save_text_json(
                    path=os.path.join(run_dir, text_filename),
                    baseline_text=baseline_text or "",
                    perturbed_texts=perturbed_texts,
                    prompt_text=prompt["text"],
                    include_prompt_tokens=include_prompt_tokens,
                )

            del perturbed_ids, divergence_index, perturbed_texts, logit_metrics
            cleanup()

        if baseline_ids_device is not None:
            del baseline_ids_device
        baseline_topk_logits = None
        baseline_topk_indices = None
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
