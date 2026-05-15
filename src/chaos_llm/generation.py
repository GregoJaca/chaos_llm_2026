from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


class BaselineDivergenceCriteria(StoppingCriteria):
    def __init__(self, baseline_ids: torch.Tensor, prompt_len: int) -> None:
        self.baseline_ids = baseline_ids
        self.prompt_len = prompt_len
        self.divergence_index = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        seq_len = input_ids.shape[1]
        gen_pos = seq_len - self.prompt_len - 1
        if gen_pos < 0:
            return False

        baseline_total = self.baseline_ids.shape[1]
        if seq_len > baseline_total:
            self.divergence_index = gen_pos
            return True

        last_token = input_ids[0, -1]
        base_token = self.baseline_ids[0, self.prompt_len + gen_pos]
        if last_token != base_token:
            self.divergence_index = gen_pos
            return True

        return False


def generate_baseline(
    model: Any,
    input_ids: Optional[torch.Tensor],
    inputs_embeds: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> torch.Tensor:
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    with torch.no_grad():
        baseline_ids = model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    return baseline_ids


def generate_with_perturbation(
    model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_kwargs: Dict[str, Any],
    baseline_ids: Optional[torch.Tensor],
    prompt_len: int,
    adaptive_stop: bool,
) -> Tuple[torch.Tensor, int]:
    stopping = None
    criterion = None
    if adaptive_stop:
        if baseline_ids is None:
            raise ValueError("baseline_ids must be provided when adaptive_stop is True")
        criterion = BaselineDivergenceCriteria(baseline_ids=baseline_ids, prompt_len=prompt_len)
        stopping = StoppingCriteriaList([criterion])

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            stopping_criteria=stopping,
            **gen_kwargs,
        )

    divergence_index = -1
    if criterion is not None and criterion.divergence_index is not None:
        divergence_index = int(criterion.divergence_index)

    return output_ids, divergence_index


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    # logits shape: [B, K]
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=-1, keepdim=True)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # p, q shape: [B, K]
    eps = 1e-12
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return torch.sum(p * torch.log(p / q), dim=-1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # p, q shape: [B, K]
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [K] or [B, K], b: [B, K]
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return F.cosine_similarity(a, b, dim=-1)


def generate_baseline_topk(
    model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_kwargs: Dict[str, Any],
    top_k: int,
    max_steps: Optional[int],
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    max_new_tokens = int(gen_kwargs["max_new_tokens"])
    if max_steps is not None:
        max_new_tokens = min(max_new_tokens, int(max_steps))

    eos_token_id = gen_kwargs.get("eos_token_id")

    past = None
    next_input_ids = None
    generated: List[int] = []
    topk_logits: List[torch.Tensor] = []
    topk_indices: List[torch.Tensor] = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = model(
                    input_ids=next_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
            topk_logits.append(values[0].detach().cpu())
            topk_indices.append(indices[0].detach().cpu())

            next_token = torch.argmax(logits, dim=-1)
            generated.append(int(next_token.item()))

            if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                break

            next_input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_input_ids)], dim=1)
            
            # Explicit cleanup to free GPU memory
            del outputs, logits, values, indices
            if step % 50 == 0:
                torch.cuda.empty_cache()

    return torch.tensor(generated, device=inputs_embeds.device), topk_logits, topk_indices


def generate_with_perturbation_topk(
    model: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_kwargs: Dict[str, Any],
    baseline_topk_logits: List[torch.Tensor],
    baseline_topk_indices: List[torch.Tensor],
    methods: List[str],
    prompt_len: int,
    adaptive_stop: bool,
    baseline_ids: Optional[np.ndarray],
    max_steps: Optional[int],
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, List[np.ndarray]]]:
    batch_size = inputs_embeds.shape[0]
    max_new_tokens = int(gen_kwargs["max_new_tokens"])
    if max_steps is not None:
        max_new_tokens = min(max_new_tokens, int(max_steps))

    eos_token_id = gen_kwargs.get("eos_token_id")

    past = None
    next_input_ids = None
    generated: List[torch.Tensor] = []
    # metrics[method] will be a list of arrays, each array is [batch_size]
    metrics: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    divergence_index = np.full((batch_size,), -1, dtype=np.int32)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=inputs_embeds.device)

    for step in range(max_new_tokens):
        with torch.no_grad():
            if step == 0:
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = model(
                    input_ids=next_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :] # [B, V]

            if step < len(baseline_topk_logits):
                base_logits = baseline_topk_logits[step].to(logits.device) # [K]
                base_indices = baseline_topk_indices[step].to(logits.device) # [K]
                
                # Extract perturbed logits at the same indices as baseline top-k
                pert_logits = logits[:, base_indices] # [B, K]
                
                p = _safe_softmax(base_logits.unsqueeze(0)) # [1, K]
                q = _safe_softmax(pert_logits) # [B, K]

                if "kl_divergence" in methods:
                    metrics["kl_divergence"].append(_kl_divergence(p, q).detach().cpu().numpy())
                if "js_divergence" in methods:
                    metrics["js_divergence"].append(_js_divergence(p, q).detach().cpu().numpy())
                if "cos_sim" in methods or "cos_dist" in methods:
                    cos_sim = _cosine_sim(base_logits, pert_logits) # [B]
                    if "cos_sim" in methods:
                        metrics["cos_sim"].append(cos_sim.detach().cpu().numpy())
                    if "cos_dist" in methods:
                        metrics["cos_dist"].append((1.0 - cos_sim).detach().cpu().numpy())

            next_token = torch.argmax(logits, dim=-1) # [B]
            generated.append(next_token.unsqueeze(1))

            if adaptive_stop and baseline_ids is not None:
                if step < len(baseline_ids):
                    # Check which sequences just diverged
                    diverged_now = (next_token != int(baseline_ids[step])) & (~finished)
                    divergence_index[diverged_now.cpu().numpy()] = step
                    finished |= diverged_now

            if eos_token_id is not None:
                finished |= (next_token == int(eos_token_id))
            
            if finished.all():
                break

            next_input_ids = next_token.unsqueeze(1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_input_ids)], dim=1)

            # Explicit cleanup
            del outputs, logits
            if step % 50 == 0:
                torch.cuda.empty_cache()

    # stack generated tokens into [B, seq_len]
    all_generated = torch.cat(generated, dim=1)
    
    return all_generated, divergence_index, metrics
