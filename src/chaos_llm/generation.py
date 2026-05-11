from typing import Any, Dict, Optional, Tuple

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


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
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_kwargs: Dict[str, Any],
) -> torch.Tensor:
    with torch.no_grad():
        baseline_ids = model.generate(
            input_ids=input_ids,
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
