"""Default-off finite visual-support penalty for Calibrated Soft-Grounded ASCD."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from ascd_detector_grounded import terminal_decoded_chair_object


def apply_soft_grounded_object_penalty(
    scores: torch.Tensor,
    *,
    tokenizer,
    generated_token_ids: Sequence[int],
    support_scores: Dict[str, float],
    top_k: int,
    probability_slope: float,
    probability_intercept: float,
    probability_clip_min: float,
    probability_clip_max: float,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply a frozen finite one-sided penalty to lexical object candidates."""
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("Soft-Grounded ASCD currently requires batch_size=1")
    if int(top_k) < 1:
        raise ValueError("Soft-Grounded ASCD top_k must be positive")
    lower, upper = float(probability_clip_min), float(probability_clip_max)
    if not (0.0 < lower <= upper <= 0.5):
        raise ValueError("soft probability clip must satisfy 0 < min <= max <= 0.5")
    slope, intercept = float(probability_slope), float(probability_intercept)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("soft calibration coefficients must be finite")
    finite_ids = torch.nonzero(torch.isfinite(scores[0]), as_tuple=False).flatten()
    if finite_ids.numel() == 0:
        raise ValueError("ASCD score distribution has no finite candidate")
    k = min(int(top_k), int(finite_ids.numel()))
    top_values, top_ids = torch.topk(scores[0], k=k)
    original_top = int(top_ids[0].item())
    result = scores.clone()
    object_candidates: List[Dict[str, object]] = []
    penalized_ids: List[int] = []
    for rank, (value, token_id) in enumerate(zip(top_values.tolist(), top_ids.tolist())):
        token_id = int(token_id)
        object_name = terminal_decoded_chair_object(tokenizer, generated_token_ids, token_id)
        if object_name is None:
            continue
        support = support_scores.get(object_name)
        probability_value = None
        clipped_probability = None
        penalty = 0.0
        if support is not None:
            raw_logit = max(-60.0, min(60.0, slope * float(support) + intercept))
            probability_value = 1.0 / (1.0 + math.exp(-raw_logit))
            clipped_probability = min(upper, max(lower, probability_value))
            penalty = math.log(clipped_probability / (1.0 - clipped_probability))
            result[0, token_id] = result[0, token_id] + penalty
            penalized_ids.append(token_id)
        object_candidates.append({
            "rank": int(rank),
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "object": object_name,
            "detector_score": None if support is None else float(support),
            "calibrated_probability": probability_value,
            "clipped_probability": clipped_probability,
            "soft_penalty": float(penalty),
            "hard_masked": False,
            "post_processor_score": float(value),
        })
    if not torch.isfinite(result).any():
        raise RuntimeError("finite soft penalties unexpectedly removed all ASCD candidates")
    selected = int(torch.argmax(result, dim=-1)[0].item())
    event = {
        "step": len(generated_token_ids),
        "original_top_token_id": original_top,
        "original_top_token": tokenizer.convert_ids_to_tokens(original_top),
        "selected_token_id": selected,
        "selected_token": tokenizer.convert_ids_to_tokens(selected),
        "top_k": k,
        "object_candidates": object_candidates,
        "soft_penalized_token_ids": penalized_ids,
        "hard_masked_token_ids": [],
        "selection_changed": bool(selected != original_top),
        "protected_no_finite": False,
    }
    return result, event
