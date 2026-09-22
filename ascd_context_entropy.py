"""Shared, dependency-light helpers for CEC-inspired candidate scoring."""

import math

import torch


def contextual_entropy_from_attention(
    attention,
    image_start,
    image_length,
    instruction_end,
    candidate_position,
):
    """Compute CEC contextual entropy from one final-layer attention tensor.

    ``attention`` is [batch, heads, query, key].  Component scores follow the
    paper: mean attention value within image, instruction and generated-history
    spans; normalize into look-back rates per head; apply a three-way softmax;
    average across heads and compute Shannon entropy.
    """
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError("attention must have shape [1, heads, query, key]")
    key_length = attention.shape[-1]
    image_start = int(image_start)
    image_end = image_start + int(image_length)
    instruction_end = int(instruction_end)
    candidate_position = int(candidate_position)
    if not (0 <= image_start < image_end <= instruction_end <= candidate_position < key_length):
        raise ValueError(
            "invalid context boundaries: "
            f"image={image_start}:{image_end} instruction_end={instruction_end} "
            f"candidate={candidate_position} key_length={key_length}"
        )
    last_query = attention[0, :, -1, :].detach().float()
    visual = last_query[:, image_start:image_end].mean(dim=-1)
    instruction_parts = []
    if image_start:
        instruction_parts.append(last_query[:, :image_start])
    if instruction_end > image_end:
        instruction_parts.append(last_query[:, image_end:instruction_end])
    if not instruction_parts:
        raise ValueError("instruction span is empty")
    instruction = torch.cat(instruction_parts, dim=-1).mean(dim=-1)
    if candidate_position > instruction_end:
        history = last_query[:, instruction_end:candidate_position].mean(dim=-1)
    else:
        history = torch.zeros_like(visual)
    component_means = torch.stack((visual, instruction, history), dim=-1)
    return contextual_entropy_from_component_means(component_means)


def contextual_entropy_from_component_means(component_means):
    """Compute contextual entropy from compact per-head [V, I, H] means."""
    component_means = torch.as_tensor(component_means).detach().float()
    if component_means.ndim != 2 or component_means.shape[-1] != 3:
        raise ValueError("component_means must have shape [heads, 3]")
    if (component_means < 0).any():
        raise ValueError("component attention means must be nonnegative")
    if not torch.isfinite(component_means).all():
        raise ValueError("nonfinite component attention")
    lookback_rates = component_means / component_means.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    per_head_distribution = torch.softmax(lookback_rates, dim=-1)
    distribution = per_head_distribution.mean(dim=0)
    entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum()
    if not torch.isfinite(entropy):
        raise ValueError("nonfinite contextual entropy")
    value = float(entropy.item())
    if value < -1e-7 or value > math.log(3.0) + 1e-6:
        raise ValueError(f"contextual entropy outside [0, log(3)]: {value}")
    return value, {
        "visual": float(distribution[0].item()),
        "instruction": float(distribution[1].item()),
        "history": float(distribution[2].item()),
    }


def calibrate_top_candidates(scores, candidate_ids, entropies, beta):
    """Add beta * entropy only to the explicitly supplied candidate ids."""
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("scores must have shape [1, vocabulary]")
    if len(candidate_ids) < 2 or len(candidate_ids) != len(entropies):
        raise ValueError("candidate ids/entropies must have equal length >= 2")
    ids = [int(candidate_id) for candidate_id in candidate_ids]
    if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= scores.shape[-1]:
        raise ValueError("candidate ids must be unique valid vocabulary ids")
    entropy_tensor = torch.as_tensor(entropies, dtype=scores.dtype, device=scores.device)
    if not torch.isfinite(entropy_tensor).all():
        raise ValueError("candidate entropies must be finite")
    calibrated = scores.clone()
    calibrated[0, torch.as_tensor(ids, device=scores.device)] += float(beta) * entropy_tensor
    return calibrated
