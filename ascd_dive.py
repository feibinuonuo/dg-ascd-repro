"""Pure tensor helpers for a frozen, paper-faithful DiVE adaptation."""

import math
from typing import Dict, List, Tuple

import torch


def configuration() -> Dict[str, float]:
    return {
        "layer_exclusion_ratio": 0.05,
        "visual_suppression_strength": 0.5,
        "chair_confidence_threshold": 0.85,
        "epsilon": 1e-6,
    }


def candidate_layer_indices(total_layers: int, exclusion_ratio: float = 0.05) -> List[int]:
    """Zero-based form of paper Eq. 1, excluding ceil(kappa L) at both ends."""
    if total_layers < 1 or not 0.0 <= exclusion_ratio < 0.5:
        raise ValueError("invalid DiVE layer-pool arguments")
    excluded = math.ceil(float(exclusion_ratio) * int(total_layers))
    return list(range(excluded, total_layers - excluded))


def vlac_score(visual_attention: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Paper Eq. 2, averaged across heads for one or more layers.

    The final axis contains visual-token attention; the penultimate axis is heads.
    Leading dimensions, if any, are preserved.
    """
    if visual_attention.ndim < 2 or visual_attention.shape[-1] < 2:
        raise ValueError("DiVE V-LAC requires head and at least two visual-token axes")
    weights = visual_attention.float().clamp_min(0.0)
    probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))
    entropy = -(probs * probs.clamp_min(float(epsilon)).log()).sum(dim=-1)
    concentration = 1.0 - entropy / math.log(int(weights.shape[-1]))
    return concentration.mean(dim=-1)


def select_visual_evidence_layers(scores: torch.Tensor, candidate_indices: List[int]) -> List[int]:
    """Paper Eq. 3: retain candidate layers with above-candidate-mean V-LAC."""
    if scores.ndim != 1 or not candidate_indices:
        raise ValueError("DiVE needs one score per layer and a nonempty candidate pool")
    candidate = scores[torch.tensor(candidate_indices, device=scores.device)]
    selected = [index for index in candidate_indices if scores[index] > candidate.mean()]
    if not selected:
        # Exact ties are a numerical corner case not addressed by the paper.
        selected = [candidate_indices[int(torch.argmax(candidate).item())]]
    return selected


def head_visual_evidence(
    attention: torch.Tensor,
    value_states: torch.Tensor,
    visual_start: int,
    visual_length: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Paper Eqs. 4--7 before the output projection.

    Inputs use `[batch, heads, query, key]` and `[batch, heads, key, dim]`.
    """
    if attention.ndim != 4 or value_states.ndim != 4:
        raise ValueError("DiVE attention/value states must be rank four")
    if attention.shape[:2] != value_states.shape[:2] or attention.shape[-1] != value_states.shape[-2]:
        raise ValueError("DiVE attention/value shapes are incompatible")
    start, end = int(visual_start), int(visual_start + visual_length)
    if start < 0 or end > attention.shape[-1] or start >= end:
        raise ValueError("invalid DiVE visual-token interval")
    visual_attn = attention[..., start:end]
    visual_values = value_states[:, :, start:end, :]
    visual_output = torch.matmul(visual_attn, visual_values)

    text_mask = torch.ones(attention.shape[-1], device=attention.device, dtype=torch.bool)
    text_mask[start:end] = False
    text_attn = attention[..., text_mask]
    text_values = value_states[:, :, text_mask, :]
    text_output = torch.matmul(text_attn, text_values)
    text_mass = text_attn.sum(dim=-1, keepdim=True)
    text_only = text_output / (text_mass + float(epsilon))
    full_output = visual_output + text_output
    return full_output - text_only


def suppress_visual_evidence(
    hidden_state: torch.Tensor,
    visual_evidence: torch.Tensor,
    gamma: float = 0.5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Paper Eq. 10 with norm-matched evidence suppression."""
    if hidden_state.shape != visual_evidence.shape:
        raise ValueError("DiVE hidden state and evidence shapes differ")
    hidden_norm = hidden_state.float().norm(dim=-1, keepdim=True)
    evidence_norm = visual_evidence.float().norm(dim=-1, keepdim=True)
    scale = float(gamma) * hidden_norm / (evidence_norm + float(epsilon))
    return hidden_state - scale.to(hidden_state.dtype) * visual_evidence


def likelihood_ratio_calibration(
    original_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    threshold: float = 0.85,
    epsilon: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Paper Eqs. 11--14, preserving candidate-set probability mass."""
    if original_logits.shape != reference_logits.shape:
        raise ValueError("DiVE original/reference logit shapes differ")
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("DiVE confidence threshold must be in (0, 1]")
    original = torch.softmax(original_logits.float(), dim=-1)
    reference = torch.softmax(reference_logits.float(), dim=-1)
    candidate = original > float(threshold) * original.max(dim=-1, keepdim=True).values
    ratio = original / reference.clamp_min(float(epsilon))
    original_mass = (original * candidate).sum(dim=-1, keepdim=True)
    ratio_mass = (ratio * candidate).sum(dim=-1, keepdim=True).clamp_min(float(epsilon))
    normalized_ratio = ratio * (original_mass / ratio_mass)
    calibrated = torch.where(candidate, normalized_ratio, original)
    logits = calibrated.clamp_min(float(epsilon)).log().to(original_logits.dtype)
    event = {
        "candidate_count": int(candidate[0].sum().item()),
        "candidate_mass_before": float(original_mass[0, 0].item()),
        "candidate_mass_after": float((calibrated * candidate).sum(dim=-1)[0].item()),
        "original_top_token_id": int(torch.argmax(original_logits, dim=-1)[0].item()),
        "calibrated_top_token_id": int(torch.argmax(logits, dim=-1)[0].item()),
    }
    return logits, event
