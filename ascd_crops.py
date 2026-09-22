"""Frozen CRoPS logit calibration helpers."""

import math
from typing import Dict, Tuple

import torch


def configuration() -> Dict[str, float]:
    return {
        "lambda_lang_prior": 0.01,
        "alpha_stat_bias": 1.0,
        "beta_cutoff": 0.1,
        "max_threshold_plausibility_constraint": 0.95,
        "aggregate_layer": 2,
        "visual_keep_fraction": 0.25,
        "text_minimum_b0": 10.0,
        "text_minimum_b1": 30.0,
        "text_minimum_lambda": 0.001,
    }


def minimum_text_tokens(step: int, b0: float = 10.0, b1: float = 30.0,
                        decay: float = 0.001) -> int:
    """Released time-dependent text-mask size (one-based decoding step)."""
    return math.floor(b0 + b1 * (1.0 - math.exp(-decay * int(step))))


def calibrate(
    direct_logits: torch.Tensor,
    language_prior_logits: torch.Tensor,
    statistical_bias_logits: torch.Tensor,
    *,
    step: int,
    lambda_lang_prior: float = 0.01,
    alpha_stat_bias: float = 1.0,
    beta_cutoff: float = 0.1,
    max_threshold_plausibility_constraint: float = 0.95,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply the released CRoPS generalized contrastive-decoding equations."""
    if not (direct_logits.shape == language_prior_logits.shape == statistical_bias_logits.shape):
        raise ValueError("CRoPS branches must have identical logit shapes")
    if step < 1:
        raise ValueError("CRoPS uses one-based decoding steps")
    if not 0.0 < beta_cutoff <= 1.0:
        raise ValueError("beta_cutoff must be in (0, 1]")
    if not 0.0 < max_threshold_plausibility_constraint <= 1.0:
        raise ValueError("plausibility threshold must be in (0, 1]")
    if lambda_lang_prior <= 0.0:
        raise ValueError("lambda_lang_prior must be positive")

    cutoff = math.log(float(beta_cutoff)) + direct_logits.max(dim=-1, keepdim=True).values
    plausible = direct_logits >= cutoff
    truncated = direct_logits.masked_fill(~plausible, -float("inf"))
    direct_probs = torch.softmax(truncated, dim=-1)
    max_probability = direct_probs.max(dim=-1, keepdim=True).values
    bypass = max_probability > float(max_threshold_plausibility_constraint)

    log_direct = torch.log_softmax(truncated, dim=-1)
    log_language = torch.log_softmax(language_prior_logits, dim=-1)
    log_statistical = torch.log_softmax(statistical_bias_logits, dim=-1)
    gamma = math.exp(-float(lambda_lang_prior) * int(step))
    language_weight = (1.0 - gamma) / gamma
    corrected = log_direct + language_weight * (log_direct - log_language)
    corrected = (1.0 + float(alpha_stat_bias)) * corrected - float(alpha_stat_bias) * log_statistical
    final = torch.where(bypass, truncated, corrected)

    direct_top = int(torch.argmax(truncated, dim=-1)[0].item())
    final_top = int(torch.argmax(final, dim=-1)[0].item())
    event = {
        "step": int(step),
        "gamma_lang_prior": float(gamma),
        "language_weight": float(language_weight),
        "max_direct_probability": float(max_probability[0, 0].item()),
        "plausibility_bypass": bool(bypass[0, 0].item()),
        "plausible_token_count": int(plausible[0].sum().item()),
        "direct_top_token_id": direct_top,
        "selected_top_token_id": final_top,
        "top_token_changed": direct_top != final_top,
    }
    return final, event
