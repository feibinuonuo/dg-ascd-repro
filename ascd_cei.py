"""Frozen Static CEI helpers for the ACL 2026 method adaptation."""

from typing import Dict

import torch


def configuration() -> Dict[str, float]:
    """Return the pre-registered LLaVA-1.5 Static CEI configuration."""
    return {
        "context_layer": -1,
        "context_position": -1,
        "injection_layer": 10,
        "alpha": 0.1,
    }


def blend_last_hidden(
    hidden_states: torch.Tensor,
    context_embedding: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend the final sequence position with the fixed context embedding."""
    if hidden_states.ndim != 3 or context_embedding.ndim not in (1, 2):
        raise ValueError("CEI expects [batch, sequence, hidden] and [hidden]/[batch, hidden]")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("CEI alpha must be in [0, 1]")
    context = context_embedding
    if context.ndim == 1:
        context = context.unsqueeze(0)
    if context.shape != hidden_states[:, -1, :].shape:
        raise ValueError("CEI context and last hidden state shapes differ")
    result = hidden_states.clone()
    context = context.to(device=result.device, dtype=result.dtype)
    result[:, -1, :] = (
        (1.0 - float(alpha)) * result[:, -1, :] + float(alpha) * context
    )
    return result
