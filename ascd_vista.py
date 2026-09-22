"""Frozen VISTA (ICML 2025) VSV and SLA formulas."""

import torch
import torch.nn.functional as F


def configuration():
    return {"vsv_lambda": 0.17, "sla_start_layer": 25,
            "sla_end_layer": 30, "sla_alpha": 0.3}


def pca_direction(negative: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    if negative.ndim != 2 or positive.shape != negative.shape:
        raise ValueError("VISTA expects matching [layers, hidden] activation matrices")
    difference = (positive.float() - negative.float()).reshape(1, -1)
    mean = difference.mean(0, keepdim=True)
    centered = difference - mean
    u, _, vh = torch.linalg.svd(centered, full_matrices=False)
    max_abs_columns = torch.argmax(torch.abs(u), dim=0)
    indices = torch.arange(u.shape[1], device=u.device)
    signs = torch.sign(u[max_abs_columns, indices])
    components = (vh * signs.view(-1, 1))[:1]
    direction = (components.sum(dim=0, keepdim=True) + mean).mean(dim=0)
    return direction.reshape_as(negative)


def vsv_transform(hidden: torch.Tensor, direction: torch.Tensor, strength: float):
    if hidden.ndim != 3 or direction.ndim != 1 or hidden.shape[-1] != direction.shape[0]:
        raise ValueError("VISTA VSV requires [batch, tokens, hidden] and [hidden]")
    value = hidden.float()
    original_norm = torch.norm(value, p=2, dim=-1, keepdim=True)
    unit_direction = F.normalize(direction.float(), dim=-1)
    schedule = 1.0 + torch.clamp(
        F.cosine_similarity(value, -direction.float()[None, None, :], dim=-1),
        min=0.0,
    ).unsqueeze(-1)
    steering = float(strength) * schedule * unit_direction[None, None, :]
    transformed = F.normalize(F.normalize(value, p=2, dim=-1) + steering,
                              p=2, dim=-1) * original_norm
    return transformed.to(dtype=hidden.dtype), {
        "schedule_min": float(schedule.min().item()),
        "schedule_max": float(schedule.max().item()),
        "input_norm_min": float(original_norm.min().item()),
        "input_norm_max": float(original_norm.max().item()),
    }


def sla_mix(final_logits, early_logits, alpha):
    if not early_logits:
        raise ValueError("VISTA SLA requires at least one early-layer logit tensor")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("VISTA SLA alpha must be in [0, 1]")
    early = torch.stack([value.float() for value in early_logits], dim=0).mean(dim=0)
    return float(alpha) * early + (1.0 - float(alpha)) * final_logits.float()
