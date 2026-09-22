"""Pure tensor helpers for the official ACL 2025 VHR implementation."""

from typing import Dict, List, Tuple

import torch


def configuration() -> Dict[str, object]:
    return {
        "augmentation_ratio": 2.0,
        "last_layers": 14,
        "include_layer_one": True,
        "outlier_filter": True,
    }


def target_layer_indices(
    total_layers: int, last_layers: int = 14, include_layer_one: bool = True
) -> List[int]:
    """Released LLaVA-1.5 layer policy: layer 1 plus the final 14 layers."""
    if total_layers < 1 or not 0 <= int(last_layers) <= total_layers:
        raise ValueError("invalid VHR layer configuration")
    layers = list(range(total_layers - int(last_layers), total_layers))
    if include_layer_one and total_layers > 1:
        layers = [1] + layers
    return list(dict.fromkeys(layers))


def select_vision_aware_heads(
    text_contrast_head_output: torch.Tensor,
    visual_head_output: torch.Tensor,
    apply_outlier_filter: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Released VHD, Eq. 6 filtering and Eq. 7 median head selection."""
    if text_contrast_head_output.shape != visual_head_output.shape:
        raise ValueError("VHR text/visual head-output shapes differ")
    if text_contrast_head_output.ndim != 2:
        raise ValueError("VHR expects [heads, head_dim] tensors")
    divergence = (
        (text_contrast_head_output.float() - visual_head_output.float()) ** 2
    ).sum(dim=-1)
    filtered = divergence.clone()
    outliers = torch.zeros_like(filtered, dtype=torch.bool)
    # Preserve the released implementation's absolute std>1 guard and default
    # unbiased torch.std behavior.
    if bool(apply_outlier_filter) and filtered.std() > 1:
        text_activation = (text_contrast_head_output.float() ** 2).sum(dim=-1)
        outliers = (
            (filtered > filtered.mean() + filtered.std())
            & (text_activation > text_activation.mean() + text_activation.std())
        )
        filtered[outliers] = 0
    selected = (filtered > filtered.median()).nonzero().flatten()
    event = {
        "num_heads": int(filtered.numel()),
        "selected_count": int(selected.numel()),
        "outlier_count": int(outliers.sum().item()),
        "raw_divergence_mean": float(divergence.mean().item()),
        "raw_divergence_std": float(divergence.std().item()),
        "filtered_divergence_median": float(filtered.median().item()),
        "filtered_divergence_max": float(filtered.max().item()),
    }
    return selected, event


def reinforce_heads(
    attention_output: torch.Tensor,
    selected_heads: torch.Tensor,
    augmentation_ratio: float = 2.0,
) -> torch.Tensor:
    """Scale selected per-head outputs before concatenation and o_proj."""
    if attention_output.ndim != 4:
        raise ValueError("VHR attention output must be [batch, heads, query, dim]")
    if float(augmentation_ratio) <= 0:
        raise ValueError("VHR augmentation ratio must be positive")
    result = attention_output.clone()
    result[:, selected_heads.to(result.device), :, :] *= float(augmentation_ratio)
    return result
