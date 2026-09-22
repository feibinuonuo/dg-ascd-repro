"""Frozen ClearSight/VAF (CVPR 2025) attention-logit transformation."""

from typing import Dict, Tuple

import torch


SOURCE_COMMIT = "5466c945be8cbd69ecc08b09455d8ed11f37ce67"


def configuration() -> Dict[str, object]:
    """Return the released LLaVA-1.5 inference defaults, without tuning."""
    return {
        "target_layers": [9, 10, 11, 12, 13, 14],
        "enhancement_multiplier": 1.15,
        "suppression_multiplier": 0.95,
        "application": "raw_attention_logits_before_causal_mask",
    }


def apply_vaf_logits(
    attn_weights: torch.Tensor,
    *,
    layer_index: int,
    sys_len: int,
    img_len: int,
    enhancement_multiplier: float = 1.15,
    suppression_multiplier: float = 0.95,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Apply released VAF in place and return a compact application event.

    The released code scales all query rows during cached decoding and only text
    rows after the image prefix during a full prompt forward. The key ranges are
    clipped solely to support variable prompt lengths; LLaVA-1.5 uses 35/576.
    """
    if attn_weights.ndim != 4:
        raise ValueError("VAF expects [batch, heads, query, key] attention logits")
    _, _, q_len, key_len = attn_weights.shape
    image_start = min(max(int(sys_len), 0), int(key_len))
    image_end = min(image_start + max(int(img_len), 0), int(key_len))
    query_start = image_end if int(q_len) > image_end else 0
    if image_end <= image_start:
        raise ValueError("VAF visual-token interval is empty")
    if query_start >= q_len:
        raise ValueError("VAF has no applicable query rows")
    attn_weights[:, :, query_start:, image_start:image_end].mul_(
        float(enhancement_multiplier)
    )
    if image_start:
        attn_weights[:, :, query_start:, :image_start].mul_(
            float(suppression_multiplier)
        )
    return attn_weights, {
        "layer_index": int(layer_index),
        "query_length": int(q_len),
        "key_length": int(key_len),
        "query_start": int(query_start),
        "image_start": int(image_start),
        "image_end": int(image_end),
        "applied_query_rows": int(q_len - query_start),
    }
