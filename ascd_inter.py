"""Frozen ICCV 2025 INTER interaction-guidance primitives."""

import math

import torch


SOURCE_COMMIT = "586f565aae19fa2c0b4fb01b134547e15e3424bf"


def inter_calibrate(original_logits, random_image_logits, empty_text_logits,
                    random_image_empty_text_logits, variance_threshold=1.0,
                    beta=0.1, parent_logits=None):
    """Apply the released four-coalition interaction and plausibility mask."""
    interaction = (
        original_logits
        - random_image_logits
        - empty_text_logits
        + random_image_empty_text_logits
    )
    variance = torch.var(interaction, dim=1)
    active = variance >= float(variance_threshold)
    if parent_logits is None:
        parent_logits = original_logits
    guided = parent_logits + interaction * active.unsqueeze(-1).to(interaction.dtype)
    cutoff = math.log(float(beta)) + original_logits.max(dim=-1, keepdim=True).values
    candidate_mask = original_logits >= cutoff
    guided = guided.masked_fill(~candidate_mask, -float("inf"))
    return guided, {
        "interaction_variance": float(variance[0].item()),
        "interaction_active": bool(active[0].item()),
        "candidate_count": int(candidate_mask.sum().item()),
        "original_top_token_id": int(torch.argmax(original_logits, dim=-1)[0].item()),
        "parent_top_token_id": int(torch.argmax(parent_logits, dim=-1)[0].item()),
        "inter_top_token_id": int(torch.argmax(guided, dim=-1)[0].item()),
    }


def configuration():
    return {
        "source_commit": SOURCE_COMMIT,
        "variance_threshold": 1.0,
        "beta": 0.1,
        "interaction": "full-random_image-empty_text+random_image_empty_text",
        "random_image": "deterministic uniform processed tensor",
        "decoding": "greedy LLaVA adaptation of the released sampling formula",
    }
