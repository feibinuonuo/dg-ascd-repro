"""Dependency-light ICCV 2025 ONLY helpers for the ASCD generation path."""

import math

import torch


def build_text_enhanced_attention(
    attention_probabilities,
    value_states,
    image_start,
    image_length,
):
    """Construct ONLY's entropy-selected one-layer attention output."""
    if attention_probabilities.ndim != 4 or value_states.ndim != 4:
        raise ValueError("ONLY attention inputs must have rank four")
    if attention_probabilities.shape[:2] != value_states.shape[:2]:
        raise ValueError("ONLY attention/value batch and head dimensions differ")
    if attention_probabilities.shape[-1] != value_states.shape[-2]:
        raise ValueError("ONLY key lengths differ")
    if attention_probabilities.shape[0] != 1:
        raise ValueError("ONLY integration currently requires batch_size=1")

    key_length = attention_probabilities.shape[-1]
    image_start = int(image_start)
    image_end = image_start + int(image_length)
    if not 1 <= image_start < image_end <= key_length:
        raise ValueError(
            f"invalid ONLY spans: text_start=1 image={image_start}:{image_end} key={key_length}"
        )

    enhanced = attention_probabilities.clone()
    text = torch.cat(
        (enhanced[..., 1:image_start], enhanced[..., image_end:]), dim=-1
    )
    if text.shape[-1] < 2 or image_end - image_start < 2:
        raise ValueError("ONLY requires at least two text and visual keys")

    text_threshold = text.mean(dim=-1, keepdim=True) + text.std(dim=-1, keepdim=True)
    text = torch.where(text > text_threshold, torch.zeros_like(text), text)
    prefix_length = image_start - 1
    enhanced[..., 1:image_start] = text[..., :prefix_length]
    enhanced[..., image_end:] = text[..., prefix_length:]
    text_normalized = text / text.sum(dim=-1, keepdim=True)
    text_normalized = text_normalized[:, :, -1:, :]

    visual = enhanced[..., image_start:image_end]
    visual_threshold = visual.mean(dim=-1, keepdim=True) + visual.std(dim=-1, keepdim=True)
    visual = torch.where(visual > visual_threshold, torch.zeros_like(visual), visual)
    enhanced[..., image_start:image_end] = visual
    visual_normalized = visual / visual.sum(dim=-1, keepdim=True)
    visual_normalized = visual_normalized[:, :, -1:, :]

    text_entropy = -(text_normalized * torch.log(text_normalized + 1e-6)).sum(dim=-1)
    visual_entropy = -(visual_normalized * torch.log(visual_normalized + 1e-6)).sum(dim=-1)
    text_entropy = torch.nan_to_num(text_entropy, nan=0.0)
    visual_entropy = torch.nan_to_num(visual_entropy, nan=float("inf"))
    ratio = (text_entropy.sum(dim=-1) / visual_entropy.sum(dim=-1)).squeeze(0)
    if torch.isnan(ratio).any():
        raise ValueError("ONLY produced NaN entropy ratios")
    removed_heads = ratio < ratio.mean()
    enhanced[:, removed_heads, :, :] = 0

    head_output = torch.matmul(enhanced, value_states)
    return head_output, {
        "entropy_ratios": ratio.detach(),
        "removed_heads": removed_heads.detach(),
    }


def build_text_enhanced_logits(
    text_enhanced_attention_output,
    hidden_states,
    final_decoder_layer,
    final_norm,
    lm_head,
):
    """Propagate ONLY's layer intervention through its released final branch."""
    if hidden_states is None or len(hidden_states) < 3:
        raise ValueError("ONLY requires all decoder hidden states")
    branch = text_enhanced_attention_output[:, -1:, :]
    final_layer_residual = hidden_states[-2][:, -1:, :]
    ordinary_final = hidden_states[-1][:, -1:, :]

    branch = final_decoder_layer.input_layernorm(branch)
    branch = 0.2 * final_layer_residual + branch
    branch_residual = branch
    branch = final_decoder_layer.post_attention_layernorm(branch)
    branch = final_decoder_layer.mlp(branch)
    branch = branch_residual + branch
    branch = final_norm(branch)
    branch = branch + 0.5 * ordinary_final
    return lm_head(branch)[:, -1, :]


def only_calibrate(
    parent_logits,
    text_enhanced_logits,
    positive_alpha=3.0,
    negative_alpha=1.0,
    beta=0.1,
    tvd_threshold=0.25,
):
    """Apply ONLY's TVD router and adaptive plausibility constraint."""
    if parent_logits.ndim != 2 or parent_logits.shape[0] != 1:
        raise ValueError("ONLY requires parent logits with shape [1, vocabulary]")
    if text_enhanced_logits.shape != parent_logits.shape:
        raise ValueError("ONLY parent and text-enhanced logits must have equal shape")
    if not 0 < float(beta) <= 1:
        raise ValueError("ONLY beta must be in (0, 1]")
    if float(tvd_threshold) < 0:
        raise ValueError("ONLY TVD threshold must be nonnegative")

    tvd = torch.sum(
        torch.abs(
            torch.softmax(parent_logits.float(), dim=-1)
            - torch.softmax(text_enhanced_logits.float(), dim=-1)
        )
    )
    if not torch.isfinite(tvd):
        raise ValueError("ONLY produced nonfinite TVD")
    if float(tvd.item()) < float(tvd_threshold):
        route = "collaboration"
        calibrated = parent_logits + float(positive_alpha) * text_enhanced_logits
    else:
        route = "contrast"
        calibrated = (
            (1.0 + float(negative_alpha)) * parent_logits
            - float(negative_alpha) * text_enhanced_logits
        )

    cutoff = math.log(float(beta)) + parent_logits.max(dim=-1, keepdim=True).values
    calibrated = calibrated.masked_fill(parent_logits < cutoff, -float("inf"))
    if not torch.isfinite(calibrated).any(dim=-1).all():
        raise ValueError("ONLY masked every candidate token")
    return calibrated, {
        "tvd": float(tvd.item()),
        "route": route,
        "parent_selected_token_id": int(torch.argmax(parent_logits, dim=-1)[0].item()),
        "text_enhanced_selected_token_id": int(
            torch.argmax(text_enhanced_logits, dim=-1)[0].item()
        ),
        "only_selected_token_id": int(torch.argmax(calibrated, dim=-1)[0].item()),
        "candidate_count": int(torch.isfinite(calibrated[0]).sum().item()),
    }
