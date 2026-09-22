"""Dependency-light AAAI 2025 MoLE greedy decoding helpers."""

import torch
import torch.nn.functional as F


def _official_js_divergence(mature_logits, premature_logits):
    """Match the released MoLE ``calculate_jsd`` implementation exactly."""
    mature_probabilities = F.softmax(mature_logits, dim=-1)
    premature_probabilities = F.softmax(premature_logits, dim=-1)
    mixture = 0.5 * (
        mature_probabilities.unsqueeze(0) + premature_probabilities
    )
    mature_log_probabilities = F.log_softmax(mature_logits, dim=-1)
    premature_log_probabilities = F.log_softmax(premature_logits, dim=-1)
    first = F.kl_div(
        mature_log_probabilities.unsqueeze(0), mixture, reduction="none"
    ).mean(dim=-1)
    second = F.kl_div(
        premature_log_probabilities, mixture, reduction="none"
    ).mean(dim=-1)
    return (0.5 * (first + second)).mean(dim=-1)


def mole_calibrate(
    parent_logits,
    original_final_logits,
    hidden_states,
    lm_head,
    prompt_attention_masses,
    top_n=5,
    final_layers=3,
):
    """Apply the released MoLE greedy CHAIR formula to one decoding step.

    ``parent_logits`` is either the ordinary final logits (MoLE-only) or the
    fixed-ASCD logits (hybrid).  Expert routing is always computed from the
    unmodified positive forward, as in the released MoLE implementation.
    The official CHAIR defaults take the code path that sums, independently
    for every vocabulary item, the largest two of the final, gated late-layer,
    and attention-selected-layer logits.
    """
    if parent_logits.ndim != 2 or parent_logits.shape[0] != 1:
        raise ValueError("MoLE requires parent logits with shape [1, vocabulary]")
    if original_final_logits.shape != parent_logits.shape:
        raise ValueError("MoLE original and parent logits must have equal shape")
    if hidden_states is None or len(hidden_states) < 2:
        raise ValueError("MoLE requires decoder hidden states")
    num_decoder_layers = len(hidden_states) - 1
    if len(prompt_attention_masses) != num_decoder_layers:
        raise ValueError(
            "MoLE attention masses must contain one value per decoder layer"
        )
    top_n = int(top_n)
    final_layers = int(final_layers)
    vocabulary = parent_logits.shape[-1]
    if not 1 <= top_n < vocabulary:
        raise ValueError("MoLE top_n must be in [1, vocabulary)")
    if not 1 <= final_layers <= num_decoder_layers:
        raise ValueError("MoLE final_layers is outside the decoder")

    masses = torch.as_tensor(
        prompt_attention_masses,
        device=parent_logits.device,
        dtype=torch.float32,
    ).flatten()
    if masses.numel() != num_decoder_layers or not torch.isfinite(masses).all():
        raise ValueError("MoLE prompt attention masses are invalid")
    attention_layer = int(torch.argmax(masses).item())

    late_layers = list(range(num_decoder_layers - final_layers, num_decoder_layers))
    required_layers = sorted(set(late_layers + [attention_layer]))
    layer_logits = {}
    for layer in required_layers:
        layer_logits[layer] = lm_head(hidden_states[layer][:, -1, :]).float()

    final = original_final_logits.float()
    _, top_indices = torch.topk(final, k=top_n, dim=-1)
    all_indices = torch.arange(vocabulary, device=final.device)
    remaining_indices = all_indices[
        ~torch.isin(all_indices, top_indices.squeeze(0))
    ]
    top_final = final[:, top_indices.squeeze(0)]
    remaining_final = final[:, remaining_indices]
    top_premature = torch.stack(
        [layer_logits[layer][:, top_indices.squeeze(0)] for layer in late_layers],
        dim=0,
    )
    remaining_premature = torch.stack(
        [layer_logits[layer][:, remaining_indices] for layer in late_layers],
        dim=0,
    )
    top_jsd = _official_js_divergence(top_final, top_premature)
    remaining_jsd = _official_js_divergence(
        remaining_final, remaining_premature
    )
    top_choice = int(torch.argmax(top_jsd).item())
    remaining_choice = int(torch.argmin(remaining_jsd).item())
    expert_enabled = top_choice == remaining_choice
    expert_layer = late_layers[top_choice] if expert_enabled else 0
    if expert_layer not in layer_logits:
        layer_logits[expert_layer] = lm_head(
            hidden_states[expert_layer][:, -1, :]
        ).float()

    expert = layer_logits[expert_layer] if expert_enabled else torch.zeros_like(final)
    attention_expert = layer_logits[attention_layer]
    stacked = torch.stack((parent_logits.float(), expert, attention_expert), dim=0)
    calibrated = torch.topk(stacked, k=2, dim=0).values.sum(dim=0)
    if not torch.isfinite(calibrated).all():
        raise ValueError("MoLE produced nonfinite logits")

    top_two_sources = torch.topk(stacked, k=2, dim=0).indices
    source_counts = {
        "parent": int((top_two_sources == 0).sum().item()),
        "late_expert": int((top_two_sources == 1).sum().item()),
        "attention_expert": int((top_two_sources == 2).sum().item()),
    }
    return calibrated.to(parent_logits.dtype), {
        "attention_layer": attention_layer,
        "attention_mass_min": float(masses.min().item()),
        "attention_mass_mean": float(masses.mean().item()),
        "attention_mass_max": float(masses.max().item()),
        "late_layers": late_layers,
        "top_jsd": [float(value) for value in top_jsd.detach().cpu().tolist()],
        "remaining_jsd": [
            float(value) for value in remaining_jsd.detach().cpu().tolist()
        ],
        "top_choice": top_choice,
        "remaining_choice": remaining_choice,
        "expert_enabled": bool(expert_enabled),
        "expert_layer": int(expert_layer),
        "source_memberships": source_counts,
        "parent_selected_token_id": int(torch.argmax(parent_logits, dim=-1)[0].item()),
        "mole_selected_token_id": int(torch.argmax(calibrated, dim=-1)[0].item()),
    }
