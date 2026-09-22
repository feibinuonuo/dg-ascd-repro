"""Dependency-light DeCo calibration used by the ASCD generation path."""

import math

import torch


def deco_calibrate(
    parent_logits,
    hidden_states,
    norm,
    lm_head,
    early_exit_layers,
    alpha=0.6,
    threshold_top_p=0.9,
    threshold_top_k=20,
):
    """Apply official DeCo greedy calibration to one parent logit vector.

    The candidate set comes from the parent distribution.  Among all
    layer/candidate pairs, DeCo selects the pair with the largest early-exit
    probability and adds the complete selected-layer logits to the parent,
    scaled by ``alpha * probability``.  Tokens outside the candidate set are
    masked exactly as in the official implementation.
    """
    if parent_logits.ndim != 2 or parent_logits.shape[0] != 1:
        raise ValueError("DeCo requires logits with shape [1, vocabulary]")
    if not 0 < float(threshold_top_p) <= 1:
        raise ValueError("threshold_top_p must be in (0, 1]")
    top_k = int(threshold_top_k)
    if top_k < 2 or top_k > parent_logits.shape[-1]:
        raise ValueError("threshold_top_k must be in [2, vocabulary]")
    layers = [int(layer) for layer in early_exit_layers]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("early_exit_layers must be nonempty and unique")
    if min(layers) < 0 or max(layers) >= len(hidden_states):
        raise ValueError("early_exit layer is outside hidden_states")

    parent_probs = torch.softmax(parent_logits.detach().float(), dim=-1)
    candidate_probs, candidate_ids = torch.topk(parent_probs, k=top_k, dim=-1)
    if not torch.isfinite(candidate_probs).all():
        raise ValueError("nonfinite DeCo parent candidate probabilities")
    cumulative = candidate_probs[0].cumsum(dim=-1)
    reached = torch.nonzero(cumulative >= float(threshold_top_p), as_tuple=False)
    candidate_count = int(reached[0, 0].item()) + 1 if len(reached) else top_k
    candidate_ids = candidate_ids[:, :candidate_count]
    candidate_probs = candidate_probs[:, :candidate_count]

    best_probability = None
    best_layer = None
    best_candidate = None
    best_logits = None
    for layer in layers:
        early_hidden = norm(hidden_states[layer])[:, -1, :]
        early_logits = lm_head(early_hidden).float()
        early_probs = torch.softmax(early_logits, dim=-1).gather(1, candidate_ids)
        if not torch.isfinite(early_probs).all():
            raise ValueError("nonfinite DeCo early-exit probabilities")
        flat_index = int(torch.argmax(early_probs).item())
        probability = float(early_probs.reshape(-1)[flat_index].item())
        if best_probability is None or probability > best_probability:
            best_probability = probability
            best_layer = layer
            best_candidate = int(candidate_ids.reshape(-1)[flat_index].item())
            best_logits = early_logits

    calibrated = parent_logits.float() + float(alpha) * best_probability * best_logits
    mask = torch.ones_like(calibrated, dtype=torch.bool)
    mask.scatter_(1, candidate_ids, False)
    calibrated = calibrated.masked_fill(mask, -float("inf"))
    selected = int(torch.argmax(calibrated, dim=-1)[0].item())
    if not math.isfinite(best_probability):
        raise ValueError("nonfinite DeCo selected probability")
    return calibrated.to(parent_logits.dtype), {
        "candidate_ids": [int(x) for x in candidate_ids[0].tolist()],
        "parent_candidate_probabilities": [float(x) for x in candidate_probs[0].tolist()],
        "selected_early_exit_layer": int(best_layer),
        "selected_premature_candidate_id": int(best_candidate),
        "premature_max_probability": float(best_probability),
        "parent_selected_token_id": int(torch.argmax(parent_logits, dim=-1)[0].item()),
        "deco_selected_token_id": selected,
    }
