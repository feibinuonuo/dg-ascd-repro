"""Frozen ICCV 2025 FuzzyCD primitives and deterministic image filters."""

import math

import cv2
import numpy as np
import torch
from PIL import Image


SOURCE_COMMIT = "fd534a9a0f8551aa1b5480a7bcbc7925b112d216"
SHARPEN_STRENGTHS = (50, 150, 300, 500)


def gaussian_memberships(mean, std, minimum, maximum, device, dtype=torch.float32):
    if not math.isfinite(float(std)) or float(std) <= 0:
        raise ValueError("FuzzyCD calibration std must be finite and positive")
    x_range = torch.linspace(float(minimum) - 1.0, float(maximum) + 1.0, 100,
                             device=device, dtype=dtype)
    center = torch.as_tensor(float(mean), device=device, dtype=dtype)
    scale = torch.as_tensor(float(std), device=device, dtype=dtype)
    low = torch.exp(-((x_range - (center - scale)) ** 2) / (2 * scale ** 2))
    med = torch.exp(-((x_range - center) ** 2) / (2 * scale ** 2))
    high = torch.exp(-((x_range - (center + scale)) ** 2) / (2 * scale ** 2))
    return x_range, (low, med, high)


def torch_interp(x, xp, fp):
    value = x.reshape(-1)
    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    intercepts = fp[:-1] - slopes * xp[:-1]
    indices = torch.bucketize(value, xp) - 1
    indices = torch.clamp(indices, 0, len(slopes) - 1)
    return slopes[indices] * value + intercepts[indices]


def js_divergence(logits_p, logits_q):
    p = torch.softmax(logits_p.float(), dim=-1).clamp_min(1e-12)
    q = torch.softmax(logits_q.float(), dim=-1).clamp_min(1e-12)
    m = 0.5 * (p + q)
    # Preserve the released PyTorch call order exactly: kl_div(log(p), m).
    return 0.5 * (
        torch.nn.functional.kl_div(p.log(), m, reduction="batchmean")
        + torch.nn.functional.kl_div(q.log(), m, reduction="batchmean")
    )


def _fuzzy_filtered_logits(x_range, memberships, original_confidence,
                           filtered_confidence, filtered_logits):
    low, med, high = memberships
    m1 = [torch_interp(original_confidence, x_range, fn) for fn in (low, med, high)]
    m2 = [torch_interp(filtered_confidence, x_range, fn) for fn in (low, med, high)]
    rules = [a * b for a in m1 for b in m2]
    consequents = (1.5, 1.0, -0.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0)
    denominator = torch.stack(rules).sum().clamp_min(1e-12)
    scale = sum(rule * value for rule, value in zip(rules, consequents)) / denominator
    return scale.to(filtered_logits.dtype) * filtered_logits, float(scale.item())


def fuzzycd_calibrate(original_logits, filtered_logits, calibration, beta=0.1,
                      parent_logits=None):
    """Apply the released four-filter Takagi-Sugeno and max-JSD rule."""
    if len(filtered_logits) != 4:
        raise ValueError("FuzzyCD requires exactly four filtered-image logits")
    x_range, memberships = gaussian_memberships(
        calibration["mean"], calibration["std"], calibration["min"],
        calibration["max"], original_logits.device,
    )
    original_conf = torch.log_softmax(original_logits.float(), dim=-1).max(dim=-1).values
    fuzzy_logits, scales, divergences = [], [], []
    for branch in filtered_logits:
        filtered_conf = torch.log_softmax(branch.float(), dim=-1).max(dim=-1).values
        fuzzy, scale = _fuzzy_filtered_logits(
            x_range, memberships, original_conf, filtered_conf, branch,
        )
        fuzzy_logits.append(fuzzy)
        scales.append(scale)
        divergences.append(float(js_divergence(fuzzy, original_logits).item()))
    selected = int(np.argmax(divergences))
    if parent_logits is None:
        parent_logits = original_logits
    # Official branch: 2*original-fuzzy.  For a hybrid parent, preserve that
    # direct correction while avoiding a second multiplication of ASCD logits.
    adjusted = parent_logits + original_logits - fuzzy_logits[selected]
    cutoff = math.log(float(beta)) + original_logits.max(dim=-1, keepdim=True).values
    adjusted = adjusted.masked_fill(original_logits < cutoff, -float("inf"))
    return adjusted, {
        "selected_filter_index": selected,
        "selected_strength": int(SHARPEN_STRENGTHS[selected]),
        "fuzzy_scales": scales,
        "js_divergences": divergences,
        "candidate_count": int((original_logits >= cutoff).sum().item()),
        "original_top_token_id": int(torch.argmax(original_logits, dim=-1)[0].item()),
        "parent_top_token_id": int(torch.argmax(parent_logits, dim=-1)[0].item()),
        "fuzzycd_top_token_id": int(torch.argmax(adjusted, dim=-1)[0].item()),
    }


def sharpen_images(image):
    array = np.asarray(image.convert("RGB"))
    results = []
    for strength in SHARPEN_STRENGTHS:
        side = (1.0 - strength) / 8.0
        kernel = np.array([[side, side, side], [side, strength, side],
                           [side, side, side]], dtype=np.float32)
        results.append(Image.fromarray(cv2.filter2D(array, -1, kernel)))
    return results


def configuration(calibration=None):
    payload = {
        "source_commit": SOURCE_COMMIT,
        "filters": "sharpen",
        "strengths": list(SHARPEN_STRENGTHS),
        "consequents": [1.5, 1.0, -0.5, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0],
        "selected_branch": "maximum Jensen-Shannon divergence",
        "contrast_alpha": 1.0,
        "beta": 0.1,
        "decoding": "greedy LLaVA adaptation of released top-1 path",
    }
    if calibration is not None:
        payload["calibration"] = dict(calibration)
    return payload
