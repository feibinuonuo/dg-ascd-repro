"""Frozen EMNLP 2025 Multi-Frequency Contrastive Decoding primitives."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SOURCE_COMMIT = "a5cd8de499aa9056171a8c8fc208eda2d3261e13"


def _gaussian_filter(image, cutoff, high_pass, device):
    """Match the released per-channel Gaussian FFT image filters."""
    if cutoff > 1.0:
        cutoff = int(cutoff)
    else:
        cutoff = int(
            math.sqrt((image.size[0] / 2) ** 2 + (image.size[1] / 2) ** 2)
            * cutoff
        )
    image = image.convert("RGB")
    image_np = np.array(image)
    filtered_channels = []
    for channel_index in range(image_np.shape[-1]):
        channel = torch.from_numpy(image_np[:, :, channel_index]).float().to(device)
        rows, cols = channel.shape
        u = torch.arange(0, rows, device=device)
        v = torch.arange(0, cols, device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
        center_u, center_v = rows // 2, cols // 2
        distance = torch.sqrt((grid_u - center_u) ** 2 + (grid_v - center_v) ** 2)
        low_transfer = torch.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
        transfer = 1 - low_transfer if high_pass else low_transfer
        shifted = torch.fft.fftshift(torch.fft.fft2(channel)) * transfer
        restored = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(shifted)))
        filtered_channels.append(restored.cpu().numpy())
    filtered = np.stack(filtered_channels, axis=-1)
    return Image.fromarray(np.clip(filtered, 0, 255).astype(np.uint8))


def gaussian_high_pass_filter(image, cutoff=0.1, device="cpu"):
    return _gaussian_filter(image, cutoff, True, torch.device(device))


def gaussian_low_pass_filter(image, cutoff=0.1, device="cpu"):
    return _gaussian_filter(image, cutoff, False, torch.device(device))


def mfcd_calibrate(original_logits, high_pass_logits, low_pass_logits,
                   high_alpha=1.0, low_alpha=1.0, beta=0.3):
    """Apply the exact released MFCD logit equation and plausibility mask."""
    original_probability = F.softmax(original_logits, dim=-1)
    maximum = original_probability.max(dim=-1, keepdim=True).values
    mask = original_probability.lt(float(beta) * maximum)
    calibrated = (
        (1.0 + float(high_alpha) + float(low_alpha)) * original_logits
        - float(low_alpha) * low_pass_logits
        - float(high_alpha) * high_pass_logits
    )
    calibrated = calibrated.masked_fill(mask, torch.finfo(calibrated.dtype).min)
    return calibrated, {
        "candidate_count": int((~mask).sum().item()),
        "original_top_token_id": int(torch.argmax(original_logits, dim=-1)[0].item()),
        "mfcd_top_token_id": int(torch.argmax(calibrated, dim=-1)[0].item()),
    }


def configuration():
    return {
        "source_commit": SOURCE_COMMIT,
        "high_alpha": 1.0,
        "low_alpha": 1.0,
        "beta": 0.3,
        "high_pass_cutoff": 0.1,
        "low_pass_cutoff": 0.1,
        "filter_type": "gaussian",
        "jsd": False,
        "entropy": False,
        "decoding": "greedy paired adaptation of the paper's fixed MFCD distribution",
    }
