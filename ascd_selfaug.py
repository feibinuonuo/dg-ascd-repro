"""Frozen Self-Aug (ICLR 2026) formulas and image augmentations."""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


AUGMENTATIONS = (
    "random_crop", "color_inversion", "horizontal_flip",
    "vertical_flip", "random_mask", "noise",
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def configuration():
    return {
        "alpha": 1.0, "tau": 0.5, "crop_ratio": 2.0,
        "mask_ratio": 2.0, "noise_step": 500,
    }


def entropy_beta(logits: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("Self-Aug SAT tau must be positive")
    probability = torch.softmax(logits.float(), dim=-1)
    safe = torch.where(probability > 0, probability, torch.ones_like(probability))
    signed_entropy = torch.sum(probability * torch.log2(safe), dim=-1, keepdim=True)
    return torch.sigmoid(float(tau) * signed_entropy)


def calibrate(visual_logits, augmented_logits, alpha=1.0, tau=0.5):
    beta = entropy_beta(visual_logits, tau)
    cutoff = visual_logits.float().amax(dim=-1, keepdim=True) + torch.log(beta)
    candidate_mask = visual_logits.float() >= cutoff
    calibrated = ((1.0 + float(alpha)) * visual_logits.float()
                  - float(alpha) * augmented_logits.float())
    calibrated = calibrated.masked_fill(~candidate_mask, -float("inf"))
    visual_top = int(torch.argmax(visual_logits, dim=-1)[0].item())
    final_top = int(torch.argmax(calibrated, dim=-1)[0].item())
    return calibrated, {
        "beta_sat": float(beta[0, 0].item()),
        "signed_entropy_bits": float(
            (torch.logit(beta)[0, 0] / float(tau)).item()
        ),
        "candidate_count": int(candidate_mask[0].sum().item()),
        "visual_top_token_id": visual_top,
        "augmented_top_token_id": int(torch.argmax(augmented_logits, dim=-1)[0].item()),
        "calibrated_top_token_id": final_top,
        "calibrated_changed_from_visual_top": final_top != visual_top,
    }


def augment_image(tensor, augmentation, crop_ratio=2.0, mask_ratio=2.0, noise_step=500):
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Self-Aug requires one BCHW image tensor")
    if augmentation not in AUGMENTATIONS:
        raise ValueError(f"Unsupported Self-Aug augmentation: {augmentation}")
    if augmentation == "random_crop":
        _, _, height, width = tensor.shape
        size = int(min(height, width) // float(crop_ratio))
        return T.Resize((height, width), antialias=True)(T.RandomCrop((size, size))(tensor))
    if augmentation == "horizontal_flip":
        return TF.hflip(tensor)
    if augmentation == "vertical_flip":
        return TF.vflip(tensor)
    if augmentation == "color_inversion":
        mean = tensor.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = tensor.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        image = torch.clamp(tensor * std + mean, 0, 1)
        return (TF.invert(image) - mean) / std
    if augmentation == "random_mask":
        _, _, height, width = tensor.shape
        size = int(min(height, width) // float(mask_ratio))
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        result = tensor.clone()
        result[:, :, top:top + size, left:left + size] = 0.0
        return result
    if not 0 <= int(noise_step) < 1000:
        raise ValueError("Self-Aug noise_step must be in [0, 999]")
    betas = torch.linspace(-6, 6, 1000, device=tensor.device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alpha_bar = torch.cumprod(1 - betas, dim=0)[int(noise_step)]
    return (torch.sqrt(alpha_bar) * tensor
            + torch.sqrt(1 - alpha_bar) * torch.randn_like(tensor))
