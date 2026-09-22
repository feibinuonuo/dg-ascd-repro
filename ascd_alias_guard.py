"""Default-off object-instance alias evidence for Detector-Grounded ASCD.

This module is initially used in observe-only mode.  It never changes ASCD
scores or selected tokens: it records whether the already frozen hard detector
would act, together with an independently cached OWLv2 synonym/alias score for
the same canonical CHAIR object.  A constrained OIAG decoder may only be
implemented if the separately frozen calibration certificate passes.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import torch

from ascd_detector_grounded import terminal_decoded_chair_object


@lru_cache(maxsize=1)
def chair_object_aliases() -> Dict[str, Tuple[str, ...]]:
    """Return exact CHAIR synonym lists keyed by their canonical object."""
    from experiments_v3.eval.chair_utils import synonyms_txt

    result: Dict[str, Tuple[str, ...]] = {}
    for line in synonyms_txt.splitlines():
        terms = tuple(dict.fromkeys(
            term.strip().lower() for term in line.split(", ")
            if re.fullmatch(r"[a-z]+(?: [a-z]+)*", term.strip().lower())
        ))
        if terms:
            result[terms[0]] = terms
    if not result:
        raise RuntimeError("No CHAIR object aliases were loaded")
    return result


def alias_support_from_scores(alias_scores: Dict[str, float]) -> float:
    if not alias_scores:
        raise ValueError("Alias support requires at least one alias score")
    return max(float(value) for value in alias_scores.values())


def alias_guard_veto(
    *, canonical_support: float, alias_support: float,
    hard_threshold: float, alias_threshold: float,
) -> bool:
    """Whether positive alias evidence vetoes the existing hard-detector mask."""
    return bool(
        float(canonical_support) < float(hard_threshold)
        and float(alias_support) >= float(alias_threshold)
    )


class Owlv2AliasRuntime:
    """Caches one alias-prompt score map per canonical object for one image."""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float16
        ).eval().to(self.device)
        self.image = None
        self.cache: Dict[str, Dict[str, float]] = {}

    def set_image(self, image) -> None:
        self.image = image
        self.cache = {}

    @torch.inference_mode()
    def scores_for(self, canonical_object: str) -> Dict[str, float]:
        if self.image is None:
            raise RuntimeError("OWLv2 alias runtime has no current image")
        if canonical_object in self.cache:
            return dict(self.cache[canonical_object])
        aliases = chair_object_aliases().get(canonical_object)
        if aliases is None:
            raise KeyError(f"No CHAIR aliases for {canonical_object!r}")
        prompts = [f"a photo of a {alias}" for alias in aliases]
        encoded = self.processor(text=[prompts], images=self.image, return_tensors="pt")
        encoded = {
            key: (value.to(self.device, dtype=torch.float16) if key == "pixel_values" else value.to(self.device))
            for key, value in encoded.items()
        }
        logits = self.model(**encoded).logits[0]
        if logits.ndim != 2 or logits.shape[1] != len(aliases):
            raise RuntimeError(
                f"Unexpected OWLv2 alias logits {tuple(logits.shape)} for {len(aliases)} aliases"
            )
        values = torch.sigmoid(logits).amax(dim=0).detach().float().cpu().tolist()
        self.cache[canonical_object] = {
            alias: float(score) for alias, score in zip(aliases, values)
        }
        return dict(self.cache[canonical_object])


def observe_alias_guard_candidates(
    scores: torch.Tensor,
    *, tokenizer, generated_token_ids: Sequence[int],
    canonical_support_scores: Dict[str, float], alias_runtime: Owlv2AliasRuntime,
    hard_threshold: float, top_k: int,
) -> Dict[str, object]:
    """Record fixed-ASCD object candidates without modifying ``scores``."""
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("Alias observation requires batch_size=1")
    if int(top_k) < 1:
        raise ValueError("Alias observation top_k must be positive")
    finite_ids = torch.nonzero(torch.isfinite(scores[0]), as_tuple=False).flatten()
    if finite_ids.numel() == 0:
        raise ValueError("ASCD score distribution has no finite candidate")
    k = min(int(top_k), int(finite_ids.numel()))
    top_values, top_ids = torch.topk(scores[0], k=k)
    candidates: List[Dict[str, object]] = []
    for rank, (value, token_id) in enumerate(zip(top_values.tolist(), top_ids.tolist())):
        token_id = int(token_id)
        object_name = terminal_decoded_chair_object(tokenizer, generated_token_ids, token_id)
        if object_name is None:
            continue
        canonical = canonical_support_scores.get(object_name)
        if canonical is None:
            continue
        aliases = alias_runtime.scores_for(object_name)
        alias_support = alias_support_from_scores(aliases)
        candidates.append({
            "rank": int(rank), "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "object": object_name, "canonical_detector_score": float(canonical),
            "alias_scores": aliases, "alias_support": float(alias_support),
            "hard_threshold": float(hard_threshold),
            "would_hard_mask": bool(float(canonical) < float(hard_threshold)),
            "post_processor_score": float(value),
        })
    return {
        "step": len(generated_token_ids), "top_k": k,
        "original_top_token_id": int(top_ids[0].item()),
        "original_top_token": tokenizer.convert_ids_to_tokens(int(top_ids[0].item())),
        "object_candidates": candidates,
    }
