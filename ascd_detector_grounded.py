"""Default-off OWLv2 object support for Detector-Grounded ASCD.

The module never changes ASCD's attention branches or contrastive formula.
At inference it caches one visual-only OWLv2 score per canonical CHAIR object
and masks only unsupported object candidates within a frozen ASCD top-k set.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

@lru_cache(maxsize=1)
def chair_canonical_map() -> Dict[str, str]:
    # Importing chair_utils triggers its legacy NLTK setup.  Keep that work
    # strictly inside the default-off detector branch rather than at module
    # import time, so historical decoding has no new side effect.
    from experiments_v3.eval.chair_utils import synonyms_txt

    mapping: Dict[str, str] = {}
    for row in synonyms_txt.splitlines():
        terms = [term.strip().lower() for term in row.split(", ") if term.strip()]
        if terms:
            mapping.update({term: terms[0] for term in terms})
    return mapping


@lru_cache(maxsize=1)
def chair_canonical_objects() -> Tuple[str, ...]:
    return tuple(sorted(set(chair_canonical_map().values())))


def canonical_chair_object(word: str) -> Optional[str]:
    cleaned = re.sub(r"[^a-z]", "", str(word).lower())
    canonical = chair_canonical_map().get(cleaned)
    return canonical if canonical in chair_canonical_objects() else None


def candidate_has_lexical_content(tokenizer, candidate_token_id: int) -> bool:
    """Return whether this candidate itself can extend a lexical object word."""
    piece = tokenizer.decode(
        [int(candidate_token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if re.fullmatch(r"\s*<[^>]+>\s*", piece):
        return False
    return bool(re.search(r"[a-zA-Z]", piece))


def terminal_decoded_chair_object(
    tokenizer, prefix_token_ids: Sequence[int], candidate_token_id: int
) -> Optional[str]:
    """Resolve an object candidate from the decoded terminal lexical word.

    This intentionally handles BPE/SentencePiece continuations by decoding the
    whole generated prefix plus candidate.  Only a terminal CHAIR synonym is
    eligible; ordinary words and unavailable detector aliases remain untouched.
    A candidate must itself add lexical content, so punctuation and special
    tokens cannot inherit the preceding object word and be masked by mistake.
    """
    if not candidate_has_lexical_content(tokenizer, candidate_token_id):
        return None
    text = tokenizer.decode(
        [int(value) for value in prefix_token_ids] + [int(candidate_token_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    match = re.search(r"([a-z]+)[^a-z]*$", text.lower())
    return canonical_chair_object(match.group(1)) if match else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Owlv2ObjectRuntime:
    """One-image cache of frozen OWLv2 object-presence scores."""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.float16
        ).eval().to(self.device)
        self.scores: Dict[str, float] = {}

    @torch.inference_mode()
    def set_image(self, image) -> Dict[str, float]:
        objects = chair_canonical_objects()
        prompts = [f"a photo of a {object_name}" for object_name in objects]
        encoded = self.processor(text=[prompts], images=image, return_tensors="pt")
        encoded = {
            key: (
                value.to(self.device, dtype=torch.float16)
                if key == "pixel_values" else value.to(self.device)
            )
            for key, value in encoded.items()
        }
        logits = self.model(**encoded).logits[0]
        if logits.ndim != 2 or logits.shape[1] != len(objects):
            raise RuntimeError(
                "Unexpected OWLv2 logits shape: "
                f"{tuple(logits.shape)} for {len(objects)} queries"
            )
        values = torch.sigmoid(logits).amax(dim=0).detach().float().cpu().tolist()
        self.scores = {
            object_name: float(score)
            for object_name, score in zip(objects, values)
        }
        return dict(self.scores)


def apply_detector_object_mask(
    scores: torch.Tensor,
    *,
    tokenizer,
    generated_token_ids: Sequence[int],
    support_scores: Dict[str, float],
    threshold: float,
    top_k: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Mask unsupported object tokens in a finite ASCD top-k, never fallback.

    Returns the original scores unchanged when no object is eligible or if an
    unexpected all-finite-token mask would leave no selectable token.
    """
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("Detector-Grounded ASCD currently requires batch_size=1")
    if int(top_k) < 1:
        raise ValueError("Detector-Grounded ASCD top_k must be positive")
    finite_ids = torch.nonzero(torch.isfinite(scores[0]), as_tuple=False).flatten()
    if finite_ids.numel() == 0:
        raise ValueError("ASCD score distribution has no finite candidate")
    k = min(int(top_k), int(finite_ids.numel()))
    top_values, top_ids = torch.topk(scores[0], k=k)
    original_top = int(top_ids[0].item())
    object_candidates: List[Dict[str, object]] = []
    mask_ids: List[int] = []
    for rank, (value, token_id) in enumerate(zip(top_values.tolist(), top_ids.tolist())):
        token_id = int(token_id)
        object_name = terminal_decoded_chair_object(
            tokenizer, generated_token_ids, token_id
        )
        if object_name is None:
            continue
        support = support_scores.get(object_name)
        eligible = support is not None
        masked = bool(eligible and float(support) < float(threshold))
        object_candidates.append({
            "rank": int(rank),
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "object": object_name,
            "detector_score": None if support is None else float(support),
            "threshold": float(threshold),
            "masked": masked,
            "post_processor_score": float(value),
        })
        if masked:
            mask_ids.append(token_id)

    result = scores
    protected_no_finite = False
    if mask_ids:
        candidate = scores.clone()
        candidate[0, torch.tensor(mask_ids, device=scores.device, dtype=torch.long)] = -float("inf")
        if torch.isfinite(candidate).any():
            result = candidate
        else:
            protected_no_finite = True
            mask_ids = []
            for item in object_candidates:
                item["masked"] = False
    selected = int(torch.argmax(result, dim=-1)[0].item())
    event = {
        "step": len(generated_token_ids),
        "original_top_token_id": original_top,
        "original_top_token": tokenizer.convert_ids_to_tokens(original_top),
        "selected_token_id": selected,
        "selected_token": tokenizer.convert_ids_to_tokens(selected),
        "top_k": k,
        "object_candidates": object_candidates,
        "masked_token_ids": mask_ids,
        "selection_changed": bool(selected != original_top),
        "protected_no_finite": protected_no_finite,
    }
    return result, event
