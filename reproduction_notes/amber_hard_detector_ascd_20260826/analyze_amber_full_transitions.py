#!/usr/bin/env python3
"""Audit caption/action transitions for the frozen AMBER full cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import spacy

from score_amber_paired import per_image_metrics


class CachedNLP:
    def __init__(self, nlp):
        self.nlp, self.cache = nlp, {}

    def __call__(self, text):
        if text not in self.cache:
            self.cache[text] = self.nlp(text)
        return self.cache[text]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--safe-words", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    fixed = json.loads(args.fixed.read_text(encoding="utf-8"))
    detector = json.loads(args.detector.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    association = json.loads(args.association.read_text(encoding="utf-8"))
    safe_words = {x.strip() for x in args.safe_words.read_text(encoding="utf-8").splitlines() if x.strip()}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))["records"]
    if not (len(fixed) == len(detector) == len(audit)):
        raise ValueError("arms and audit must be image-aligned")
    nlp = CachedNLP(spacy.load("en_core_web_lg"))
    transitions = {key: 0 for key in ("fixed_clean_detector_clean", "fixed_clean_detector_hall", "fixed_hall_detector_clean", "fixed_hall_detector_hall")}
    changed = masks_changed = masks_unchanged = selection_changed = 0
    word_deltas = []
    action_images = {key: 0 for key in ("candidate_images", "masked_images", "selection_change_images", "finite_protection_images")}
    for f_row, d_row, a_row in zip(fixed, detector, audit):
        image_id = int(f_row["id"])
        if image_id != int(d_row["id"]) or image_id != int(a_row["image_id"]):
            raise ValueError("misaligned image IDs")
        ann = annotations[image_id - 1]
        metric_args = (list(ann["truth"]), list(ann["hallu"]), association, safe_words, nlp, 0.8)
        f_clean = per_image_metrics(str(f_row["response"]), *metric_args)["chair_score"] == 0
        d_clean = per_image_metrics(str(d_row["response"]), *metric_args)["chair_score"] == 0
        transitions[f"fixed_{'clean' if f_clean else 'hall'}_detector_{'clean' if d_clean else 'hall'}"] += 1
        response_changed = f_row["response"] != d_row["response"]
        changed += int(response_changed)
        has_mask = int(a_row["masked_candidates"]) > 0
        masks_changed += int(has_mask and response_changed)
        masks_unchanged += int(has_mask and not response_changed)
        selection_changed += int(a_row["selection_changes"] > 0 and response_changed)
        word_deltas.append(len(str(d_row["response"]).split()) - len(str(f_row["response"]).split()))
        action_images["candidate_images"] += int(a_row["object_candidates_considered"] > 0)
        action_images["masked_images"] += int(has_mask)
        action_images["selection_change_images"] += int(a_row["selection_changes"] > 0)
        action_images["finite_protection_images"] += int(a_row["no_finite_protections"] > 0)
    payload = {
        "schema_version": 1,
        "num_images": len(fixed),
        "caption_transition_counts": transitions,
        "caption_exact_match_count": len(fixed) - changed,
        "caption_changed_count": changed,
        "masked_caption_changed_count": masks_changed,
        "masked_caption_unchanged_count": masks_unchanged,
        "selection_change_caption_changed_count": selection_changed,
        "detector_action_image_counts": action_images,
        "word_delta_detector_minus_fixed": {
            "mean": sum(word_deltas) / len(word_deltas),
            "min": min(word_deltas),
            "max": max(word_deltas),
            "positive": sum(x > 0 for x in word_deltas),
            "zero": sum(x == 0 for x in word_deltas),
            "negative": sum(x < 0 for x in word_deltas),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
