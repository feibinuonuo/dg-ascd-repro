#!/usr/bin/env python3
"""Export auditable AMBER transition candidates for a qualitative paper panel."""

from __future__ import annotations

import json
from pathlib import Path

import spacy

from score_amber_paired import per_image_metrics


class CachedNLP:
    def __init__(self, nlp):
        self.nlp = nlp
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            self.cache[text] = self.nlp(text)
        return self.cache[text]


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
DATA = ROOT / "data" / "amber_official_20260826" / "source" / "data"
OUTPUT = Path(__file__).resolve().parent / "scores" / "figure_case_candidates_v1.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    fixed = json.loads((RUNS / "amber-fixed-full-n1004-v1.responses.json").read_text())
    detector = json.loads((RUNS / "amber-detector-full-n1004-v1.responses.json").read_text())
    audit = json.loads((RUNS / "amber-detector-full-n1004-v1.detector.json").read_text())["records"]
    annotations = json.loads((DATA / "annotations.json").read_text())
    association = json.loads((DATA / "relation.json").read_text())
    safe_words = {x.strip() for x in (DATA / "safe_words.txt").read_text().splitlines() if x.strip()}
    nlp = CachedNLP(spacy.load("en_core_web_lg"))
    candidates = []
    for fixed_row, detector_row, audit_row in zip(fixed, detector, audit):
        image_id = int(fixed_row["id"])
        if image_id != int(detector_row["id"]) or image_id != int(audit_row["image_id"]):
            raise ValueError("image alignment failure")
        annotation = annotations[image_id - 1]
        metric_args = (list(annotation["truth"]), list(annotation["hallu"]), association, safe_words, nlp, 0.8)
        fixed_metrics = per_image_metrics(str(fixed_row["response"]), *metric_args)
        detector_metrics = per_image_metrics(str(detector_row["response"]), *metric_args)
        changed_events = [event for event in audit_row["events"] if event.get("selection_changed")]
        if fixed_metrics["chair_score"] != detector_metrics["chair_score"]:
            candidates.append({
                "image_id": image_id,
                "transition": (
                    "hall_to_clean"
                    if fixed_metrics["chair_score"] > 0 and detector_metrics["chair_score"] == 0
                    else "clean_to_hall"
                    if fixed_metrics["chair_score"] == 0 and detector_metrics["chair_score"] > 0
                    else "hall_count_changed"
                ),
                "fixed_metrics": fixed_metrics,
                "detector_metrics": detector_metrics,
                "truth": annotation["truth"],
                "hallu": annotation["hallu"],
                "fixed_response": fixed_row["response"],
                "detector_response": detector_row["response"],
                "audit": {
                    "masked_candidates": audit_row["masked_candidates"],
                    "selection_changes": audit_row["selection_changes"],
                    "no_finite_protections": audit_row["no_finite_protections"],
                    "changed_events": changed_events,
                },
            })
    OUTPUT.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n")
    summary = {key: sum(row["transition"] == key for row in candidates) for key in {row["transition"] for row in candidates}}
    print(json.dumps({"output": str(OUTPUT), "num_candidates": len(candidates), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
