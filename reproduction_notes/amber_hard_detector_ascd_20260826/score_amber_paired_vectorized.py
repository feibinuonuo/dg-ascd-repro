#!/usr/bin/env python3
"""Vectorized, numerically equivalent bootstrap for ``score_amber_paired.py``.

The metric extraction is deliberately imported from the frozen scorer.  This
helper changes only the implementation of resampling, not AMBER scoring rules,
the random seed, or output schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import spacy

sys.path.insert(0, str(Path(__file__).parent))
from score_amber_paired import aggregate, per_image_metrics  # noqa: E402


METRICS = {
    "CHAIR": ("chair_score", "chair_num", 1.0, 0.0),
    "Hal": ("non_hallu_score", "non_hallu_num", -1.0, 100.0),
    "Cover": ("cover_score", "cover_num", 1.0, 0.0),
    "Cog": ("cog_score", "cog_num", 1.0, 0.0),
    "mean_caption_words": ("caption_words", "non_hallu_num", 1.0 / 100.0, 0.0),
}


def bootstrap_delta(left, right, metric: str, replicates: int, seed: int) -> dict:
    num_key, den_key, scale, offset = METRICS[metric]
    left_num = np.asarray([row["metrics"][num_key] for row in left], dtype=np.float64)
    left_den = np.asarray([row["metrics"][den_key] for row in left], dtype=np.float64)
    right_num = np.asarray([row["metrics"][num_key] for row in right], dtype=np.float64)
    right_den = np.asarray([row["metrics"][den_key] for row in right], dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, replicates, 256):
        idx = rng.integers(0, len(left), size=(min(256, replicates - start), len(left)))
        left_value = offset + scale * 100.0 * left_num[idx].sum(axis=1) / left_den[idx].sum(axis=1)
        right_value = offset + scale * 100.0 * right_num[idx].sum(axis=1) / right_den[idx].sum(axis=1)
        values.append(left_value - right_value)
    samples = np.concatenate(values)
    left_value = offset + scale * 100.0 * left_num.sum() / left_den.sum()
    right_value = offset + scale * 100.0 * right_num.sum() / right_den.sum()
    return {
        "point_estimate_pp": float(left_value - right_value),
        "ci95_pp": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "bootstrap_replicates": replicates,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--safe-words", type=Path, required=True)
    parser.add_argument("--detector-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    association = json.loads(args.association.read_text(encoding="utf-8"))
    safe_words = {x.strip() for x in args.safe_words.read_text(encoding="utf-8").splitlines() if x.strip()}
    nlp = spacy.load("en_core_web_lg")
    raw = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in {
        "Vanilla": args.vanilla, "Fixed": args.fixed, "Detector": args.detector}.items()}
    ids = {name: [int(row["id"]) for row in rows] for name, rows in raw.items()}
    if any(x != ids["Vanilla"] for x in ids.values()) or len(set(ids["Vanilla"])) != len(ids["Vanilla"]):
        raise ValueError("AMBER arms do not have the same ordered unique IDs")
    evaluated = {}
    for name, rows in raw.items():
        evaluated[name] = [{"id": int(row["id"]), "metrics": per_image_metrics(
            str(row["response"]), list(annotations[int(row["id"]) - 1]["truth"]),
            list(annotations[int(row["id"]) - 1]["hallu"]), association, safe_words, nlp, 0.8)} for row in rows]
    audit_records = json.loads(args.detector_audit.read_text(encoding="utf-8")).get("records", [])
    if [int(row["image_id"]) for row in audit_records] != ids["Vanilla"]:
        raise ValueError("Detector audit IDs do not match AMBER answers")
    audit = {"records": len(audit_records)}
    for key in ("object_candidates_considered", "masked_candidates", "selection_changes", "no_finite_protections"):
        audit[key] = sum(int(row[key]) for row in audit_records)
    audit["eos_terminated"] = sum(bool(row["terminated_by_eos"]) for row in audit_records)
    audit["hit_max_new_tokens"] = sum(bool(row["hit_max_new_tokens"]) for row in audit_records)
    payload = {
        "schema_version": 1,
        "bootstrap_implementation": "vectorized-equivalent-v1",
        "benchmark": "AMBER-generative",
        "num_paired_images": len(ids["Vanilla"]),
        "metrics": {name: aggregate(rows) for name, rows in evaluated.items()},
        "paired_detector_minus_fixed": {m: bootstrap_delta(evaluated["Detector"], evaluated["Fixed"], m, args.replicates, args.seed) for m in METRICS},
        "paired_fixed_minus_vanilla": {m: bootstrap_delta(evaluated["Fixed"], evaluated["Vanilla"], m, args.replicates, args.seed) for m in METRICS},
        "detector_audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
