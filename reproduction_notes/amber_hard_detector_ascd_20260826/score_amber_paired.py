#!/usr/bin/env python3
"""Official-faithful AMBER generative scoring plus paired image bootstrap."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import spacy
from nltk.stem import WordNetLemmatizer


def extract_nouns(text: str) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith("NN")]


def per_image_metrics(response: str, truth: list[str], hallu: list[str], association: dict, safe_words_global: set[str], nlp, similarity: float) -> dict:
    hallucination_words = {word for word, values in association.items() for word in [word, *values]}
    nouns = [word for word in extract_nouns(response) if word in hallucination_words]
    safe_words, safe_list = [], []
    for idx, word in enumerate(truth):
        values = association[word]
        safe_words.extend(values)
        safe_list.extend([idx] * len(values))
    hallu_words, hallu_list = [], []
    for idx, word in enumerate(hallu):
        values = association[word]
        hallu_words.extend(values)
        hallu_list.extend([idx] * len(values))
    safe_words.extend(truth)
    safe_list.extend([0] * len(truth))
    hallu_words.extend(hallu)
    hallu_list.extend([0] * len(hallu))
    safe_flags = [0] * len(nouns)
    safe_base = len(safe_list) - len(truth)
    hallu_base = len(hallu_list) - len(hallu)
    for idx, noun in enumerate(nouns):
        if noun in safe_words_global:
            continue
        if noun in safe_words:
            for position, word in enumerate(safe_words):
                if noun == word:
                    safe_list[safe_list[position] + safe_base if position < safe_base else position] = 1
                    break
            continue
        if noun in hallu_words:
            for position, word in enumerate(hallu_words):
                if noun == word:
                    hallu_list[hallu_list[position] + hallu_base if position < hallu_base else position] = 1
                    break
        for position, word in enumerate(hallu_words):
            if nlp(noun).similarity(nlp(word)) > similarity:
                hallu_list[hallu_list[position] + hallu_base if position < hallu_base else position] = 1
                break
        matched_safe = False
        for position, word in enumerate(safe_words):
            if nlp(noun).similarity(nlp(word)) > similarity:
                matched_safe = True
                safe_list[safe_list[position] + safe_base if position < safe_base else position] = 1
                break
        if not matched_safe:
            safe_flags[idx] = 1
    return {
        "chair_score": int(sum(safe_flags)),
        "chair_num": int(len(safe_flags)),
        "cover_score": int(sum(safe_list[-len(truth):])),
        "cover_num": int(len(truth)),
        "cog_score": int(sum(hallu_list[-len(hallu):])),
        "cog_num": int(len(hallu)),
        "non_hallu_score": int(sum(safe_flags) == 0),
        "non_hallu_num": 1,
        "caption_words": len(response.split()),
    }


def aggregate(rows: list[dict]) -> dict:
    totals = Counter()
    for row in rows:
        totals.update({key: int(value) for key, value in row["metrics"].items()})
    def rate(num: str, den: str) -> float:
        return 100.0 * totals[num] / totals[den] if totals[den] else float("nan")
    return {
        "CHAIR": rate("chair_score", "chair_num"),
        "Cover": rate("cover_score", "cover_num"),
        "Hal": 100.0 - rate("non_hallu_score", "non_hallu_num"),
        "Cog": rate("cog_score", "cog_num"),
        "mean_caption_words": totals["caption_words"] / totals["non_hallu_num"] if totals["non_hallu_num"] else float("nan"),
        "counts": dict(totals),
    }


def bootstrap_delta(left: list[dict], right: list[dict], metric: str, replicates: int, seed: int) -> dict:
    if len(left) != len(right):
        raise ValueError("paired bootstrap requires equal image counts")
    rng = np.random.default_rng(seed)
    samples = []
    size = len(left)
    for _ in range(replicates):
        selected = rng.integers(0, size, size=size)
        a, b = aggregate([left[int(i)] for i in selected]), aggregate([right[int(i)] for i in selected])
        samples.append(a[metric] - b[metric])
    point = aggregate(left)[metric] - aggregate(right)[metric]
    return {
        "point_estimate_pp": point,
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
    safe_words = {line.strip() for line in args.safe_words.read_text(encoding="utf-8").splitlines() if line.strip()}
    nlp = spacy.load("en_core_web_lg")
    raw_arms = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in {
        "Vanilla": args.vanilla, "Fixed": args.fixed, "Detector": args.detector,
    }.items()}
    ids = {name: [int(row["id"]) for row in rows] for name, rows in raw_arms.items()}
    if any(value != ids["Vanilla"] for value in ids.values()) or len(set(ids["Vanilla"])) != len(ids["Vanilla"]):
        raise ValueError("AMBER arms do not have the same ordered unique IDs")
    evaluated = {}
    for name, rows in raw_arms.items():
        evaluated[name] = [{
            "id": int(row["id"]),
            "metrics": per_image_metrics(
                str(row["response"]),
                list(annotations[int(row["id"]) - 1]["truth"]),
                list(annotations[int(row["id"]) - 1]["hallu"]),
                association, safe_words, nlp, 0.8,
            ),
        } for row in rows]
    audit = json.loads(args.detector_audit.read_text(encoding="utf-8"))
    audit_records = audit.get("records", [])
    if [int(row["image_id"]) for row in audit_records] != ids["Vanilla"]:
        raise ValueError("Detector audit IDs do not match AMBER answers")
    audit_summary = {
        "records": len(audit_records),
        "object_candidates_considered": sum(int(row["object_candidates_considered"]) for row in audit_records),
        "masked_candidates": sum(int(row["masked_candidates"]) for row in audit_records),
        "selection_changes": sum(int(row["selection_changes"]) for row in audit_records),
        "no_finite_protections": sum(int(row["no_finite_protections"]) for row in audit_records),
        "eos_terminated": sum(bool(row["terminated_by_eos"]) for row in audit_records),
        "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in audit_records),
    }
    payload = {
        "schema_version": 1,
        "benchmark": "AMBER-generative",
        "num_paired_images": len(ids["Vanilla"]),
        "metrics": {name: aggregate(rows) for name, rows in evaluated.items()},
        "paired_detector_minus_fixed": {metric: bootstrap_delta(evaluated["Detector"], evaluated["Fixed"], metric, args.replicates, args.seed) for metric in ("CHAIR", "Hal", "Cover", "Cog", "mean_caption_words")},
        "paired_fixed_minus_vanilla": {metric: bootstrap_delta(evaluated["Fixed"], evaluated["Vanilla"], metric, args.replicates, args.seed) for metric in ("CHAIR", "Hal", "Cover", "Cog", "mean_caption_words")},
        "detector_audit": audit_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
