#!/usr/bin/env python3
"""Score immutable POPE answers and paired arm differences.

This reproduces the project's yes/no normalization and does not select a
decoder setting.  Bootstrap samples retain the pairing by sampling question
indices jointly across the two arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_arm(value: str) -> tuple[str, Path]:
    name, marker, path = value.partition("=")
    if not marker or not name or not path:
        raise argparse.ArgumentTypeError("--arm must be NAME=PATH")
    return name, Path(path)


def normalized_prediction(value) -> int:
    text = value[0] if isinstance(value, list) and value else value
    text = str(text).strip()
    if "." in text:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return 0 if any(word in {"No", "not", "no"} for word in words) else 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels_by_question(questions: list[dict], annotation_dir: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for category in ("adversarial", "popular", "random"):
        category_questions = [row for row in questions if row["category"] == category]
        labels = [json.loads(line)["label"] for line in (annotation_dir / f"coco_pope_{category}.json").open()]
        if len(category_questions) != len(labels):
            raise ValueError(f"POPE {category} question/label mismatch: {len(category_questions)} != {len(labels)}")
        result.update({int(row["question_id"]): int(label == "yes") for row, label in zip(category_questions, labels)})
    return result


def metrics(prediction: np.ndarray, label: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((prediction == 1) & (label == 1)))
    fp = int(np.sum((prediction == 1) & (label == 0)))
    tn = int(np.sum((prediction == 0) & (label == 0)))
    fn = int(np.sum((prediction == 0) & (label == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(label.size), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / label.size if label.size else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": float(np.mean(prediction)) if label.size else 0.0,
    }


def bootstrap_difference(candidate: np.ndarray, reference: np.ndarray, label: np.ndarray, repetitions: int, seed: int) -> dict[str, list[float] | int]:
    rng = np.random.default_rng(seed)
    names = ("accuracy", "precision", "recall", "f1", "yes_ratio")
    samples = {name: [] for name in names}
    batch_size = min(500, repetitions)
    for offset in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - offset)
        indices = rng.integers(0, label.size, size=(size, label.size))
        for name in names:
            values = []
            for index in indices:
                values.append(metrics(candidate[index], label[index])[name] - metrics(reference[index], label[index])[name])
            samples[name].extend(values)
    return {
        name: [float(np.quantile(samples[name], 0.025)), float(np.quantile(samples[name], 0.975))]
        for name in names
    } | {"repetitions": repetitions}


def answer_map(path: Path) -> dict[int, int]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    result = {int(row["question_id"]): normalized_prediction(row["text"]) for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"Duplicate question id in {path}")
    return result


def self_test() -> None:
    assert normalized_prediction("No, there is not.") == 0
    assert normalized_prediction("Yes") == 1
    labels = np.array([1, 0])
    assert metrics(np.array([1, 0]), labels)["accuracy"] == 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--label-question-file", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=parse_arm, default=[])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: passed")
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.reference not in dict(args.arm):
        raise ValueError("--reference must name one --arm")

    questions = [json.loads(line) for line in args.questions.open(encoding="utf-8") if line.strip()]
    label_questions = [json.loads(line) for line in args.label_question_file.open(encoding="utf-8") if line.strip()]
    label_map = labels_by_question(label_questions, args.annotation_dir)
    expected_ids = [int(row["question_id"]) for row in questions]
    arms = {name: answer_map(path) for name, path in args.arm}
    for name, answer in arms.items():
        if set(answer) != set(expected_ids):
            raise ValueError(f"{name} does not exactly cover selected question ids")
    categories = ("adversarial", "popular", "random", "pooled")
    output: dict = {
        "schema_version": 1,
        "normalization": "project eval_pope first-sentence yes/no rule",
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256(args.questions),
        "arms": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in args.arm},
        "reference": args.reference,
        "categories": {},
    }
    for category_index, category in enumerate(categories):
        rows = questions if category == "pooled" else [row for row in questions if row["category"] == category]
        ids = [int(row["question_id"]) for row in rows]
        label = np.asarray([label_map[question_id] for question_id in ids], dtype=np.int8)
        predictions = {name: np.asarray([answers[question_id] for question_id in ids], dtype=np.int8) for name, answers in arms.items()}
        section = {"metrics": {name: metrics(prediction, label) for name, prediction in predictions.items()}}
        reference = predictions[args.reference]
        comparisons = {}
        for name, prediction in predictions.items():
            if name == args.reference:
                continue
            transitions = {
                "reference_correct_to_candidate_correct": int(np.sum((reference == label) & (prediction == label))),
                "reference_wrong_to_candidate_correct": int(np.sum((reference != label) & (prediction == label))),
                "reference_correct_to_candidate_wrong": int(np.sum((reference == label) & (prediction != label))),
                "reference_wrong_to_candidate_wrong": int(np.sum((reference != label) & (prediction != label))),
            }
            comparisons[name] = {
                "point_delta": {key: float(section["metrics"][name][key] - section["metrics"][args.reference][key]) for key in ("accuracy", "precision", "recall", "f1", "yes_ratio")},
                "paired_bootstrap_ci95": bootstrap_difference(prediction, reference, label, args.bootstrap_repetitions, args.bootstrap_seed + category_index),
                "correctness_transitions": transitions,
            }
        section["paired_vs_reference"] = comparisons
        output["categories"][category] = section
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["categories"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
