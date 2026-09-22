"""Paired comparison of two CHAIR diagnostic runs on the same image subset."""

import argparse
import json
import os
import random
import statistics
from collections import defaultdict

from .summarize_token_diagnostics import quantile


TOKEN_METRICS = {
    "alpha_t": lambda row: row["alpha_t"],
    "positive_margin": lambda row: row["positive_margin"],
    "positive_normalized_entropy": lambda row: row["positive_normalized_entropy"],
    "branch_js_divergence": lambda row: row["branch_js_divergence"],
    "positive_image_mass": lambda row: row["positive_attention"]["image_mass"],
    "negative_image_mass": lambda row: row["negative_attention"]["image_mass"],
    "selected_signed_branch_gap": lambda row: row["selected_signed_branch_gap"],
    "top_token_agreement": lambda row: float(row["top_token_agreement"]),
    "ascd_changed_top_token": lambda row: float(row["ascd_changed_top_token"]),
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_paired_delta(pairs, repetitions, seed):
    if not pairs:
        return None
    observed = statistics.fmean(candidate - baseline for baseline, candidate in pairs)
    if len(pairs) == 1 or repetitions <= 0:
        return {
            "paired_samples": len(pairs),
            "baseline_mean": statistics.fmean(pair[0] for pair in pairs),
            "candidate_mean": statistics.fmean(pair[1] for pair in pairs),
            "delta_candidate_minus_baseline": observed,
            "bootstrap_ci95": None,
        }

    rng = random.Random(seed)
    deltas = []
    for _ in range(repetitions):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(
            statistics.fmean(candidate - baseline for baseline, candidate in sampled)
        )
    deltas.sort()
    return {
        "paired_samples": len(pairs),
        "baseline_mean": statistics.fmean(pair[0] for pair in pairs),
        "candidate_mean": statistics.fmean(pair[1] for pair in pairs),
        "delta_candidate_minus_baseline": observed,
        "bootstrap_ci95": [
            quantile(deltas, 0.025),
            quantile(deltas, 0.975),
        ],
    }


def chair_sentence_values(chair):
    return {
        sentence["image_id"]: {
            "CHAIRs": float(sentence["metrics"]["CHAIRs"]),
            "hallucinated_mentions": len(sentence["mscoco_hallucinated_words"]),
            "generated_object_mentions": len(sentence["mscoco_generated_words"]),
            "recalled_gt_objects": len(
                set(sentence["mscoco_generated_words"])
                & set(sentence["mscoco_gt_words"])
            ),
            "gt_objects": len(sentence["mscoco_gt_words"]),
        }
        for sentence in chair["sentences"]
    }


def aggregate_chair_metric(by_image, sampled_images, metric):
    records = [by_image[image_id] for image_id in sampled_images]
    if metric == "CHAIRs":
        return statistics.fmean(record["CHAIRs"] for record in records)
    if metric == "CHAIRi":
        numerator = sum(record["hallucinated_mentions"] for record in records)
        denominator = sum(record["generated_object_mentions"] for record in records)
    elif metric == "Recall":
        numerator = sum(record["recalled_gt_objects"] for record in records)
        denominator = sum(record["gt_objects"] for record in records)
    else:
        raise ValueError(f"Unsupported CHAIR metric: {metric}")
    return numerator / denominator if denominator else 0.0


def bootstrap_paired_chair_delta(
    baseline_by_image,
    candidate_by_image,
    image_ids,
    metric,
    repetitions,
    seed,
):
    baseline_value = aggregate_chair_metric(baseline_by_image, image_ids, metric)
    candidate_value = aggregate_chair_metric(candidate_by_image, image_ids, metric)
    result = {
        "paired_samples": len(image_ids),
        "estimand": "corpus_level_ratio" if metric != "CHAIRs" else "sentence_rate",
        "baseline_mean": baseline_value,
        "candidate_mean": candidate_value,
        "delta_candidate_minus_baseline": candidate_value - baseline_value,
        "bootstrap_ci95": None,
    }
    if len(image_ids) == 1 or repetitions <= 0:
        return result

    rng = random.Random(seed)
    deltas = []
    for _ in range(repetitions):
        sampled_images = [image_ids[rng.randrange(len(image_ids))] for _ in image_ids]
        deltas.append(
            aggregate_chair_metric(candidate_by_image, sampled_images, metric)
            - aggregate_chair_metric(baseline_by_image, sampled_images, metric)
        )
    deltas.sort()
    result["bootstrap_ci95"] = [
        quantile(deltas, 0.025),
        quantile(deltas, 0.975),
    ]
    return result


def token_values_by_label(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        image_id = row["image_id"]
        labels = ("all", row["chair_label"])
        for label in labels:
            for metric, getter in TOKEN_METRICS.items():
                grouped[label][(image_id, metric)].append(float(getter(row)))

    result = defaultdict(lambda: defaultdict(dict))
    for label, values in grouped.items():
        for (image_id, metric), metric_values in values.items():
            result[label][metric][image_id] = statistics.fmean(metric_values)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-chair", required=True)
    parser.add_argument("--candidate-chair", required=True)
    parser.add_argument(
        "--baseline-labeled",
        default=None,
        help="Optional labeled token JSONL; provide both labeled files or neither.",
    )
    parser.add_argument(
        "--candidate-labeled",
        default=None,
        help="Optional labeled token JSONL; provide both labeled files or neither.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.baseline_labeled) != bool(args.candidate_labeled):
        raise ValueError(
            "Provide both --baseline-labeled and --candidate-labeled, or neither."
        )

    baseline_chair = load_json(args.baseline_chair)
    candidate_chair = load_json(args.candidate_chair)
    baseline_sentence = chair_sentence_values(baseline_chair)
    candidate_sentence = chair_sentence_values(candidate_chair)
    chair_images = sorted(set(baseline_sentence) & set(candidate_sentence))
    if set(baseline_sentence) != set(candidate_sentence):
        raise ValueError("Baseline and candidate CHAIR files do not contain identical image ids.")

    comparison = {
        "schema_version": 2,
        "baseline": {
            "chair": os.path.abspath(args.baseline_chair),
            "labeled": (
                os.path.abspath(args.baseline_labeled)
                if args.baseline_labeled
                else None
            ),
        },
        "candidate": {
            "chair": os.path.abspath(args.candidate_chair),
            "labeled": (
                os.path.abspath(args.candidate_labeled)
                if args.candidate_labeled
                else None
            ),
        },
        "num_paired_images": len(chair_images),
        "overall_metrics": {},
        "paired_chair_metrics": {},
        "paired_token_metrics_by_label": {},
        "interpretation_warning": None,
    }

    for metric in ("CHAIRs", "CHAIRi", "Recall"):
        baseline_value = float(baseline_chair["overall_metrics"][metric])
        candidate_value = float(candidate_chair["overall_metrics"][metric])
        comparison["overall_metrics"][metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta_candidate_minus_baseline": candidate_value - baseline_value,
        }
        comparison["paired_chair_metrics"][metric] = bootstrap_paired_chair_delta(
            baseline_sentence,
            candidate_sentence,
            chair_images,
            metric,
            args.bootstrap_repetitions,
            args.seed,
        )

    if args.baseline_labeled:
        comparison["interpretation_warning"] = (
            "Token labels come from each method's own generated caption. "
            "These comparisons are diagnostic correlations, not causal estimates."
        )
        baseline_tokens = token_values_by_label(load_jsonl(args.baseline_labeled))
        candidate_tokens = token_values_by_label(load_jsonl(args.candidate_labeled))
        for label in sorted(set(baseline_tokens) | set(candidate_tokens)):
            comparison["paired_token_metrics_by_label"][label] = {}
            for metric in TOKEN_METRICS:
                baseline_by_image = baseline_tokens[label][metric]
                candidate_by_image = candidate_tokens[label][metric]
                paired_images = sorted(set(baseline_by_image) & set(candidate_by_image))
                pairs = [
                    (baseline_by_image[image_id], candidate_by_image[image_id])
                    for image_id in paired_images
                ]
                comparison["paired_token_metrics_by_label"][label][metric] = (
                    bootstrap_paired_delta(
                        pairs, args.bootstrap_repetitions, args.seed
                    )
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    print(
        "comparison_ok "
        f"paired_images={comparison['num_paired_images']} "
        f"chair_s_delta={comparison['overall_metrics']['CHAIRs']['delta_candidate_minus_baseline']:.6f} "
        f"chair_i_delta={comparison['overall_metrics']['CHAIRi']['delta_candidate_minus_baseline']:.6f} "
        f"recall_delta={comparison['overall_metrics']['Recall']['delta_candidate_minus_baseline']:.6f}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
