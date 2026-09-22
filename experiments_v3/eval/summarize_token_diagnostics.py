"""Validate and summarize greedy ASCD token diagnostic JSONL files."""

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict


SCALAR_METRICS = (
    "alpha_t",
    "positive_margin",
    "negative_margin",
    "positive_entropy",
    "negative_entropy",
    "positive_normalized_entropy",
    "negative_normalized_entropy",
    "branch_js_divergence",
    "candidate_js_divergence",
    "candidate_count",
    "selected_signed_branch_gap",
    "selected_contrastive_correction",
    "positive_attention.image_mass",
    "positive_attention.image_entropy",
    "positive_attention.system_mass",
    "positive_attention.history_mass",
    "negative_attention.image_mass",
    "negative_attention.image_entropy",
    "negative_attention.system_mass",
    "negative_attention.history_mass",
)

OPTIONAL_SCALAR_METRICS = (
    "alpha_components.base_alpha",
    "alpha_components.positive_image_mass",
    "alpha_components.preserve_score",
    "alpha_components.final_alpha",
    "neutral_directional.neutral_js.positive_neutral",
    "neutral_directional.neutral_js.neutral_negative",
    "neutral_directional.neutral_js.positive_negative",
    "neutral_directional.neutral_top_probability",
    "neutral_directional.g_pos",
    "neutral_directional.g_neutral",
    "neutral_directional.g_neg",
    "neutral_directional.positive_support",
    "neutral_directional.negative_support",
    "neutral_directional.correction_gap",
    "neutral_directional.joint_support",
    "neutral_directional.support_conflict",
    "neutral_attention.image_mass",
)


def dotted_get(record, path):
    value = record
    for part in path.split("."):
        value = value[part]
    return value


def dotted_has(record, path):
    try:
        dotted_get(record, path)
    except (KeyError, TypeError):
        return False
    return True


def quantile(sorted_values, probability):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def describe(values):
    values = [float(value) for value in values]
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0],
        "p10": quantile(ordered, 0.10),
        "p25": quantile(ordered, 0.25),
        "p50": quantile(ordered, 0.50),
        "p75": quantile(ordered, 0.75),
        "p90": quantile(ordered, 0.90),
        "max": ordered[-1],
    }


def load_and_validate(path, expected_samples=None):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            records.append(record)

    if not records:
        raise ValueError(f"No diagnostic records found in {path}")

    by_sample = defaultdict(list)
    for line_number, record in enumerate(records, start=1):
        for key in (
            "schema_version",
            "sample_index",
            "image_id",
            "step",
            "selected_token_id",
            "generated_text_so_far",
            "top_token_agreement",
            "ascd_changed_top_token",
            "positive_attention",
            "negative_attention",
        ):
            if key not in record:
                raise ValueError(f"Missing {key!r} in diagnostic record {line_number}")

        if record["schema_version"] != 1:
            raise ValueError(
                f"Unsupported schema_version={record['schema_version']} at record {line_number}"
            )
        for branch in ("positive_attention", "negative_attention"):
            if not record[branch].get("available", False):
                raise ValueError(
                    f"{branch} is unavailable at record {line_number}; "
                    "check eager attention replacement and diagnostics_enabled."
                )

        for metric in SCALAR_METRICS:
            try:
                value = float(dotted_get(record, metric))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing/non-numeric metric {metric!r} at record {line_number}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite metric {metric!r}={value} at record {line_number}"
                )
        for metric in OPTIONAL_SCALAR_METRICS:
            if not dotted_has(record, metric):
                continue
            value = float(dotted_get(record, metric))
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite optional metric {metric!r}={value} "
                    f"at record {line_number}"
                )

        if "neutral_directional" in record:
            for key in (
                "positive_neutral_top_agreement",
                "neutral_negative_top_agreement",
                "three_branch_top_agreement",
                "ascd_selected_differs_from_neutral_top",
                "directional_consistency",
                "neutral_top_token_id",
                "rival_token_id",
                "rival_from_positive_cutoff",
            ):
                if key not in record["neutral_directional"]:
                    raise ValueError(
                        f"Missing neutral_directional.{key!s} at record {line_number}"
                    )
            if not record.get("neutral_attention", {}).get("available", False):
                raise ValueError(
                    f"neutral_attention is unavailable at record {line_number}"
                )

        by_sample[(record["sample_index"], record["image_id"])].append(record)

    if expected_samples is not None and len(by_sample) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, found {len(by_sample)} in {path}"
        )

    for sample_key, sample_records in by_sample.items():
        steps = [int(record["step"]) for record in sample_records]
        expected_steps = list(range(len(sample_records)))
        if steps != expected_steps:
            raise ValueError(
                f"Non-contiguous steps for sample={sample_key}: "
                f"expected {expected_steps[:5]}... got {steps[:5]}..."
            )

    return records, by_sample


def make_summary(path, records, by_sample):
    optional_metrics = tuple(
        metric
        for metric in OPTIONAL_SCALAR_METRICS
        if all(dotted_has(record, metric) for record in records)
    )
    scalar_metrics = SCALAR_METRICS + optional_metrics
    summary = {
        "input": os.path.abspath(path),
        "schema_version": 1,
        "num_samples": len(by_sample),
        "num_tokens": len(records),
        "tokens_per_sample": describe(
            [len(sample_records) for sample_records in by_sample.values()]
        ),
        "top_token_agreement_rate": statistics.fmean(
            [float(record["top_token_agreement"]) for record in records]
        ),
        "ascd_changed_top_token_rate": statistics.fmean(
            [float(record["ascd_changed_top_token"]) for record in records]
        ),
        "metrics": {
            metric: describe([dotted_get(record, metric) for record in records])
            for metric in scalar_metrics
        },
        "conditional": {},
    }
    if all("neutral_directional" in record for record in records):
        summary["neutral_directional_rates"] = {
            key: statistics.fmean(
                [float(record["neutral_directional"][key]) for record in records]
            )
            for key in (
                "positive_neutral_top_agreement",
                "neutral_negative_top_agreement",
                "three_branch_top_agreement",
                "ascd_selected_differs_from_neutral_top",
                "directional_consistency",
                "rival_from_positive_cutoff",
            )
        }

    for condition_name, predicate in (
        ("top_tokens_agree", lambda row: row["top_token_agreement"]),
        ("top_tokens_disagree", lambda row: not row["top_token_agreement"]),
        ("ascd_changed_token", lambda row: row["ascd_changed_top_token"]),
        ("ascd_kept_token", lambda row: not row["ascd_changed_top_token"]),
    ):
        subset = [record for record in records if predicate(record)]
        summary["conditional"][condition_name] = {
            "count": len(subset),
            "rate": len(subset) / len(records),
            "branch_js_divergence": (
                describe([record["branch_js_divergence"] for record in subset])
                if subset
                else None
            ),
            "positive_margin": (
                describe([record["positive_margin"] for record in subset])
                if subset
                else None
            ),
            "alpha_t": (
                describe([record["alpha_t"] for record in subset])
                if subset
                else None
            ),
        }
    return summary


def write_per_sample_csv(path, by_sample):
    fieldnames = (
        "sample_index",
        "image_id",
        "num_tokens",
        "top_token_agreement_rate",
        "ascd_changed_top_token_rate",
        "mean_alpha_t",
        "mean_positive_margin",
        "mean_branch_js_divergence",
        "mean_positive_image_mass",
        "mean_negative_image_mass",
        "final_generated_text",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (sample_index, image_id), rows in sorted(by_sample.items()):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "image_id": image_id,
                    "num_tokens": len(rows),
                    "top_token_agreement_rate": statistics.fmean(
                        float(row["top_token_agreement"]) for row in rows
                    ),
                    "ascd_changed_top_token_rate": statistics.fmean(
                        float(row["ascd_changed_top_token"]) for row in rows
                    ),
                    "mean_alpha_t": statistics.fmean(row["alpha_t"] for row in rows),
                    "mean_positive_margin": statistics.fmean(
                        row["positive_margin"] for row in rows
                    ),
                    "mean_branch_js_divergence": statistics.fmean(
                        row["branch_js_divergence"] for row in rows
                    ),
                    "mean_positive_image_mass": statistics.fmean(
                        row["positive_attention"]["image_mass"] for row in rows
                    ),
                    "mean_negative_image_mass": statistics.fmean(
                        row["negative_attention"]["image_mass"] for row in rows
                    ),
                    "final_generated_text": rows[-1]["generated_text_so_far"],
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Token diagnostic JSONL path.")
    parser.add_argument("--output", required=True, help="Summary JSON path.")
    parser.add_argument(
        "--per-sample-csv",
        default=None,
        help="Optional per-image diagnostic summary CSV path.",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=None,
        help="Fail if the number of unique samples differs from this value.",
    )
    args = parser.parse_args()

    records, by_sample = load_and_validate(args.input, args.expected_samples)
    summary = make_summary(args.input, records, by_sample)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    if args.per_sample_csv:
        write_per_sample_csv(args.per_sample_csv, by_sample)

    print(
        "diagnostics_ok "
        f"samples={summary['num_samples']} tokens={summary['num_tokens']} "
        f"agreement={summary['top_token_agreement_rate']:.4f} "
        f"changed={summary['ascd_changed_top_token_rate']:.4f}"
    )
    print(f"summary={args.output}")
    if args.per_sample_csv:
        print(f"per_sample_csv={args.per_sample_csv}")


if __name__ == "__main__":
    main()
