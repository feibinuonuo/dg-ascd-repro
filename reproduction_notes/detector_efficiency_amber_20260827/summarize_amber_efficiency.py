#!/usr/bin/env python3
"""Summarize frozen AMBER Detector-ASCD timing runs without quality selection."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def gpu_summary(path: Path) -> dict:
    memory, utilization, power, temperature = [], [], [], []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 6:
                continue
            try:
                memory.append(float(row[2].strip()))
                utilization.append(float(row[3].strip()))
                power.append(float(row[4].strip()))
                temperature.append(float(row[5].strip()))
            except ValueError:
                continue
    if not memory:
        raise ValueError(f"no monitor rows: {path}")
    return {
        "monitor_samples": len(memory),
        "peak_memory_mib": max(memory),
        "mean_gpu_utilization_percent": statistics.mean(utilization),
        "peak_power_w": max(power),
        "peak_temperature_c": max(temperature),
    }


def generation_seconds(path: Path, expected: int) -> int:
    normalized = path.read_text(errors="replace").replace("\r", "\n")
    pattern = re.compile(rf"100%\|[^\n]*?{expected}/{expected} \[(\d+):(\d+)(?::(\d+))?<")
    matches = pattern.findall(normalized)
    if not matches:
        raise ValueError(f"missing final tqdm duration: {path}")
    first, second, third = matches[-1]
    return int(first) * 3600 + int(second) * 60 + int(third) if third else int(first) * 60 + int(second)


def response_token_count(rows, tokenizer) -> int:
    return sum(len(tokenizer.encode(str(row["response"]), add_special_tokens=False)) for row in rows)


def mean_std(rows: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in rows]
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--warm-equivalence", type=Path, required=True)
    parser.add_argument("--frozen-detector-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=False)
    rows, outputs = [], {}
    for mode in ("fixed", "detector_cold", "detector_warm"):
        for repetition in ("r1", "r2"):
            tag = f"amber-efficiency-{mode}-{repetition}-n20-seed20260826"
            meta_path = args.input_dir / f"{tag}.meta.json"
            answers_path = args.input_dir / f"{tag}.responses.json"
            log_path = args.input_dir / f"{tag}.log"
            gpu_path = args.input_dir / f"{tag}.gpu.csv"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta["exit_status"]) != 0:
                raise ValueError(f"failed run: {tag}")
            answers = json.loads(answers_path.read_text(encoding="utf-8"))
            if len(answers) != 20 or len({int(row["id"]) for row in answers}) != 20:
                raise ValueError(f"invalid answers: {tag}")
            tokens = response_token_count(answers, tokenizer)
            generated = generation_seconds(log_path, 20)
            row = {
                "mode": mode,
                "repetition": repetition,
                "gpu": int(meta["gpu"]),
                "process_wall_seconds": (int(meta["end_ns"]) - int(meta["start_ns"])) / 1e9,
                "generation_seconds": generated,
                "caption_tokens": tokens,
                "mean_caption_tokens": tokens / 20,
                "caption_tokens_per_generation_second": tokens / generated,
                "images_per_generation_second": 20 / generated,
                "answers": str(answers_path),
                **gpu_summary(gpu_path),
            }
            rows.append(row)
            outputs[(mode, repetition)] = answers
            if mode != "fixed":
                audit = json.loads((args.input_dir / f"{tag}.detector.json").read_text(encoding="utf-8"))
                if len(audit.get("records", [])) != 20:
                    raise ValueError(f"invalid detector audit: {tag}")
                row["detector_masks"] = sum(int(item["masked_candidates"]) for item in audit["records"])
                row["detector_selection_changes"] = sum(int(item["selection_changes"]) for item in audit["records"])
                if mode == "detector_warm":
                    warm = json.loads((args.input_dir / f"{tag}.warm-cache.json").read_text(encoding="utf-8"))
                    if int(warm["set_image_calls"]) != 20:
                        raise ValueError(f"invalid warm cache calls: {tag}")

    reference = json.loads(args.frozen_detector_reference.read_text(encoding="utf-8"))
    if outputs[("detector_cold", "r1")] != reference:
        raise AssertionError("cold r1 differs from frozen Detector-AMBER n20 output")
    equivalence = {
        "warm_n2_prior_passed": json.loads(args.warm_equivalence.read_text(encoding="utf-8")).get("status") == "passed",
        "fixed_r1_r2_identical": outputs[("fixed", "r1")] == outputs[("fixed", "r2")],
        "cold_r1_r2_identical": outputs[("detector_cold", "r1")] == outputs[("detector_cold", "r2")],
        "warm_r1_r2_identical": outputs[("detector_warm", "r1")] == outputs[("detector_warm", "r2")],
        "cold_warm_r1_identical": outputs[("detector_cold", "r1")] == outputs[("detector_warm", "r1")],
        "cold_warm_r2_identical": outputs[("detector_cold", "r2")] == outputs[("detector_warm", "r2")],
        "cold_r1_matches_prior_frozen_AMBER_n20": True,
    }
    if not all(equivalence.values()):
        raise AssertionError(f"non-equivalent deterministic outputs: {equivalence}")
    grouped = {}
    numeric = (
        "process_wall_seconds", "generation_seconds", "caption_tokens", "mean_caption_tokens",
        "caption_tokens_per_generation_second", "images_per_generation_second", "peak_memory_mib",
        "mean_gpu_utilization_percent", "peak_power_w", "peak_temperature_c",
    )
    for mode in ("fixed", "detector_cold", "detector_warm"):
        selected = [row for row in rows if row["mode"] == mode]
        grouped[mode] = {field: mean_std(selected, field) for field in numeric}
    fixed = grouped["fixed"]
    relative = {}
    for mode in ("detector_cold", "detector_warm"):
        relative[mode] = {
            "process_wall_seconds_delta_vs_fixed": grouped[mode]["process_wall_seconds"]["mean"] - fixed["process_wall_seconds"]["mean"],
            "process_wall_seconds_percent_vs_fixed": 100.0 * (grouped[mode]["process_wall_seconds"]["mean"] / fixed["process_wall_seconds"]["mean"] - 1.0),
            "generation_seconds_delta_vs_fixed": grouped[mode]["generation_seconds"]["mean"] - fixed["generation_seconds"]["mean"],
            "peak_memory_mib_delta_vs_fixed": grouped[mode]["peak_memory_mib"]["mean"] - fixed["peak_memory_mib"]["mean"],
        }
    score_cache = json.loads(args.score_cache.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "benchmark": "same-GPU sequential AMBER n20 frozen cost-only workload; repetitions quantify timing variance only",
        "runs": rows,
        "by_mode": grouped,
        "relative_to_fixed": relative,
        "owlv2_visual_cost": {
            key: score_cache[key]
            for key in (
                "owlv2_model_load_seconds", "cold_visual_total_seconds", "cold_visual_mean_seconds",
                "warm_score_lookup_mean_milliseconds", "torch_peak_allocated_mib",
            )
        },
        "deterministic_equivalence": equivalence,
        "interpretation_warning": "Cold unique-image end-to-end latency is primary. Warm-cache latency excludes OWLv2 model load and visual forward. Different caption lengths make aggregate timing differences descriptive, not causal quality statistics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
