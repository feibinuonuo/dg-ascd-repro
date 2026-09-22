#!/usr/bin/env python3
"""Precompute frozen OWLv2 image-support vectors and measure visual cost."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from ascd_detector_grounded import Owlv2ObjectRuntime
from experiments_v3.eval.model_vqa_amber_detector import load_questions


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=Path, required=True)
    parser.add_argument("--query-file", type=str, required=True)
    parser.add_argument("--manifest-file", type=str, required=True)
    parser.add_argument("--question-limit", type=int, default=20)
    parser.add_argument("--detector-model-path", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    questions, query_sha256, manifest_sha256 = load_questions(
        args.query_file, args.manifest_file, 1, 0, args.question_limit
    )
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    runtime = Owlv2ObjectRuntime(args.detector_model_path, device="cuda")
    synchronize()
    model_load_seconds = time.perf_counter() - start
    image_rows, cache = [], {}
    for question in questions:
        image_file = str(question["image"])
        image = Image.open(args.image_folder / image_file).convert("RGB")
        synchronize()
        started = time.perf_counter()
        scores = runtime.set_image(image)
        synchronize()
        elapsed = time.perf_counter() - started
        if not scores or not all(math.isfinite(float(value)) for value in scores.values()):
            raise RuntimeError(f"non-finite detector score vector: {image_file}")
        cache[image_file] = {str(key): float(value) for key, value in scores.items()}
        image_rows.append({"id": int(question["id"]), "image": image_file, "cold_visual_seconds": elapsed})
    lookup_seconds = []
    for image_file in cache:
        for _ in range(100):
            started = time.perf_counter()
            copied = dict(cache[image_file])
            lookup_seconds.append(time.perf_counter() - started)
            if not copied:
                raise RuntimeError("empty copied score vector")
    payload = {
        "schema_version": 1,
        "workload": "AMBER-generative immutable n20 seed20260826",
        "detector_model_path": str(Path(args.detector_model_path).resolve()),
        "query_sha256": query_sha256,
        "manifest_sha256": manifest_sha256,
        "num_images": len(image_rows),
        "owlv2_model_load_seconds": model_load_seconds,
        "cold_visual_seconds": [row["cold_visual_seconds"] for row in image_rows],
        "cold_visual_mean_seconds": sum(row["cold_visual_seconds"] for row in image_rows) / len(image_rows),
        "cold_visual_total_seconds": sum(row["cold_visual_seconds"] for row in image_rows),
        "warm_score_lookup_mean_milliseconds": 1000.0 * sum(lookup_seconds) / len(lookup_seconds),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "per_image": image_rows,
        "score_cache": cache,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"score_cache", "per_image"}}, indent=2))


if __name__ == "__main__":
    main()
