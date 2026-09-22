#!/usr/bin/env python3
"""Validate the frozen AMBER visual-score cache used only for timing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments_v3.eval.model_vqa_amber_detector import load_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--query-file", type=str, required=True)
    parser.add_argument("--manifest-file", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    questions, query_sha256, manifest_sha256 = load_questions(args.query_file, args.manifest_file, 1, 0, 20)
    expected = [str(row["image"]) for row in questions]
    cache = payload.get("score_cache", {})
    if payload.get("query_sha256") != query_sha256 or payload.get("manifest_sha256") != manifest_sha256:
        raise AssertionError("query or manifest provenance mismatch")
    if list(cache) != expected:
        raise AssertionError("score-cache image order mismatch")
    dimensions = {len(scores) for scores in cache.values()}
    if len(dimensions) != 1 or not next(iter(dimensions)):
        raise AssertionError("inconsistent score-vector dimensions")
    if not all(math.isfinite(float(value)) for scores in cache.values() for value in scores.values()):
        raise AssertionError("non-finite score")
    result = {
        "status": "passed",
        "num_images": len(expected),
        "object_score_dimension": next(iter(dimensions)),
        "query_sha256": query_sha256,
        "manifest_sha256": manifest_sha256,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
