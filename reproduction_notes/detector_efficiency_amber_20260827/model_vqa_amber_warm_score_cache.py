#!/usr/bin/env python3
"""Process-local warm visual-score-cache timing adapter for frozen Detector-ASCD.

It replaces only the OWLv2 ``set_image`` call with an exact, validated score
vector from the same frozen OWLv2 model and AMBER image.  The base AMBER
adapter still configures the same frozen Detector-ASCD decoder and performs
all generation.  This script is solely for a separately labelled warm-cache
cost measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments_v3.eval import model_vqa_amber_detector as base


class WarmScoreCacheRuntime:
    score_cache: dict[str, dict[str, float]] = {}
    current_image_file: str | None = None
    calls: list[str] = []

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.model_path, self.device, self.scores = str(model_path), str(device), {}

    def set_image(self, _image) -> dict[str, float]:
        image_file = self.current_image_file
        if image_file not in self.score_cache:
            raise KeyError(f"warm score cache missing image: {image_file}")
        self.calls.append(str(image_file))
        self.scores = dict(self.score_cache[str(image_file)])
        return dict(self.scores)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = base.parser()
    parser.add_argument("--detector-score-cache-file", type=Path, required=True)
    parser.add_argument("--warm-cache-meta-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.detector_grounded_audit_file:
        raise ValueError("warm-cache timing requires --detector-grounded-audit-file")
    if args.warm_cache_meta_file.exists():
        raise FileExistsError(f"Refusing to overwrite {args.warm_cache_meta_file}")
    cache_payload = json.loads(args.detector_score_cache_file.read_text(encoding="utf-8"))
    cache = cache_payload.get("score_cache")
    if not isinstance(cache, dict) or not cache:
        raise ValueError("invalid score cache")
    WarmScoreCacheRuntime.score_cache = {
        str(image_file): {str(name): float(score) for name, score in scores.items()}
        for image_file, scores in cache.items()
    }
    WarmScoreCacheRuntime.calls = []

    native_open = base.Image.open

    def tracked_open(path, *open_args, **open_kwargs):
        WarmScoreCacheRuntime.current_image_file = Path(path).name
        return native_open(path, *open_args, **open_kwargs)

    base.Image.open = tracked_open
    base.Owlv2ObjectRuntime = WarmScoreCacheRuntime
    base.load_configs(args)
    base.evaluate(args)
    audit = json.loads(Path(args.detector_grounded_audit_file).read_text(encoding="utf-8"))
    image_ids = [int(row["image_id"]) for row in audit.get("records", [])]
    meta = {
        "schema_version": 1,
        "mode": "Detector-ASCD warm validated visual-score cache",
        "score_cache_file": str(args.detector_score_cache_file),
        "score_cache_sha256": sha256(args.detector_score_cache_file),
        "cache_entries": len(WarmScoreCacheRuntime.score_cache),
        "set_image_calls": len(WarmScoreCacheRuntime.calls),
        "set_image_files": WarmScoreCacheRuntime.calls,
        "response_ids": image_ids,
        "note": "OWLv2 model load and visual forward are intentionally excluded; Detector-ASCD decoding is unchanged.",
    }
    args.warm_cache_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with args.warm_cache_meta_file.open("x", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
