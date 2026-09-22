"""Offline integrity and packaging checks for the DG-ASCD release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.14235107600688934


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list released artifact roles and paths")
    args = parser.parse_args()

    with (ROOT / "ARTIFACT_MANIFEST.tsv").open(newline="", encoding="utf-8") as stream:
        artifacts = list(csv.DictReader(stream, delimiter="\t"))
    with (ROOT / "SOURCE_PROVENANCE.tsv").open(newline="", encoding="utf-8") as stream:
        sources = list(csv.DictReader(stream, delimiter="\t"))
    for row in artifacts:
        target = ROOT / row["release_path"]
        if not target.is_file() or target.stat().st_size != int(row["bytes"]) or digest(target) != row["sha256"]:
            raise RuntimeError(f"Artifact missing or mismatched: {row['release_path']}")
        if args.list:
            print(f"{row['role']}\t{row['release_path']}")
    for row in sources:
        target = ROOT / row["release_path"]
        if not target.is_file() or digest(target) != row["source_sha256"]:
            raise RuntimeError(f"Source missing or mismatched: {row['release_path']}")

    llava_path = ROOT / "configs/policies/llava_frozen.json"
    qwen_path = ROOT / "configs/policies/qwen_transfer.json"
    llava = json.loads(llava_path.read_text(encoding="utf-8"))
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    for name, policy in (("LLaVA", llava), ("Qwen", qwen)):
        if policy["threshold"] != THRESHOLD or policy["top_k"] != 8:
            raise RuntimeError(f"{name} frozen threshold/top-k mismatch")
        if policy["runtime_detector"] != "google/owlv2-base-patch16-ensemble":
            raise RuntimeError(f"{name} detector mismatch")
    if qwen["source_policy_path"] != "configs/policies/llava_frozen.json":
        raise RuntimeError("Qwen source-policy path is not portable")
    if qwen["source_policy_sha256"] != digest(llava_path):
        raise RuntimeError("Qwen source-policy checksum mismatch")
    if (ROOT / "models").exists() or (ROOT / "data").exists():
        raise RuntimeError("Model or image data directory present in release")
    print(f"PASS: {len(sources)} source files; {len(artifacts)} frozen artifacts; policies and hashes verified")


if __name__ == "__main__":
    main()
