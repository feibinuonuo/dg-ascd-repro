"""Derive the frozen LLaVA primary cohort from the immutable VCD n=500 run.

The primary cohort is defined as rows 200--499 of the seed-42 sample order.
This script validates those image IDs against the already frozen Fixed-ASCD
primary answers before writing the lossless VCD subset.
"""

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()

    for path in (args.source, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, args.audit):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    source_rows = load_jsonl(args.source)
    reference_rows = load_jsonl(args.reference)
    subset = source_rows[args.start : args.start + args.count]
    if len(subset) != args.count or len(reference_rows) != args.count:
        raise ValueError("unexpected source or reference row count")

    source_ids = [int(row["image_id"]) for row in subset]
    reference_ids = [int(row["image_id"]) for row in reference_rows]
    if source_ids != reference_ids:
        raise ValueError("VCD subset does not match the frozen primary image order")
    if len(set(source_ids)) != args.count:
        raise ValueError("primary image IDs are not unique")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in subset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "schema_version": 1,
        "derivation": "lossless ordered slice from immutable VCD n=500 answers",
        "start_index_zero_based": args.start,
        "end_index_exclusive": args.start + args.count,
        "num_rows": args.count,
        "image_ids_match_frozen_fixed_primary": True,
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "first_image_id": source_ids[0],
        "last_image_id": source_ids[-1],
    }
    with args.audit.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"derived_vcd_primary_ok rows={args.count} output={args.output}")


if __name__ == "__main__":
    main()
