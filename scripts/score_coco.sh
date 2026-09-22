#!/usr/bin/env bash
set -euo pipefail
FIXED="${1:?usage: score_coco.sh FIXED_ANSWERS DG_ANSWERS TAG}"
DG="${2:?DG answers required}"
TAG="${3:?tag required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${COCO_ROOT:?Set COCO_ROOT to a directory containing annotations/}"
test -s "$FIXED"; test -s "$DG"; test -d "$COCO_ROOT/annotations"
mkdir -p outputs/scores
FIXED_SCORE="outputs/scores/$TAG.fixed.chair.json"
DG_SCORE="outputs/scores/$TAG.dg.chair.json"
PAIRED="outputs/scores/$TAG.dg-vs-fixed.comparison.json"
for file in "$FIXED_SCORE" "$DG_SCORE" "$PAIRED"; do
  test ! -e "$file" || { echo "Refusing to overwrite $file" >&2; exit 2; }
done
"${PYTHON_BIN:-python}" -m experiments_v3.eval.chair_utils --cap_file "$FIXED" --image_id_key image_id --caption_key caption --coco_path "$COCO_ROOT/annotations" --cache outputs/chair.pkl --save_path "$FIXED_SCORE"
"${PYTHON_BIN:-python}" -m experiments_v3.eval.chair_utils --cap_file "$DG" --image_id_key image_id --caption_key caption --coco_path "$COCO_ROOT/annotations" --cache outputs/chair.pkl --save_path "$DG_SCORE"
"${PYTHON_BIN:-python}" -m experiments_v3.eval.compare_diagnostic_runs --baseline-chair "$FIXED_SCORE" --candidate-chair "$DG_SCORE" --output "$PAIRED" --bootstrap-repetitions 2000 --seed 42
printf 'RESULT_FILE %s\n' "$FIXED_SCORE" "$DG_SCORE" "$PAIRED"
