#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: run_amber.sh fixed|dg}"
case "$MODE" in fixed|dg) ;; *) echo "Mode must be fixed or dg" >&2; exit 2 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${AMBER_IMAGES:?Set AMBER_IMAGES to the official generative image directory}"
: "${AMBER_QUERY:?Set AMBER_QUERY to query_generative.json}"
: "${LLAVA_MODEL:?Set LLAVA_MODEL to the local LLaVA checkpoint}"
test -d "$AMBER_IMAGES"; test -s "$AMBER_QUERY"
MANIFEST="artifacts/reproduction_notes/amber_hard_detector_ascd_20260826/manifests/amber_generative_full_n1004.json"
test -s "$MANIFEST"
mkdir -p outputs
OUT="outputs/amber-${MODE}-n1004.responses.json"
AUDIT="outputs/amber-${MODE}-n1004.detector.json"
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
DETECTOR_ARGS=()
if [[ "$MODE" == dg ]]; then
  : "${OWL_MODEL:?Set OWL_MODEL to the local OWLv2 checkpoint}"
  test ! -e "$AUDIT" || { echo "Refusing to overwrite $AUDIT" >&2; exit 2; }
  DETECTOR_ARGS=(--detector-grounded-audit-file "$AUDIT" --detector-grounded-policy-file configs/policies/llava_frozen.json --detector-grounded-model-path "$OWL_MODEL")
fi
ASCD_DISABLE_CUDNN=1 "${PYTHON_BIN:-python}" -m experiments_v3.eval.model_vqa_amber_detector \
  --model-path "$LLAVA_MODEL" --image-folder "$AMBER_IMAGES" --query-file "$AMBER_QUERY" \
  --manifest-file "$MANIFEST" --answers-file "$OUT" --conv-mode llava_v1 \
  --question-limit 1004 --max-new-tokens 512 --greedy-decoding --contrastive-layer-ids all \
  --direct-steer-config experiments_v3/assets/ascd_pos0625/direct_steer_config.yaml \
  --contrastive-config experiments_v3/assets/ascd_pos0625/contrastive_config.yaml \
  --contrastive-decoding-config experiments_v3/assets/ascd_pos0625/contrastive_decoding.yaml \
  "${DETECTOR_ARGS[@]}"
test -s "$OUT"
test "$MODE" != dg || test -s "$AUDIT"
printf 'RESULT_FILE %s\n' "$OUT"
