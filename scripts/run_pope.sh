#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: run_pope.sh fixed|dg}"
case "$MODE" in fixed|dg) ;; *) echo "Mode must be fixed or dg" >&2; exit 2 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
: "${POPE_QUESTIONS:?Set POPE_QUESTIONS to the official llava_pope_test.jsonl}"
: "${POPE_IMAGES:?Set POPE_IMAGES to the COCO image directory used by POPE}"
: "${LLAVA_MODEL:?Set LLAVA_MODEL to the local LLaVA checkpoint}"
test -s "$POPE_QUESTIONS"; test -d "$POPE_IMAGES"
mkdir -p outputs
OUT="outputs/pope-${MODE}-n8910.answers.jsonl"
AUDIT="outputs/pope-${MODE}-n8910.detector.json"
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
DETECTOR_ARGS=()
if [[ "$MODE" == dg ]]; then
  : "${OWL_MODEL:?Set OWL_MODEL to the local OWLv2 checkpoint}"
  test ! -e "$AUDIT" || { echo "Refusing to overwrite $AUDIT" >&2; exit 2; }
  DETECTOR_ARGS=(--detector-grounded-audit-file "$AUDIT" --detector-grounded-policy-file configs/policies/llava_frozen.json --detector-grounded-model-path "$OWL_MODEL")
fi
ASCD_DISABLE_CUDNN=1 "${PYTHON_BIN:-python}" -m experiments_v3.eval.model_vqa_pope_detector \
  --model-path "$LLAVA_MODEL" --image-folder "$POPE_IMAGES" \
  --question-file "$POPE_QUESTIONS" --answers-file "$OUT" --conv-mode llava_v1 \
  --question-limit 8910 --max-new-tokens 64 --greedy-decoding \
  --contrastive-layer-ids all \
  --direct-steer-config experiments_v3/assets/ascd_pos0625/direct_steer_config.yaml \
  --contrastive-config experiments_v3/assets/ascd_pos0625/contrastive_config.yaml \
  --contrastive-decoding-config experiments_v3/assets/ascd_pos0625/contrastive_decoding.yaml \
  "${DETECTOR_ARGS[@]}"
test "$(wc -l < "$OUT")" -eq 8910
test "$MODE" != dg || test -s "$AUDIT"
printf 'RESULT_FILE %s\n' "$OUT"
