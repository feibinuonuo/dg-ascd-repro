#!/usr/bin/env bash
set -euo pipefail

# Re-run one frozen COCO/CHAIR pair member from the repository root.
MODEL="${1:?usage: run_coco.sh llava|qwen fixed|dg primary|fresh}"
MODE="${2:?mode required}"
COHORT="${3:?cohort required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

case "$MODEL:$COHORT" in
  llava:primary|qwen:primary)
    MANIFEST="artifacts/reproduction_notes/qwen25vl_second_backbone_20260812/splits/evaluation_seed42_n500.json"
    START=200; N=300 ;;
  qwen:fresh)
    MANIFEST="artifacts/reproduction_notes/detector_grounded_ascd_qwen_20260821/splits/fresh_qwen_holdout_seed20260821_n300.json"
    START=0; N=300 ;;
  *) echo "Unsupported model/cohort: $MODEL/$COHORT" >&2; exit 2 ;;
esac
case "$MODE" in fixed|dg) ;; *) echo "Mode must be fixed or dg" >&2; exit 2 ;; esac

: "${COCO_ROOT:?Set COCO_ROOT to a directory containing annotations/ and val2014/}"
test -d "$COCO_ROOT/annotations"
test -d "$COCO_ROOT/val2014"
test -s "$MANIFEST"

OUT="outputs/${MODEL}-${COHORT}-${MODE}.answers.jsonl"
AUDIT="outputs/${MODEL}-${COHORT}-${MODE}.detector.json"
mkdir -p outputs
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
test "$MODE" != dg || test ! -e "$AUDIT" || { echo "Refusing to overwrite $AUDIT" >&2; exit 2; }

DETECTOR_ARGS=()
if [[ "$MODEL" == llava ]]; then
  : "${LLAVA_MODEL:?Set LLAVA_MODEL to the local LLaVA checkpoint}"
  MODEL_PATH="$LLAVA_MODEL"
  MODULE=experiments_v3.eval.model_vqa_chair
  CONFIG=ascd_pos0625
  MAX_NEW_TOKENS=512
  export ASCD_DISABLE_CUDNN=1
  if [[ "$MODE" == dg ]]; then
    : "${OWL_MODEL:?Set OWL_MODEL to the local OWLv2 checkpoint}"
    DETECTOR_ARGS=(--detector-grounded-audit-file "$AUDIT" --detector-grounded-policy-file configs/policies/llava_frozen.json --detector-grounded-model-path "$OWL_MODEL")
  fi
else
  : "${QWEN_MODEL:?Set QWEN_MODEL to the local Qwen checkpoint}"
  MODEL_PATH="$QWEN_MODEL"
  MODULE=experiments_v3.eval.model_vqa_chair-qwen
  CONFIG=qwen25vl_cc594898_local_top64_fixed
  MAX_NEW_TOKENS=128
  if [[ "$MODE" == dg ]]; then
    : "${OWL_MODEL:?Set OWL_MODEL to the local OWLv2 checkpoint}"
    DETECTOR_ARGS=(--detector-grounded-audit-file "$AUDIT" --detector-grounded-policy-file configs/policies/qwen_transfer.json --detector-grounded-model-path "$OWL_MODEL")
  fi
fi

"${PYTHON_BIN:-python}" -m "$MODULE" \
  --model-path "$MODEL_PATH" \
  --annotation-folder "$COCO_ROOT/annotations" \
  --image-folder "$COCO_ROOT/val2014" \
  --image-manifest "$MANIFEST" --manifest-start-index "$START" \
  --answers-file "$OUT" --max_new_tokens "$MAX_NEW_TOKENS" \
  --sample_num "$N" --sample_seed 42 --num-chunks 1 --chunk-idx 0 \
  --contrastive_layer_ids all --greedy_decoding \
  --direct_steer_config "experiments_v3/assets/$CONFIG/direct_steer_config.yaml" \
  --contrastive_config "experiments_v3/assets/$CONFIG/contrastive_config.yaml" \
  --contrastive_decoding_config "experiments_v3/assets/$CONFIG/contrastive_decoding.yaml" \
  "${DETECTOR_ARGS[@]}"

test "$(wc -l < "$OUT")" -eq "$N"
test "$MODE" != dg || test -s "$AUDIT"
printf 'RESULT_FILE %s\n' "$OUT"
