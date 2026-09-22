# DG-ASCD post-freeze sensitivity protocol

Declared before inspecting any new sensitivity outputs on 2026-09-10.

## Evaluation cohort

- Backbone: LLaVA-1.5-7B.
- Cohort: the frozen primary rows 200--499 from `evaluation_seed42_n500.json` (`n=300`).
- Decoding: the same Fixed ASCD parent, direct-steer/contrastive configs, greedy decoding, seed 42, and `max_new_tokens=512` used in the main LLaVA primary comparison.
- Detector: the same local `google/owlv2-base-patch16-ensemble` checkpoint, prompt, score definition, canonical-object parser, and finite-candidate protection.
- Scoring: the same COCO/CHAIR annotations and local scorer.

## One-factor grids

1. Threshold sensitivity: `tau ∈ {0.10, 0.12, 0.14235107600688934, 0.16, 0.18}` with `k=8` fixed. The center is the frozen calibration-selected policy; four neighboring points are post-freeze analyses and cannot replace the main setting.
2. Candidate-set sensitivity: `k ∈ {4, 8, 16}` with `tau=0.14235107600688934` fixed. The center is the frozen main policy.

## Reporting rule

- Report every grid point, regardless of direction.
- Report CHAIRs, CHAIRi, Recall, and caption length.
- Interpret smoothness/trade-off patterns; do not select a new operating point.
- Main claims and confidence intervals remain based on the pre-existing frozen `tau=0.1424, k=8` policy.

## Failed startup attempts

The first six launches used explanatory `method/status` strings that were rejected by the runner's strict frozen-policy schema before any sample was generated. After restoring the exact required schema and model-file hash field, runs were restarted with `-v2` suffixes. Two `-v2` low-threshold launches (`tau=0.10,0.12`) then encountered CUDA out-of-memory during model/detector initialization because other processes occupied physical GPU0/1; they also generated zero samples and will be rerun with new suffixes on released GPUs. These startup failures do not alter the grid or results.
