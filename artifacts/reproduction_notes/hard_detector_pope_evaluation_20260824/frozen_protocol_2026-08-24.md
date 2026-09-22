# Frozen evaluation-only protocol: Hard Detector-Grounded ASCD on POPE (2026-08-24)

## Purpose and status

This is an **external evaluation of the already frozen positive method**, not
a new optimization branch and not a retry of VEP, Soft-Grounded, CRC, or OIAG.
It tests whether the exact fixed Hard Detector-Grounded ASCD policy preserves
its object-hallucination direction on the project's official POPE data.

The method is frozen before reading any new POPE Detector result:

- parent: Fixed ASCD `ascd_pos0625`;
- policy: `../detector_grounded_ascd_20260821/calibration/detector-owlv2-fixed-ascd-topk8-policy-v1.json`;
- OWLv2: local `google/owlv2-base-patch16-ensemble`, hash-checked by the
  frozen policy;
- action: unchanged finite-top-8 hard masking of unsupported canonical CHAIR
  object candidates; no prompt, threshold, alpha, top-k, or Qwen setting is
  changed;
- decoding: LLaVA-1.5-7B, greedy, `max_new_tokens=64`, the project POPE
  convention.

## Dataset and arms

Use the complete immutable `data/llava_dataset/pope/llava_pope_test.jsonl`
(8,910 questions) and all three official strata: `random`, `popular`, and
`adversarial`, as evaluated by the existing `experiments_v3/eval/eval_pope.py`.
The labels are not used during decoding, implementation, or policy choice.

Run independently named answer files for:

1. Vanilla LLaVA;
2. Fixed ASCD; and
3. the frozen Hard Detector-Grounded ASCD.

No historical answer file is overwritten or silently reused as a baseline.

## Engineering ladder

1. New standalone POPE adapter is default-off: without its detector arguments
   it must use the same LLaVA/Fixed-ASCD generation path as the existing
   `model_vqa_loader` evaluator.
2. `py_compile`; existing Detector unit tests; parser and image-id unit tests.
3. Fixed n=2: the new adapter and the preserved base POPE evaluator must have
   the same question ids and decoded texts (UUIDs are non-deterministic and
   excluded from equality).
4. Hard Detector n=2: two answer records plus two audit records; policy/hash
   validation, finite-candidate protection, and image-cache consistency must
   pass.
5. A deterministic 30-question stratified smoke (10 per POPE stratum) records
   all three arms.  It is only an engineering check; its metrics choose no
   setting and do not gate the full evaluation.
6. Only then run all 8,910 questions for the three frozen arms, write separate
   detector audit, score all three POPE strata, and calculate paired
   Fixed-versus-Detector outcome differences with nonparametric question-level
   bootstrap CIs.

## Reporting and interpretation

Primary POPE metrics are accuracy, precision, recall, F1, and yes ratio per
stratum and pooled.  The paired comparison reports answer transitions and
bootstrap CIs for Detector minus Fixed.  This benchmark does not replace the
COCO-caption Recall analysis: its yes/no recall has a different definition and
must never be presented as a recovery of the CHAIR object Recall cost.

A positive POPE change broadens evidence for the fixed method's object
hallucination behavior.  A null or adverse result bounds generalization.  In
neither case may the method be retuned on POPE.
