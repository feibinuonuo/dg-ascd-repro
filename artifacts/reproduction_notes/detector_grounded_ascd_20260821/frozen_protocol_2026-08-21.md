# Detector-Grounded ASCD: frozen calibration protocol

Date frozen: 2026-08-21

## Question

Can an independent, text-conditioned object detector provide a sufficiently
reliable *object-local* image-support signal to constrain ASCD, without using
COCO annotations at inference time?

This is a new method-search branch.  It does not retune, extend, or rescue the
whole-caption CLIP or crop-CLIP verifier branches, both of which are No-Go.

## Verifier and score

- Detector: `google/owlv2-base-patch16-ensemble`, loaded from a locally saved
  official checkpoint, in evaluation mode, FP16, with no fine-tuning.
- For each unique `(image, canonical COCO object)` proposed in the already
  generated Fixed-ASCD observe-only trace, query OWLv2 with exactly
  `a photo of a {object}`.
- Image support is the maximum sigmoid detection logit over OWLv2's predicted
  boxes for that query.  No boxes, masks, captions, or COCO annotations are
  input to the detector at inference time.
- Canonical COCO object mapping is exactly the CHAIR synonym mapping.  Repeated
  proposals of the same image/object pair are deduplicated.

## Independent calibration and labels

Use only the already completed v2 profiling-disjoint Fixed-ASCD run, manifest
positions 100--199, n=100.  Its corresponding CHAIR output supplies present /
absent labels solely for calibration analysis; it is never passed into OWLv2
and the seed-42 evaluation split remains untouched.

- Records 0--49: feasibility half.
- Records 50--99: threshold-selection half.
- The exact input audit and CHAIR label SHA-256 values, detector file hashes,
  and all scored pairs must be recorded.

## Pre-registered go/no-go rule

The detector branch can proceed only when **both** halves contain at least 15
present and 15 absent pairs and each has ROC-AUC >= 0.700.  This deliberately
exceeds the 0.600 crop-CLIP feasibility boundary because a detector-specific
score is being considered for an actual decoding constraint.

If and only if the rule passes, freeze one global detector threshold on the
selection half by maximizing balanced accuracy; ties take the highest threshold
(the conservative choice).  A candidate object is detector-supported iff its
score is greater than or equal to that frozen value.

## Conditional decoder and experimental ladder

Only after a calibration pass will `Detector-Grounded ASCD` be implemented.
It will retain ASCD as the sole decoder distribution: within an ASCD top-k
candidate set, only completed CHAIR-object candidates with detector support
below the frozen threshold may be masked.  No Vanilla branch and no
ASCD/Vanilla fallback are allowed, preventing the branch-mixing drift observed
in the earlier verifier experiment.  Non-object candidates retain their ASCD
logits exactly.

Before any effectiveness result: formula unit tests -> `py_compile` -> full
historical regression -> Fixed default-off n=2 byte equivalence -> detector
n=2 smoke.  The n=20 gate compares a newly generated matched Fixed-ASCD parent
with the candidate on identical data/seed/generation budget.  It passes only
if Recall drops no more than 2 pp and either CHAIRs or CHAIRi improves.  Only a
pass may proceed to n=100, then (if still favorable) n=500.
