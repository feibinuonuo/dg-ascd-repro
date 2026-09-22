# LLaVA primary unified VCD baseline

## Purpose

This baseline supplies a same-backbone, same-image, same-scorer horizontal reference. It is not the parent-controlled estimand of the paper: DG-ASCD is a local extension of Fixed ASCD, so the primary causal comparison remains DG-ASCD minus Fixed ASCD.

## Cohort derivation and comparability

- Source: immutable LLaVA-1.5-7B VCD run with 500 answers, greedy decoding, seed 42, `cd_alpha=1`, `cd_beta=0.1`, `noise_step=999`, and `max_new_tokens=512`.
- Derived cohort: zero-based rows `[200,500)`, exactly 300 answers.
- Frozen-primary reference: `llava-detector-grounded-fixed-n300-start200-seed42-derived-v2.answers.jsonl`.
- Image-ID audit: all 300 IDs and their order match the frozen LLaVA primary cohort exactly.
- Evaluation: the same local COCO/CHAIR annotations and `experiments_v3.eval.chair_utils` scorer used for the Fixed/DG primary results.

## Absolute results

| Method | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Length |
|---|---:|---:|---:|---:|
| VCD | 54.00 | 15.10 | 82.21 | 101.2 |
| Fixed ASCD | 30.30 | 8.23 | 69.26 | 101.0 |
| DG-ASCD | 25.00 | 6.16 | 68.53 | 102.3 |

## Paired effects versus VCD

Effects are candidate minus VCD in percentage points, with 2,000 image-paired bootstrap replicates and seed 42.

| Comparison | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|
| Fixed ASCD − VCD | −23.67 [−30.00, −17.66] | −6.87 [−8.87, −4.89] | −12.95 [−15.62, −10.33] |
| DG-ASCD − VCD | −29.00 [−35.00, −23.33] | −8.94 [−10.77, −7.01] | −13.68 [−16.35, −11.01] |

Interpretation: Fixed ASCD and DG-ASCD have markedly lower object-hallucination rates than VCD in this unified chain, while VCD retains higher object Recall. The result therefore reinforces the hallucination–coverage trade-off; it is not a universal quality ranking.

## Artifact hashes (SHA-256)

- Source n=500 answers: `e8ea6232f063c37f79904c76cac840ee5e66da5e49c1e46561da8f9124ce5848`
- Derived n=300 answers: `a8cc0569f3a0d012b03b2de566a997c7d494a1b0d653333eaeb09609ef2967ec`
- Derivation audit: `f58f527c0e6750bddc97924804dd630125f3ea861cb78dd71d7c8cbf4103ad5d`
- CHAIR score: `c0695b487a27fad29b120cdfd9da6ab08ea1c72b7aed27ea9d2a8b957710b5fa`
- Fixed-minus-VCD paired comparison: `930c216eec0478a52b022c40925160e66caef1624d6d6e064da5248f6e4e6ed9`
- DG-minus-VCD paired comparison: `bddeda377419bcb9128606bba0a07fe931ae2c8226bdda119c4d8aa35b10e9ae`

Qwen VCD was not added in this round because the available VCD path is LLaVA-specific; porting it would constitute a new implementation rather than a direct reuse of the unified baseline.
