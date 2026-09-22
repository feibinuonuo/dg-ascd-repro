# DG-ASCD post-freeze sensitivity results

The grid and reporting rule were declared in `sensitivity_protocol_2026-09-10.md` before inspecting new outputs. All runs use the frozen LLaVA primary rows 200--499 (`n=300`), Fixed ASCD parent, detector checkpoint and prompt, greedy decoding, seed 42, and the same local COCO/CHAIR scorer. The calibration-selected center (`tau=0.14235107600688934`, `k=8`) remains the main policy.

## Output metrics

| Axis | Setting | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Length |
|---|---:|---:|---:|---:|---:|
| threshold (`k=8`) | 0.10 | 24.67 | 6.60 | 68.00 | 102.7 |
| threshold (`k=8`) | 0.12 | 24.33 | 6.01 | 68.11 | 102.8 |
| threshold (`k=8`) | **0.142351 (frozen)** | **25.00** | **6.16** | **68.53** | **102.3** |
| threshold (`k=8`) | 0.16 | 25.33 | 6.12 | 67.79 | 102.9 |
| threshold (`k=8`) | 0.18 | 26.00 | 6.30 | 67.58 | 104.0 |
| top-k (`tau` frozen) | 4 | 25.00 | 6.16 | 68.53 | 102.3 |
| top-k (`tau` frozen) | **8 (frozen)** | **25.00** | **6.16** | **68.53** | **102.3** |
| top-k (`tau` frozen) | 16 | 25.00 | 6.16 | 68.53 | 102.3 |

## Image-paired effects versus Fixed ASCD

Differences are sensitivity candidate minus Fixed ASCD in percentage points. Confidence intervals use 2,000 image-paired bootstrap replicates with seed 42.

| `tau` | ΔCHAIRs [95% CI] | ΔCHAIRi [95% CI] | ΔRecall [95% CI] |
|---:|---:|---:|---:|
| 0.10 | -5.67 [-8.67, -2.99] | -1.62 [-2.64, -0.70] | -1.26 [-2.34, -0.22] |
| 0.12 | -6.00 [-9.00, -3.00] | -2.21 [-3.33, -1.20] | -1.16 [-2.29, 0.00] |
| **0.142351** | **-5.33 [-8.67, -2.33]** | **-2.07 [-3.20, -1.04]** | **-0.74 [-2.02, +0.57]** |
| 0.16 | -5.00 [-8.33, -1.67] | -2.11 [-3.32, -1.03] | -1.47 [-2.97, +0.10] |
| 0.18 | -4.33 [-8.01, -0.67] | -1.93 [-3.24, -0.72] | -1.68 [-3.49, +0.10] |

Every threshold retains a paired CHAIRs and CHAIRi reduction whose 95% CI is below zero. Recall remains lower than Fixed ASCD at every point, although its interval crosses zero at the frozen center and the two higher thresholds. This is a robustness analysis, not evidence for replacing the frozen center.

## Action audit

| Axis | Setting | object candidates | masks | selection changes | finite protections | images with mask |
|---|---:|---:|---:|---:|---:|---:|
| threshold | 0.10 | 4,444 | 597 | 236 | 132 | 150 |
| threshold | 0.12 | 4,396 | 640 | 259 | 143 | 157 |
| threshold | **0.142351** | **4,391** | **846** | **314** | **189** | **172** |
| threshold | 0.16 | 4,502 | 1,073 | 378 | 254 | 187 |
| threshold | 0.18 | 4,655 | 1,427 | 504 | 332 | 208 |
| top-k | 4 | 3,873 | 664 | 304 | 189 | 156 |
| top-k | **8** | **4,391** | **846** | **314** | **189** | **172** |
| top-k | 16 | 4,549 | 913 | 310 | 189 | 178 |

Increasing `tau` monotonically increases masks, selection changes, protections, and the number of images with a mask. Corpus-level metrics are not strictly monotone because an early token change alters the downstream autoregressive candidate trajectory. The `k=4`, `k=8`, and `k=16` answer files are byte-identical (SHA-256 `389f1ca7205047c950164163f3422730077b200c28f32ea898ad7913e3257cff`) despite different audit counts; extra lower-ranked actions never changed the final caption on this cohort.

## Run validity

The first schema-invalid launches and two initialization OOM attempts produced zero valid samples and are retained as `invalid-startup` in the artifact manifest. Valid low-threshold runs use the `-v3` suffix; valid higher-threshold and top-k runs use `-v2`. All eight reported rows contain 300 answers and a detector audit, and no failed startup contributes to scoring.
