# Detector-Grounded ASCD: reproducibility package

This repository accompanies *Detector-Grounded Contrastive Decoding for Object Hallucination Mitigation in Multimodal Large Language Models*. It releases the candidate-local DG-ASCD implementation, the frozen LLaVA/Qwen policies, split manifests, generated outputs, scoring results, and action audits. The main estimand is DG-ASCD minus its unchanged Fixed-ASCD parent on the same images. The LLaVA VCD run is a same-setting horizontal reference, not a parent-controlled intervention.

**Public repository:** <https://github.com/feibinuonuo/dg-ascd-repro>. Releases and Git commit identifiers provide versioned snapshots of the code and study-specific artifacts.

## What is here

| Location | Contents |
| --- | --- |
| `ascd_detector_grounded.py` | Canonical CHAIR object parsing, OWLv2 image-score cache, top-*k* mask, finite-candidate protection |
| `ascd_utils_v3/`, `ascd_utils_v3_qwen/` | Frozen ASCD decoder integrations |
| `experiments_v3/eval/`, `experiments_vcd/eval/` | CHAIR, AMBER, POPE, and VCD runners/scorers |
| `experiments_v3/assets/`, `text_centric_heads/` | Exact parent decoder YAML files and two small head maps |
| `configs/policies/` | Portable copies of the frozen LLaVA and Qwen detector policies |
| `artifacts/` | Hashed historical outputs, split manifests, scores, audits, sensitivity and efficiency records |
| `ARTIFACT_MANIFEST.tsv` | Original-to-release path and SHA-256 mapping for released results |
| `SOURCE_PROVENANCE.tsv` | SHA-256 mapping for copied source files |

The historical policy JSONs remain byte-for-byte in `artifacts/`. The two `configs/policies/` copies change **only file-path fields and the resulting source-policy hash**, not detector, threshold, *k*, candidate rule, or ASCD settings. This lets the Qwen loader validate the LLaVA source policy after a fresh clone. Run all commands from the repository root.

The package does **not** contain model weights, detector weights, COCO or AMBER images, private credentials, or conda environments. The 78 released result artifacts total about 55 MiB. See `THIRD_PARTY.md` before redistribution.

## Environment and external inputs

Python 3.10 and a CUDA-capable GPU are needed for generation. The historical environment used PyTorch 2.1.2, LLaVA-side `transformers==4.39.3`, and Qwen-side `transformers==4.51.3`; `pyproject.toml` records these two mutually exclusive extras. Use separate environments:

```bash
conda create -n dg-ascd-llava python=3.10 -y
conda activate dg-ascd-llava
pip install -e '.[llava_series]'
conda create -n dg-ascd-qwen python=3.10 -y
conda activate dg-ascd-qwen
pip install -e '.[qwen]'
```

Download or obtain separately: LLaVA-1.5-7B, Qwen2.5-VL-7B-Instruct at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, OWLv2 `google/owlv2-base-patch16-ensemble`, MS COCO 2014 validation images and CHAIR/COCO annotations, and the official AMBER generative images/query file. The OWLv2 `model.safetensors` used for calibration has SHA-256 `e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7`. The model loaders require **local** checkpoint directories; set them using the variables in the commands below. Download sources and dataset terms are the original providers' responsibility, not this repository's license.

The common COCO prompt is `Please describe this image in detail.`; greedy decoding and the parent YAML files are fixed. The LLaVA detector policy was selected using calibration images 100–199; the primary cohort uses run indices 200–499. Qwen uses the same threshold without Qwen-specific detector tuning. The primary and fresh split manifests are under `artifacts/reproduction_notes/` and their exact names are indexed by `ARTIFACT_MANIFEST.tsv`.

## Verify released evidence without a GPU

```bash
python scripts/verify_release.py
python -m compileall -q ascd_detector_grounded.py ascd_utils_v3 ascd_utils_v3_qwen experiments_v3/eval experiments_vcd/eval
```

The verifier checks every artifact hash, the two frozen policy settings, the portable Qwen source-policy hash, and the absence of model/image files. `python scripts/verify_release.py --list` prints the role-to-file mapping. Frozen raw outputs and precomputed scores can be inspected without rerunning 7B models.

## Re-run the main COCO/CHAIR comparison

The scripts refuse to overwrite existing outputs and write only into `outputs/` (ignored by Git). Paths are external to the repository. Activate the appropriate environment first.

```bash
export COCO_ROOT=/absolute/path/to/COCO-CHAIR
export LLAVA_MODEL=/absolute/path/to/llava-v1.5-7b
export QWEN_MODEL=/absolute/path/to/Qwen2.5-VL-7B-Instruct-cc594898
export OWL_MODEL=/absolute/path/to/owlv2-base-patch16-ensemble
export CUDA_VISIBLE_DEVICES=0

bash scripts/run_coco.sh llava fixed primary
bash scripts/run_coco.sh llava dg primary
bash scripts/score_coco.sh outputs/llava-primary-fixed.answers.jsonl outputs/llava-primary-dg.answers.jsonl llava-primary

# In the separate Qwen environment:
bash scripts/run_coco.sh qwen fixed primary
bash scripts/run_coco.sh qwen dg primary
bash scripts/score_coco.sh outputs/qwen-primary-fixed.answers.jsonl outputs/qwen-primary-dg.answers.jsonl qwen-primary
bash scripts/run_coco.sh qwen fixed fresh
bash scripts/run_coco.sh qwen dg fresh
```

`COCO_ROOT` must contain `annotations/` and `val2014/`. The split manifests store image IDs/order, not images. For the Qwen fresh pair, pass the two fresh output files to `score_coco.sh` in the same way. The paired scorer uses image-level resampling with 2,000 bootstrap replicates and seed 42; see the manuscript's Online Resource 1 for the metric definitions and sensitivity CIs.

## Re-run AMBER

```bash
export AMBER_IMAGES=/absolute/path/to/AMBER/images/image
export AMBER_QUERY=/absolute/path/to/AMBER/query_generative.json
export LLAVA_MODEL=/absolute/path/to/llava-v1.5-7b
export OWL_MODEL=/absolute/path/to/owlv2-base-patch16-ensemble
bash scripts/run_amber.sh fixed
bash scripts/run_amber.sh dg
```

The AMBER full-generative manifest is released under `artifacts/reproduction_notes/amber_hard_detector_ascd_20260826/manifests/`. The historical Vanilla, Fixed, DG, and paired score outputs are included in `ARTIFACT_MANIFEST.tsv`. Official AMBER annotations/scoring resources must be obtained from the benchmark provider.

The AMBER paired scorer is `reproduction_notes/amber_hard_detector_ascd_20260826/score_amber_paired_cached_vectorized.py`. It requires the official `annotations`, `association`, and `safe_words` files in addition to the three response files; its `--help` lists all path arguments. The reported bootstrap uses the frozen script defaults (10,000 replicates, seed 20260826). The transition and figure-case extraction scripts are in the same directory.

## POPE action-coverage check

```bash
export POPE_QUESTIONS=/absolute/path/to/llava_pope_test.jsonl
export POPE_IMAGES=/absolute/path/to/COCO/val2014
bash scripts/run_pope.sh fixed
bash scripts/run_pope.sh dg
```

`reproduction_notes/hard_detector_pope_evaluation_20260824/score_pope_paired.py` performs the paired POPE scoring using separately obtained official label-question and annotation files. The released answer and audit artifacts document the zero-action result.

## Baseline, sensitivity, and scope

The VCD n=300 answers were derived by exact run-index slicing from a frozen n=500 greedy run (`cd_alpha=1`, `cd_beta=0.1`, `noise_step=999`, max 512 new tokens); the source, derived answers, CHAIR score, and image-ID audit are in the manifest. No VCD hyperparameter was selected on the 300-image reporting subset. Sensitivity policies/results and POPE zero-action outputs are likewise indexed. POPE's short yes/no outputs did not produce complete canonical-object candidates, so DG-ASCD did not act in that setting.

The main limitations are the CHAIR object ontology, detector false veto of true objects, and task-specific action coverage. Reproducing generation requires the original third-party checkpoints and annotations; hash verification alone does not reproduce the GPU experiments.

## Provenance

This project extends the Apache-2.0 ASCD implementation and carries LLaVA/LAVIS-derived components under their respective notices. The modified CHAIR scorer traces to the CHAIR benchmark implementation. See `THIRD_PARTY.md` and in-file notices. Do not use this repository as a substitute for citing ASCD, CHAIR, OWLv2, VCD, and the benchmark papers.
