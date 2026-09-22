# Frozen efficiency protocol: Detector-Grounded ASCD on AMBER (2026-08-27)

## Purpose

Measure deployment cost of the already frozen Hard Detector-ASCD, not quality.
No AMBER outcome, caption, detector score, or timing result selects a model,
prompt, threshold, top-k, ASCD alpha/head map, candidate rule, or decoding
parameter.

## Fixed workload and environment

- Workload: immutable AMBER `amber_generative_n20_seed20260826.json`.
- Model/policy: LLaVA-1.5-7B, Fixed ASCD `ascd_pos0625`, frozen OWLv2 policy
  (`threshold=0.1423510760`, `top_k=8`), greedy decoding, `max_new_tokens=512`,
  and `ASCD_DISABLE_CUDNN=1`.
- Hardware: one idle GPU, checked before every run; all formal runs sequential
  on the same physical GPU.
- Timing: process wall time, progress-bar generation duration, decoded caption
  tokens/s, images/s, and 200-ms `nvidia-smi` peak-memory monitoring.

## Conditions

Each condition is run twice solely to expose timing variance; repetitions are
not independent model-effect samples.

1. **Fixed**: default-off AMBER adapter, without an OWLv2 runtime.
2. **Detector cold**: normal frozen Detector-ASCD.  The process loads OWLv2
   and computes each unique image's visual support vector once.
3. **Detector warm-score-cache**: process-local cache adapter uses the exact
   support vectors precomputed by frozen OWLv2 on this workload, then runs the
   unchanged Detector-ASCD decoder.  This simulates an already-populated
   image-score cache; it is not represented as the ordinary unique-image
   AMBER latency.

The score-cache precomputation separately records OWLv2 model-load time,
per-image cold visual inference time, GPU memory, and in-memory score-lookup
time.  Warm captions must be byte-identical to cold Detector captions before
their timing is reported.

## Reporting limits

Report cold end-to-end latency as the primary deployment cost for unique
images.  Report warm-cache latency separately, including that it excludes
OWLv2 model loading and visual forwards.  Any Fixed/Detector difference is
descriptive because decoded lengths can differ.  Do not infer statistical
method superiority from timing repetitions.
