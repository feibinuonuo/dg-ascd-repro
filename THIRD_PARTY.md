# Provenance and third-party materials

- The base ASCD repository is distributed under Apache-2.0; its license text is included as `LICENSE`. The DG-ASCD integration extends this code. Cite the published ASCD paper separately.
- `llava/` is inherited from the ASCD/LLaVA integration. Review original LLaVA attribution and license before public release; this package does not assert authorship of LLaVA components.
- `lavis/` contains Salesforce LAVIS code with BSD-3-Clause notices in source files. Its license text is included in `licenses/LAVIS-BSD-3-Clause.txt`. The repository's Apache-2.0 file does not supersede in-file third-party notices.
- `experiments_v3/eval/chair_utils.py` identifies its provenance as the CHAIR evaluation implementation and includes a CHAIR synonym ontology. The scorer is included for reproducibility, not claimed as original DG-ASCD work.
- `experiments_vcd/` and the VCD results represent a reproduced implementation/reference baseline. Cite VCD and preserve source notices.
- The head maps are small frozen ASCD configuration artifacts; model checkpoint weights are excluded. OWLv2, LLaVA, Qwen, COCO, and AMBER must be obtained from their providers under their respective terms.

Before public release, the repository owner should complete a final third-party license check for copied LLaVA, LAVIS, CHAIR, and VCD files. This is a provenance checklist, not legal advice.
