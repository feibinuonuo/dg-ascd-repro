# Public-release checklist

Repository: <https://github.com/feibinuonuo/dg-ascd-repro>. The repository contains the implementation and 78 hash-verified result artifacts. Complete every item below before submitting or resubmitting a manuscript that cites the repository as public.

1. Keep GitHub two-factor authentication enabled and never place passwords, personal access tokens, or private keys in the repository.
2. Review copied LLaVA/LAVIS/CHAIR/VCD notices and the historical absolute paths retained in hash-preserved artifacts. Do not rewrite frozen artifacts silently: any redaction requires a new manifest and a provenance note.
3. Run `python scripts/verify_release.py` from the repository root. The expected result is `PASS: 780 source files; 78 frozen artifacts; policies and hashes verified`.
4. Push all documentation changes and record the resulting public commit hash with the submission records. A tagged release or commit-specific URL may also be cited when a permanently versioned snapshot is desired.
5. In GitHub, change repository visibility to **Public**. Verify the repository URL, README, code, `ARTIFACT_MANIFEST.tsv`, and `artifacts/` from a logged-out/incognito browser.
6. Create a tagged GitHub release for the resubmission snapshot if convenient. A release or archival DOI improves versioning, but the editor's immediate requirement is an openly accessible repository containing both code and study-specific data/artifacts.
7. Keep model checkpoints and third-party benchmark images outside Git. The `.gitignore` excludes root `models/`, root `data/`, local `outputs/`, and common secret files. The README provides acquisition instructions for third-party inputs that cannot be redistributed.

Offline checks already completed: artifact/source hash verification; shell and Python syntax; five detector-grounding unit tests; AMBER adapter and Qwen wiring tests; POPE scoring self-test; and import smoke tests for LLaVA, Qwen, AMBER, POPE, and VCD entry points. Full GPU generation from a fresh clone remains unverified because it requires third-party checkpoints/datasets and a CUDA environment.
