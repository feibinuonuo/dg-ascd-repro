# Private-to-public release checklist

Current state: the local `main` branch is staged, but there is no initial commit, GitHub remote, or public URL. These are deliberate pending steps; the manuscript must not cite this repository as public yet.

1. Rotate any password that has been shared in a chat, enable GitHub two-factor authentication, and use an authorized GitHub connection or browser/device-code login. Never paste passwords or personal access tokens into the chat or repository.
2. Confirm the desired Git commit identity and whether the public commit should use a GitHub-provided no-reply email address. The staged files have not been committed with an assumed author identity.
3. Create `feibinuonuo/dg-ascd-repro` as a **private** repository, without initializing another README/license. Push the staged local `main` branch after an attributed commit. Verify that `ARTIFACT_MANIFEST.tsv` and its 78 files are present in the private remote.
4. Before public release, inspect copied LLaVA/LAVIS/CHAIR/VCD notices, review historical absolute paths retained in the hash-preserved artifacts, and decide whether their disclosure is acceptable. Do not rewrite artifacts silently: any redaction requires a new public hash manifest and a provenance note.
5. At submission, change visibility to public, verify access in a logged-out browser, preferably archive a release with a persistent DOI, and update the manuscript's Data/Code Availability statements with the real public URL and release identifier.
6. Keep model checkpoints, benchmark images, local outputs, and credentials outside Git. The `.gitignore` excludes root `models/`, root `data/`, local `outputs/`, and common secret files.

Offline checks already completed: artifact/source hash verification; shell and Python syntax; five detector-grounding unit tests; AMBER adapter and Qwen wiring tests; POPE scoring self-test; and import smoke tests for LLaVA, Qwen, AMBER, POPE, and VCD entry points. Full GPU generation from a fresh clone remains unverified because it requires third-party checkpoints/datasets and a CUDA environment.
