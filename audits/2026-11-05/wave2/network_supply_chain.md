# Wave 2: network and supply chain audit (mlframe)

Scope: `src/mlframe`, `.github/workflows`. Grepped for requests/urllib/httpx/hf_hub_download/from_pretrained/snapshot_download/urlretrieve/verify=False/torch.hub/curl/wget, and for workflow `uses:` and `pip install` lines.

Result: no outbound raw HTTP clients in `src/mlframe` (urllib appears only as `urllib.parse` and in unpickle deny-lists). Network access is limited to Hugging Face hub loads, which are revision-pinned. All workflow actions are pinned to full commit SHAs. Two low-severity findings.

## NET-W2-1 (Low): HF models outside the pin table load hub `main`
- File: `src/mlframe/training/feature_handling/hf_provider.py:73-86` (used at :240, :259, :275)
- Statement: when a caller names a model that is not in `PINNED_REVISIONS` and passes no `revision`, `resolve_revision` returns None and transformers loads the mutable `main` branch. The only safeguard is a UserWarning.
- Failure scenario: the upstream repo is force-pushed or compromised, so embeddings change silently between runs and a cached FE recipe no longer reproduces. With `trust_remote_code=True` (opt-in, warned), the new code runs on load.
- Disposition: DOC. This behaviour is deliberate, warned about, and needed to support arbitrary models. Optionally add a strict mode that raises instead of warning.

## NET-W2-2 (Low): unpinned `pip install packaging` in matrix plan jobs
- File: `.github/workflows/pandas-matrix.yml:51,159`; `.github/workflows/polars-matrix.yml:53,163`
- Statement: `packaging` is installed with no version pin and no hash, then used by `scripts/ci_version_matrix.py` to pick the test legs.
- Failure scenario: a malicious or breaking `packaging` release runs inside CI with the job token, or it changes how the matrix legs are resolved. Only the plan step is exposed and the job has no publish secrets.
- Disposition: RESOLVED-needed. Pin it, e.g. `packaging==<ver>`, the same way coverage and pip-audit are already pinned.

## Checked and rejected
- MIST hub load, `feature_selection/filters/_neural_mi.py:493`: pinned via `MIST_PINNED_REVISIONS`. REJECTED: not a finding.
- `trust_remote_code` defaults to False and enabling it raises a warning (hf_provider.py:230). REJECTED.
- Workflow `uses:` entries are all pinned to 40-character SHAs, including the shared `fingoldo/py-ci-shared` action. REJECTED.
- `ci.yml:765` installs the locally built wheel `dist/*.whl` into a smoke venv. Dependencies resolve from PyPI by design, as a smoke test of the end-user install path. REJECTED.
- No `requests`/`httpx`/`urlretrieve` calls, so there are no missing-timeout or `verify=False` sites.
