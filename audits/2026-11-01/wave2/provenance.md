# Wave 2 audit: downloads and provenance (PRV3)

Read-only pass over `src/mlframe`. Baselines read first: `audits/2026-10-31/_TRACKER.md`, `audits/2026-11-01/_TRACKER.md`, `audits/2026-10-31/network.md`, `audits/2026-10-31/supply_chain.md`, `audits/2026-11-01/network.md`, `audits/2026-11-01/supply_chain.md`. Items recorded there (NET-01..09, NET2-01..06, SUP-01..15, SUP2-1..8, DESER-02, DESER-06) are not re-reported.

## Fetch points found

| Where | What is fetched | Verified before use | Source controlled by |
|---|---|---|---|
| `training/feature_handling/hf_provider.py:237, 256-277` | HF hub model and tokenizer (`AutoModel`, `AutoTokenizer`) | Revision pinned for the three shipped defaults (`PINNED_REVISIONS`, lines 67-71); unlisted models load hub `main` with a warning | Model owner on the hub; caller chooses model name |
| `feature_selection/filters/_neural_mi.py:461-495` | MIST (`grgera/MIST`, `grgera/MIST-QR`) via `MISTForHF.from_pretrained` | Pinned to commit SHAs (`MIST_PINNED_REVISIONS`, lines 462-465); no `trust_remote_code` argument | Hub repo owner, but pinned SHA limits drift |
| `feature_selection/filters/_neural_mi.py:277-318` | InfoNet checkpoint, fetched by hand from Google Drive; code only prints a `gdown` command in an error | sha256 trust-on-first-use sidecar (NET2-03, fixed `6cdf0b0c2`); `torch.load(weights_only=True)` in `_vendored/infonet/infer.py:51` | Google Drive folder owner; first download is trusted |
| `feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py:118-200` | OpenML datasets via `fetch_openml` into a local cache | Cache read verifies SHA-256 from sidecar JSON (lines 190-197) and loads with `allow_pickle=False`; fetch is gated to bench code | OpenML; the sidecar is written from whatever the first fetch returned (TOFU) |
| `feature_selection/boruta_shap/__init__.py:823-829` | `fetch_california_housing` fallback | None (sklearn-managed download) | sklearn data host; documented demo path (NET2-05) |
| `integrations/mlflow.py` | MLflow tracking server URI | Not a download of code | Operator config |

No `urllib`, `urlretrieve`, `requests`, `httpx`, `aiohttp`, `torch.hub`, `load_dataset`, or `hf_hub_download` call was found in `src/mlframe` outside `_benchmarks`. No runtime `pip`, `uv`, or `ensurepip` subprocess was found in library code. `verify=False` and `ssl` overrides were not found.

## New findings

### PRV3-1 (Low), `training/feature_handling/hf_provider.py:74-88` and `:237`
A caller-supplied `params={"revision": ...}` is returned as `str(explicit)` without checking that it is a 40-hex commit SHA. A branch name such as `main` or a tag is accepted silently, so the "pinned" intent is not enforced for explicit values. The warning at lines 83-87 covers only the case where no revision is given for an unlisted model.
- Impact: a config that writes `revision: main` loads a moving target with no warning.
- Proposed disposition: **DOC**. The caller chose the revision, and the docstring says so. A SHA-format check would be a behaviour change for callers who pass tags deliberately.

### PRV3-2 (Low), `training/feature_handling/hf_provider.py:113-121`
When neither `HF_HOME` nor `TRANSFORMERS_CACHE` is set, the cross-process lock directory falls back to `tempfile.gettempdir()/mlframe-hf-locks`. The path is predictable and shared on multi-user hosts. Another local user can pre-create it or its lock file to stall `acquire()` (timeout 600 s, line 254). The lock file holds no code and no data, and the failure mode is a stall, not a load of unverified bytes.
- Proposed disposition: **FUTURE**. Reason: a per-user directory under the home cache would close it. It needs a small change plus a test, and the impact is limited to availability.

### PRV3-3 (Info), `training/feature_handling/hf_provider.py:227-237` and `feature_selection/filters/_neural_mi.py:491`
Confirmed that the pinned revision is applied on the `trust_remote_code=True` path too (`hf_provider.py:256-276`), and that `MISTForHF.from_pretrained` is called with the pinned SHA and no `trust_remote_code` flag. The `trust_remote_code` opt-in path therefore still runs pinned repo code only. This is a positive check, not a defect.
- Proposed disposition: **RESOLVED** (already safe). Cite `hf_provider.py:256-276` and `_neural_mi.py:491`.

### PRV3-4 (Info), `feature_selection/filters/_neural_mi.py:302-308`
The "checkpoint not found" error prints a shell command with a `gdown.download_folder` call for the user to run by hand. Nothing is fetched by the library. The printed command pins the Drive folder ID, not a file hash. The hash is then recorded on first use (NET2-03).
- Proposed disposition: **DOC**. The manual step is intended and is covered by NET2-03.

### PRV3-5 (Info), `feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py:190-197`
The sha256 sidecar is written from the first `fetch_openml` result, so the first fetch is trusted and every later load is pinned to it. This matches the trust-on-first-use model already recorded for NET2-03.
- Proposed disposition: **DOC**. Reason: benchmark-only code; the cache is gitignored and local.

## Counts

| Severity | New |
|---|---|
| High | 0 |
| Medium | 0 |
| Low | 2 (PRV3-1, PRV3-2) |
| Info | 3 (PRV3-3, PRV3-4, PRV3-5) |

## Not verified

- Live HF hub and Drive behaviour (no network use in this audit).
- The `mist_statinf` package loader internals (third party).
- Whether `ensurepip` or a `pip` call exists in code paths not reached by the grep patterns used; the pattern search covered `pip`, `uv`, `ensurepip`, and `sys.executable` install calls.
