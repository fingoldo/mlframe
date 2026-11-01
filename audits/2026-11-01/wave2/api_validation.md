# Wave 2 audit: public API input validation (API3)

Read-only pass over `src/mlframe`. No file was edited and no git state was touched. Scope was sampled, not exhaustive (see "Not covered").

Vocabulary: RESOLVED (already safe, citation), FUTURE (real, deferred, reason), DOC (intended, to document), REJECTED (not a risk, reason).

Earlier rounds were read first (`audits/2026-10-31/_TRACKER.md`, `audits/2026-11-01/_TRACKER.md`). Already-recorded items are not re-reported: DESER-01/02/04/05, DSR2-01 (now fixed in `core/helpers.py`), DSR2-02, INP2-1, INP2-4, NET2-01, SEC2-1 and RDS2-1 are cross-referenced where relevant.

## Entry points sampled

| Entry point | Location | Input from a caller |
|---|---|---|
| `read_trained_models` | `src/mlframe/inference/predict.py:135` | `featureset` (path segment), `trusted_root`, `allowed_extensions`, `X` |
| `_load_features_file` | `predict.py:90` | features sidecar path (derived from `featureset`) |
| `load_mlframe_model` | `src/mlframe/training/io.py:547` | `file`, `safe`, `strict_version` |
| `validate_trusted_path` | `src/mlframe/core/helpers.py:~18-51` | path string plus trusted root |
| `get_models_raw_predictions` | `predict.py:263` | `trained_models`, `X`, `category_domains` |
| `_check_model_feature_order` | `predict.py:60` | model feature names against the sidecar list |

Not located in this pass: `predict_mlframe_models_suite` and `train_mlframe_models_suite` (no `def` matched in the first 40 hits), and any HTTP/RPC server module (none surfaced in the grep).

## Findings

### API3-1 (low): inference errors echo server-side path and caller column names
- Location: `src/mlframe/inference/predict.py:192-196` (no-features ValueError names `fpath`/`featureset`) and `predict.py:204-209` (missing-columns ValueError lists up to 10 names from the features sidecar).
- Entry point: `read_trained_models`.
- Value reaching the sink: `featureset` is joined into `fpath` (relative, `inference_folder/featureset`). The message text reaches whatever layer relays a `ValueError` to a remote client.
- Assessment: the echoed path is relative and the names come from the trained contract, not from request data beyond the caller's own columns. Disclosure is minor.
- Proposed disposition: **DOC**. A service wrapper that returns exception text to clients should map `ValueError` to a generic 4xx and log the detail.

### API3-2 (info): inference logs carry exception text and full path
- Location: `predict.py:186` (`exc_info=True`), `predict.py:247` (`%s` of `e`), `predict.py:119`.
- Entry point: `read_trained_models`.
- Assessment: these go to the logger, not to the return value. No path to the requester.
- Proposed disposition: **REJECTED** (logged, not returned; no request-visible leak).

### API3-3 (low): `load_mlframe_model` returns a shared cached object that callers can mutate
- Location: `src/mlframe/training/io.py:568-587`. The in-code NOTE (lines 584-586) says the cached object is shared and that `clean_mlframe_model()` mutates it in place, poisoning later cache hits for the same `(path, mtime, safe)`.
- Entry point: `load_mlframe_model`, the repeat-load pattern the comment names for a long-lived inference service.
- Value reaching the sink: no request value. The hazard is a caller-side mutation that persists across requests in one process. A multi-tenant service that loads one bundle for two requests can see one request's mutation in the other.
- Proposed disposition: **FUTURE**. Returning a copy, or an opt-in `copy=True`, costs memory on exactly the large bundles the cache exists for. A decision on the default is needed first, so this is not a unilateral change. Until then, a docstring line stating the no-mutation contract is the minimum (see also DOC for API3-3 below).

### API3-4 (info): backslash normalisation makes validated and opened path strings differ on POSIX
- Location: `src/mlframe/core/helpers.py` (`validate_trusted_path`, the `path.replace("\\", "/")` before `realpath`), against `predict.py:166` where the opened path is `join(inference_folder, featureset)` unchanged.
- Entry point: `read_trained_models` with `featureset` containing `\`.
- Assessment: on POSIX, `..\..\x` is validated as `../../x` and rejected. The open uses the literal name, which would be inside the root anyway. The validator is stricter than the open, so there is no bypass. A legitimate POSIX filename containing `\` is refused, which is a usability cost, not a security one.
- Proposed disposition: **REJECTED** as a security risk; the refusal of such names is intended.

### API3-5 (info): `allowed_extensions` and `featureset` on Windows
- Location: `predict.py:164` lowercases the extension allowlist; `predict.py:221` lowercases file extensions. Consistent, so case-variant extensions cannot slip past.
- Proposed disposition: **REJECTED** (no bypass).

## Positive controls confirmed (RESOLVED)

- `predict.py:166-176`: containment check runs always. With no `trusted_root`, the effective root is `inference_folder`, so `featureset="../.."` or an absolute path is refused. Citation: `predict.py:167-176`.
- `core/helpers.py` `validate_trusted_path`: `trusted_root=None` raises (fail-closed); both sides use `realpath`, closing the symlink escape (DSR2-01, commit `1c3e12f11`).
- `predict.py:182-188` and `predict.py:190-197`: a missing features sidecar is refused unless `allow_inferred_features=True` (default False). Without it, a model is never fed an unchecked column contract.
- `predict.py:60-` `_check_model_feature_order`: a model without fitted feature names is refused unless `allow_unvalidatable_models=True` (default False).
- `predict.py:204-210`: a features file naming a column absent from `X` raises a clear error and never indexes `X` with an unchecked list.
- `predict.py:218-227`: model files filtered by an extension allowlist, and each checked by `_verify_sidecar` before load. Integrity only, not authenticity; this is already recorded as DESER-02/04 and is not re-reported.
- `io.py:568-597`: `load_mlframe_model` cache key includes `mtime_ns`, so an overwrite invalidates. The cache size cap env var is parsed with a fallback to the default on a bad value (`io.py:573-577`).
- `io.py:612-616`: decompression is bounded by `BoundedReader(_zraw, max_decompressed_bytes())`, so a crafted compressed bundle cannot expand without limit.

## Counts

| Severity | Count |
|---|---|
| High | 0 |
| Medium | 0 |
| Low | 2 (API3-1, API3-3) |
| Info | 3 (API3-2, API3-4, API3-5) |

Proposed dispositions: DOC 1 (API3-1), FUTURE 1 (API3-3), REJECTED 3 (API3-2, API3-4, API3-5), RESOLVED 7 positive controls listed above.

## Not covered

- `predict_mlframe_models_suite`, `train_mlframe_models_suite` and their argument parsing (not located in the sampled grep).
- Feature-selection `fit` methods (RFECV `fit` at `feature_selection/wrappers/rfecv/_fit.py:76`, Boruta `fit` at `boruta_shap/_fit_explain.py:506`): row/column size caps and `fit_params` handling were not read. INP2-1 and INP2-2 already cover two allocation cases.
- Serialization round-trips beyond `load_mlframe_model` and `read_trained_models` (`estimators/pipelines.py`, `_metadata_loader.py`).
- Any HTTP or RPC server module. None was found in the grep; a web-layer caller was therefore not modelled.
- Concurrency between `_verify_sidecar` and the subsequent open (TOCTOU), which belongs to the concurrency class.
- Dynamic per-column size from request frames (no row-count cap on `get_models_raw_predictions`); INP2-4 already records this as intended.
