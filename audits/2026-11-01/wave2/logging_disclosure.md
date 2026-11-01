# Wave 2 audit: information disclosure through logs and exceptions

Scope: `src/mlframe` (benchmarks excluded), read-only. Method: grep for logger/print/log_throttle calls carrying
config, params, kwargs, tokens, row/frame names, paths, URLs; `exc_info=True` and traceback capture; `repr(exc)`
and `repr(...)` in messages; telemetry/tracking/HTTP client calls. Already recorded and not repeated: SEC-04,
SEC2-1, SEC2-2, SEC2-3, the MLflow scrub (NET-01, NET2-02, NET2-06, RDS2-1, SEC-01).

Limits: this is a sampled grep pass, not an exhaustive read of every log call. No remote telemetry, error-reporting
or HTTP-posting client was found in `src/mlframe` (grep for requests/httpx/urllib, sentry, wandb, mlflow param/tag
writes returned only unrelated hits), so the remote-service class is empty here outside the MLflow integration.

## Findings

| ID | Severity | Location | Sensitive value and source | Who can read | Proposed disposition |
|---|---|---|---|---|---|
| LOG3-1 | low | `src/mlframe/reporting/_async_render.py:505-510` and `:517` | `_record_internal_failure` and `_fail_now` store `"".join(traceback.format_exception(exc))` in `self._summary.failures` and log it at WARNING with `exc_info=True`. Exception messages raised by pandas, numpy or polars during rendering can embed offending row or cell values, so a traceback can carry data values from a user frame. | Anyone with the process log, and any caller that reads `summary.failures` | DOC: the traceback is the intended diagnostic for a failed render, and the data it could carry is the caller's own data in the caller's own process. Document that failure records can contain exception text. |
| LOG3-2 | low | `src/mlframe/calibration/post.py:592-598` (`failed_calibrators[calibrator_name] = repr(exc)`, log at `:592`) | `repr(exc)` of a user-supplied calibrator exception is returned to the caller in the result dict and logged at WARNING. The message is whatever the user's calibrator raised. Same class as LOG3-1, but the value is returned, not only logged. | Process log readers and any holder of the returned result dict | DOC: the field is documented as explicitly surfaced so callers can see which candidates failed (see docstring at `:707`). Callers that persist the result should scrub it. |
| LOG3-3 | info | `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_fit_helpers.py:425-431` | `print()` writes the selected feature names, their `mrmr_gain` values and the engineered feature names to stdout after every MRMR fit. No row values are printed. | Anyone who can read the stdout of the process or its job log | DOC: printed by design as the selection summary; feature names are not secret for this library. |
| LOG3-4 | info | `src/mlframe/training/io.py:539` | Logs `bundle_path` at WARNING when an unverified bundle loads under `UNVERIFIED_ENV_VAR`. Local filesystem path only. | Process log readers | REJECTED: a local path is not a secret, and the warning is the intended signal that the sha256 gate was bypassed. |
| LOG3-5 | info | `src/mlframe/data/datasets/replay.py:98` | Logs `scenario` name and two spec hashes at WARNING on a replay mismatch. Hashes are content identifiers of a synthetic-data spec, not credentials. | Process log readers | REJECTED: no sensitive value; the hashes are the diagnostic. |
| LOG3-6 | info | `src/mlframe/calibration/_post_train_calibrators.py:304-308` | `repr(exc)` plus `exc_info=True`, at DEBUG, when the row-index hash for the overlap check raises `TypeError`. | Only readers of DEBUG logs | REJECTED: DEBUG level only, and a `TypeError` from hashing an index does not normally embed row values. |

## Counts

- High: 0
- Medium: 0
- Low: 2 (LOG3-1, LOG3-2)
- Info: 4 (LOG3-3 to LOG3-6)
- Total: 6

## Notes on what was checked and found clean

- Config and parameter logging: `training/_trainer_configure.py`, `neural/_flat_torch_module/__init__.py:512-539`
  (scheduler config, hyperparameters only), `training/pipeline/_pipeline_fit_transform.py:168` (categorical column
  names). No credential-bearing config object is logged.
- Frame and row logging: the `logger.info` calls in `training/preprocessing.py` log row and column counts, not values.
- `_extractors_showcase.py` prints dtype and info summaries from its own demo; it is not library code on a hot path.
- The HF model string is already tracked as SEC2-2.
