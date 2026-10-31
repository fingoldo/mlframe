# Network, HTTP servers and external endpoints audit (mlframe, src/mlframe)

Read-only audit. No code or git state changed.

## Summary

- No network listener in `src/mlframe`. No Flask, FastAPI, Dash, uvicorn, `http.server`, `socketserver`, or raw `socket.bind/listen` code was found. There are no `api*.py` modules and no `@app.route` or `add_api_route` handlers. Nothing binds to `0.0.0.0` or any other address.
- No direct HTTP client calls (`requests`, `httpx`, `aiohttp`, `urlopen`) were found. All outbound traffic goes through third-party libraries: HuggingFace (`transformers` `from_pretrained`), MLflow (`mlflow.*`), `gdown` (documented as a manual step only), and `sklearn.datasets.fetch_*` (benchmark code only).
- Outbound data flow: MLflow metrics and artifacts go to whatever tracking server the user configures. Nothing else sends telemetry.
- Not verified in full: the InfoNet checkpoint load path in `_neural_mi.py` (around lines 297 onward) was not read to the end. The `sys.path` injection there is unrated.

Counts by severity: High 0, Medium 0, Low 4, Info 5.

## Findings table

| ID | File:line | Severity | Proposed status |
|---|---|---|---|
| NET-01 | `src/mlframe/integrations/mlflow.py:71-96` | Low | DOC |
| NET-02 | `src/mlframe/feature_selection/filters/_neural_mi.py:284-291` | Low | FUTURE |
| NET-03 | `src/mlframe/feature_selection/filters/_neural_mi.py:443-477` | Info | RESOLVED |
| NET-04 | `src/mlframe/training/feature_handling/hf_provider.py:66-88, 204-256` | Info | RESOLVED |
| NET-05 | `src/mlframe/training/feature_handling/hf_provider.py:74-88` | Low | DOC |
| NET-06 | `src/mlframe/training/feature_handling/hf_provider.py:236-256`, `src/mlframe/feature_selection/filters/_neural_mi.py:473` | Low | FUTURE |
| NET-07 | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py:136-146` (and round3/round4/_downstream_shared bench files) | Info | REJECTED |
| NET-08 | `src/mlframe/integrations/mlflow.py:34-41, 150-177` | Info | RESOLVED |
| NET-09 | `src/mlframe/reporting/renderers/_plotly_interactivity.py:24` | Info | REJECTED |

## Details

### NET-01: MLflow HTML embed (Low, DOC)
`embed_website_to_mlflow` (`integrations/mlflow.py:71-96`) writes an `<iframe src=...>` into an MLflow artifact. The URL is HTML-escaped (`html.escape`, line 83), which blocks attribute breakout. The iframe sandbox is `allow-same-origin allow-scripts` (line 88). Together those two flags let the framed page script its own origin, so the sandbox offers little protection. `fname` is used as a raw filesystem path (line 95), so `../` or an absolute path writes outside the intended directory. The docstring (lines 74-80) already says both inputs must be internally controlled.
Exposure: only if a caller passes untrusted input. No listener is involved.
Disposition DOC: the contract is already in the docstring. Optional follow-up: drop `allow-same-origin` from the sandbox, or reject `fname` values that contain a path separator or `..`.

### NET-02: InfoNet checkpoint has no integrity check (Low, FUTURE)
The InfoNet checkpoint `infonet_pretrained.pt` comes from a Google Drive folder. The code only tells the user to fetch it with `gdown` (`_neural_mi.py:284-291`). No SHA-256 check is made against the file, unlike the sha256-sidecar scheme used elsewhere in mlframe (`utils/safe_pickle.py`, `inference/predict.py`). If the load path uses `torch.load` without `weights_only=True`, a tampered file would run code. That part was not verified (see the note above).
Fix: pin a SHA-256 for the checkpoint and verify it before loading. Require `weights_only=True` on the load.
Disposition FUTURE: real but only reachable if an attacker can write to the user's cache directory or swap the Drive file. Deferred because the load path needs a full read first.

### NET-03: MIST model pinned to a commit SHA (Info, RESOLVED)
`MIST_PINNED_REVISIONS` (`_neural_mi.py:444-447`) pins both `grgera/MIST` and `grgera/MIST-QR` to commit SHAs. `_get_mist_hf_model` passes `revision=` at line 473, so a moved or compromised hub repo cannot change the model silently.

### NET-04: HuggingFace loads are hardened (Info, RESOLVED)
- `PINNED_REVISIONS` (`hf_provider.py:67-71`) pins the three default models to commit SHAs.
- `trust_remote_code` defaults to `False` (`hf_provider.py:207`). A caller must opt in explicitly (lines 204-214), and a warning is logged when they do.
- The revision is passed to every `from_pretrained` call (lines 237-256, 432-436).

### NET-05: Unlisted HF model loads hub `main` (Low, DOC)
`resolve_revision` (`hf_provider.py:74-88`) returns `None` for an unpinned model. The hub then serves whatever `main` is at that moment. The code emits a `UserWarning` (lines 83-87) telling the caller to pass `params={'revision': <sha>}`. The behaviour is intended and visible.
Disposition DOC: documented in the warning and docstring.

### NET-06: No timeouts on model and dataset downloads (Low, FUTURE)
`from_pretrained` calls (`hf_provider.py:236-256, 431-436`, `_neural_mi.py:473`) pass no timeout. Neither does `transformers`, so a stalled connection can hang the fit. The rest of mlframe uses timeouts where it can (for example `FileLock(..., timeout=30)` in `utils/_param_oracle_store.py:93`). The HF stack can be bounded through the `HF_HUB_DOWNLOAD_TIMEOUT` environment variable.
Fix: document the environment variable, or set it in the HF provider before the first load.
Disposition FUTURE: availability issue only, no data leak. Deferred.

### NET-07: Benchmark dataset downloads (Info, REJECTED)
`fetch_openml` and the `fetch_*` helpers in `feature_selection/_benchmarks/` (for example `fs_hybrid/_realdata_cache.py:140-146`, `round3_realdata_bench.py:36-47`, `round4_broad_realdata_bench.py:133`, `_downstream_shared.py:70`) download public datasets with no checksum or timeout. These are developer benchmark scripts, not runtime code, and they run only when invoked by hand.
Disposition REJECTED: not in any production path, and the data is public.

### NET-08: MLflow credentials scrubbed from errors (Info, RESOLVED)
`_strip_userinfo` (`integrations/mlflow.py:34-41`) redacts `scheme://user:pass@` before logging. `get_or_create_mlflow_run` (lines 150-177) logs and re-raises the scrubbed message rather than the raw exception. The DSL filter values are escaped (lines 104-112), so a run name cannot break out of the `search_runs` filter.  <!-- pragma: allowlist secret -->

### NET-09: Plotly download button (Info, REJECTED)
`_plotly_interactivity.py:24` keeps the modebar `download` button. It saves the chart image in the browser, with no request to an external service.
Disposition REJECTED: a client-side action, not a data transmission.

## Scope note

- Timeouts: `grep` for `timeout=` found only local lock and subprocess timeouts (`_param_oracle_store.py:93`, `kernel_tuning_cache/_ensure.py:164`). No network call sets a timeout.
- TLS: no `verify=False`, `CERT_NONE`, or `check_hostname=False` anywhere in `src/mlframe`.
- Telemetry: no `wandb`, `comet`, `neptune`, `clearml`, or `sentry` imports were found in the source.
