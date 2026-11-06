# Wave 4: network and servers (mlframe, src/mlframe)

Read-only. Prior dispositions NET-01..NET-09 (audits/2026-10-31) re-checked on HEAD; none found wrong. No new listener, HTTP client, TLS bypass or unpinned download in src/mlframe outside `_benchmarks/`.

## Findings

| ID | Severity | File:line | Failure scenario | Proposed disposition |
|---|---|---|---|---|
| NET4-01 | Info | `src/mlframe/integrations/mlflow.py:93-96` vs `:106` | Docstring says the url "is embedded verbatim ... with no sanitization", but line 106 applies `html.escape(url, quote=True)`. A reader may wrongly add a second escape (double-encoding `&` in query strings) or wrongly assume attribute breakout is possible. | DOC: correct the docstring to say the url is HTML-escaped but its scheme is not checked. |
| NET4-02 | Low | `src/mlframe/integrations/mlflow.py:106,111` | `html.escape` blocks attribute breakout but not a `javascript:` or `data:` scheme. With `sandbox='allow-same-origin allow-scripts'` a `javascript:` src runs script in the MLflow UI artifact origin when a caller passes an untrusted url. Extends NET-01 (DOC) with the scheme gap it did not mention. | FUTURE: allow only `http`/`https` schemes (raise otherwise) and drop `allow-same-origin`; contract already says internal urls only, so not RESOLVED-needed. |

Counts: High 0, Medium 0, Low 1, Info 1.

## Checked and safe

- No `requests`, `httpx`, `aiohttp`, `urllib`, `urlopen`, `socket`, `http.server`, flask, fastapi, uvicorn, `0.0.0.0`, `start_http_server` in src/mlframe (non-benchmark).
- No `verify=False` / `CERT_NONE`.
- `torch.load` sites all use `weights_only=True`: `feature_selection/filters/_vendored/infonet/infer.py:51`, `training/neural/_flat_torch_module/__init__.py:600`, `training/neural/_recurrent_wrapper_base.py:581`, `training/_unpickle_policy.py:145`. InfoNet checkpoint gets a sidecar digest (`_neural_mi.py:282`), NET-02 RESOLVED holds.
- HF loads pinned by revision: `_neural_mi.py:493`, `hf_provider.py:259-275, 454`; downloads serialised by `PIDAwareFileLock(timeout=600.0)` (`hf_provider.py:257`). Network timeout still absent, as dispositioned DOC in NET-06.
- `embed_website_to_mlflow` refuses `..` in fname (`mlflow.py:103-104`) and writes via temp+rename.
- MLflow tracking URI is never set by mlframe (no `set_tracking_uri`); user-configured.
