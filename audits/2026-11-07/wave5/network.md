# Wave 5: network and servers (mlframe, src/mlframe)

Read-only. Prior rounds re-checked on HEAD: NET-01..NET-09 (audits/2026-10-31) and NET4-01..NET4-02 (audits/2026-11-06/wave4).
NET4-02 is now fixed on HEAD: `integrations/mlflow.py:106-108` refuses any url scheme other than http/https.
No new listener, HTTP client, TLS bypass or unpinned download was found in src/mlframe outside `_benchmarks/`.

## Findings

| ID | Severity | File:line | Failure scenario | Proposed disposition |
|---|---|---|---|---|
| NET5-01 | Info | `src/mlframe/integrations/mlflow.py:95-97` vs `:106-110` | NET4-01 is still open and now more wrong. The docstring says the url "is embedded verbatim ... with no sanitization", but the code rejects non-http(s) schemes (`:107-108`) and applies `html.escape` (`:110`). A reader may add a second escape (double-encoding `&` in query strings) or remove the scheme check, believing it is redundant. | DOC (RESOLVED-needed as a docstring fix): say the url must be http/https, is HTML-escaped, and that the iframe keeps `allow-same-origin allow-scripts`, so only trusted http(s) origins should be passed. |

Counts: High 0, Medium 0, Low 0, Info 1.

## Checked and safe

- No `requests`, `httpx`, `aiohttp`, `urllib.request`/`urlopen`, `socket`, `http.server`, flask, fastapi, uvicorn, `0.0.0.0`, `start_http_server`, `hf_hub_download`, `snapshot_download` or `set_tracking_uri` in src/mlframe outside `_benchmarks/`. `webbrowser` appears only in the unpickle deny-list (`training/io.py:209`). `gdown` appears only in a user-facing message (`_neural_mi.py:307`).
- No `verify=False` or `CERT_NONE`.
- `torch.load` always uses `weights_only=True`: `_vendored/infonet/infer.py:51`, `neural/_flat_torch_module/__init__.py:600`, `neural/_recurrent_wrapper_base.py:581`, `training/_unpickle_policy.py:145`.
- HuggingFace loads are pinned by revision: `_neural_mi.py:493`, `hf_provider.py:259-275, 454`. The missing network timeout is still covered by NET-06 (FUTURE).
- `embed_website_to_mlflow` refuses a `..` path component in `fname` (`mlflow.py:104-105`) and an unsafe scheme (`:107-108`).
- MLflow run-search filters escape run names (`mlflow.py:141-147`). Credentials are scrubbed from URIs before logging (`mlflow.py:37-45`).
