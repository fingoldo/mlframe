# Wave 2: SUBPROCESS and SECRETS audit (src/mlframe)

Scope: `src/mlframe` only, read-only. Searched for `subprocess`, `os.system`, `os.popen`, `Popen`, `shell=True`,
token/api_key/password/secret/credential names, `HF_TOKEN`/`use_auth_token`, MLflow tracking URIs, `os.environ` writes, and URLs with userinfo.

## Result: no reachable leak or injection found

### Subprocess sites read (production code)
| Site | Verdict |
|---|---|
| `system/kernel_tuning_cache/_ensure.py:199` | argv list: `sys.executable`, fixed module, registered kernel name; no shell, timeout set. Safe. |
| `training/_gpu_state_probe.py:98,109` | fixed `nvidia-smi` argv, no shell, timeout, output parsed only. Safe. |
| `training/_polars_native_support.py:90` | fixed source string run by this interpreter, no shell. Safe. |
| `utils/native_import_probe.py:55` | fixed executable and argument list. Safe. |
| `training/pipeline/_pipeline_extensions_pysr.py:246` | only catches `CalledProcessError`; no call. Safe. |
| `_benchmarks/*` (several) | fixed `git show <rev>:<path>` argv lists, dev-only scripts. Safe. |

No `shell=True`, `os.system` or `os.popen` anywhere in `src/mlframe`. No secret is passed in argv.

### Secret handling read
- `integrations/mlflow.py:40-57`: `_strip_userinfo` / `_scrubbed_exc_info` redact `scheme://<userinfo>@` from exception text and drop the unscrubbed cause chain before logging. Correct.
- `training/feature_handling/providers.py`: `EmbeddingProvider` resolves `env:VAR` secrets lazily, masks secret-named keys recursively in `__repr__`, `__str__`, `model_dump` and `signature`, and scrubs URL userinfo in non-secret string values. The `resolve_secrets` KeyError names only the env var, not a value. Correct.
- No HuggingFace token, MLflow password or API key literal or logging site found; `os.environ` writes are all thread/device knobs.

## Findings

### SEC-W2-1 (Info) `integrations/mlflow.py:91` embed_website_to_mlflow
- Statement: `url` is HTML-escaped but its scheme is not restricted, so a `javascript:` URL would still be placed in the iframe `src`.
- Scenario: only reachable if a caller passes an untrusted URL, which the docstring explicitly forbids; iframe also carries a `sandbox` attribute that allows scripts with same-origin.
- Disposition: DOC (already documented as caller contract). Optional hardening: reject non-http(s) schemes.

### SEC-W2-2 (Info) `training/feature_handling/providers.py:275` model_dump(scrub_secrets=False)
- Statement: the opt-out returns resolved secrets in plain text by design.
- Scenario: a caller logs the unscrubbed dump; no such caller exists in `src/mlframe`.
- Disposition: DOC (intended handoff path; docstring already warns).
