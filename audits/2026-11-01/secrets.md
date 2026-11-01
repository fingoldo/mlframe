# Security audit, class: SECRETS AND CREDENTIALS (second pass)

Date: 2026-11-01. Scope: `src/mlframe` (full-tree regex sweep), `tests`, `.github`, config files. Read-only. Prior pass: `audits/2026-10-31/secrets.md` (SEC-01..SEC-05). No prior item is re-reported here.

## Summary

Counts: High 0, Medium 0, Low 1, Info 2.

No new live credential was found. The full-tree sweep for quoted literal assignments to `api_key`/`password`/`secret`/`token`/`bearer` in `*.py`, `*.yml`, `*.yaml`, `*.json`, `*.toml`, `*.cfg`, `*.ini`, `*.env` returned one hit, a fake test fixture. The prior pass recommended this follow-up, and it is now done for that pattern set.

## Findings table

| ID | file:line | Severity | Proposed status |
|---|---|---|---|
| SEC2-1 | `src/mlframe/training/feature_handling/providers.py:235-240` | Low | FUTURE (cross-ref SEC-04) |
| SEC2-2 | `src/mlframe/training/feature_handling/hf_provider.py:222-225` | Info | DOC |
| SEC2-3 | `tests/integrations/test_mlflow_scrubbed_giveup_log.py:15` | Info | REJECTED |

## Details

### SEC2-1 Userinfo in a non-secret param value is hashed into the cache signature (Low)

Kind: credential in a URL-valued param, the same class as SEC-04, seen from the cache side.

Exposure: `EmbeddingProvider.signature` builds `params_repr` from `_scrub_dict(self.params)` (line 235). `_scrub_dict` masks by key name only. A value such as `"https://user:pass@host"` under a non-secret key passes through unchanged and is fed to `blake2b(digest_size=8)` (line 239). The digest is written into the cache key `f"{kind}:{model}:{hash}"` (line 240). The 8-byte digest is not a clear-text leak. It is still an offline-guessable oracle for a low-entropy password, and it can persist in on-disk cache filenames or artifact metadata. <!-- pragma: allowlist secret -->

Failure scenario: a user sets `params={"base_url": "https://u:hunter2@host"}`. The cache key then carries an unsalted 8-byte hash of a string that contains the password. A dictionary attack against a leaked cache directory is then possible. <!-- pragma: allowlist secret -->

Fix: apply the same userinfo masking proposed for SEC-04 inside `_scrub_value` before hashing. Then the signature hashes the masked form. Add a test that two providers differing only in the password produce the same signature, and that the signature does not change when the password changes.

Status FUTURE. It is a sub-case of SEC-04 and should be fixed together with it.

### SEC2-2 Model identifier is logged at INFO on every acquire (Info)

Kind: possible credential in a model string, low likelihood.

Exposure: `hf_provider.py:222-225` logs `model_name` and `revision` at INFO. `model_name` comes from `cfg.model`, which is normally a Hugging Face repo id. The resolved `params` (after `resolve_secrets`, line 202) are not logged. Only `model`, `revision`, `device`, and `dtype` are printed.

Failure scenario: a user passes a full URL with userinfo as `model=` (not a supported form, since `from_pretrained` expects a repo id). That value is written to INFO logs verbatim.

Status DOC: the logged fields are safe for supported inputs. Document in the `EmbeddingProvider` docstring that `model` must be a repo id, not a URL, and must not embed credentials.

### SEC2-3 Fake credential in test fixture (Info)

`tests/integrations/test_mlflow_scrubbed_giveup_log.py:15` defines `SECRET = "supersecret"` with a `# pragma: allowlist secret` marker. It is a fixture used to check that the SEC-01 fix keeps the value out of logs. It is not a real credential.

Status REJECTED: not a risk.

## Controls verified in this pass (RESOLVED evidence)

- `providers.py:246-263` `resolve_secrets`: the `KeyError` for an unset `env:` variable names only the variable, not its value.
- `providers.py:255-260` resolved values live only in the copy returned by `model_copy`, not in `self.params`, so the stored config keeps the `env:VAR` reference.
- `hf_provider.py:202-207`: `trust_remote_code` is opt-in and warned. No token is read from a literal in source.
- Log sweep: a regex over `logger.*`/`print(` calls that contain `api_key`, `token`, `secret`, `password`, or `credential` found only `logger.debug` lines whose matches are the word "token" in tokenizer or sentinel contexts (`_training_loop_fallback_helpers.py:568`, `_mrmr_fingerprints.py:274`, `cb/_cb_text_probe.py:281`, `_classification_report.py:243`). These are NLP or sentinel tokens. REJECTED as credential leaks.
- Environment-variable sweep (`os.environ`/`getenv`): all matches are feature flags or paths (`MLFRAME_*`, `HF_HOME`, `CUDA_HOME`, `PYUTILZ_KERNEL_CACHE_DIR`). None reads a credential.

## Coverage note

The regex sweep covered quoted literal values only. It does not detect unquoted values, base64 blobs, or private-key bodies. The earlier key-material check (`BEGIN ... PRIVATE KEY`, `AKIA`, `ghp_`, `xox`, `AIza`, `sk-`) was not rerun in this pass and should be rerun on `tests` and `*.yml`/`*.json`/`*.env` as a CI-side check.
