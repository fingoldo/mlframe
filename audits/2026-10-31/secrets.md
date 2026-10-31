# Security audit, class: SECRETS AND CREDENTIALS

Date: 2026-10-31. Scope: `src/mlframe`, `tests`, `.github`, `.pre-commit-config.yaml`, `.secrets.baseline`. Read-only; no code or git state changed.

## Summary

No hard-coded live credentials were found. No private keys, cloud keys (AKIA), GitHub/Slack/Google API tokens, or real connection strings with passwords exist in `src/` or `tests/`. The only embedded credential-shaped strings are a fake test fixture and CI `${{ github.token }}` references.

Real findings are all in the handling of credentials that users supply, not in committed secrets:

- SEC-01 (Medium): the MLflow `start_run` retry path logs an unscrubbed traceback, so a tracking-server password in the URL can reach log files.
- SEC-02 (Low): CI workflows embed the GitHub token in a git remote URL held in a shell variable.
- SEC-03 (Low): the secret-key heuristic misses common names (`apikey`, `pwd`, `passphrase`, `cookie`).
- SEC-04 (Low): params values that carry userinfo (`scheme://user:pass@host`) are not scrubbed; only key names are checked.  <!-- pragma: allowlist secret -->
- SEC-05 (Info): `EmbeddingProvider.model_dump(scrub_secrets=False)` is an intentional escape hatch with no in-tree caller that uses it.

Counts: High 0, Medium 1, Low 3, Info 1.

## Findings table

| ID | file:line | Severity | Proposed status |
|---|---|---|---|
| SEC-01 | `src/mlframe/integrations/mlflow.py:148` | Medium | FUTURE |
| SEC-02 | `.github/workflows/pandas-matrix.yml:163`, `.github/workflows/polars-matrix.yml:167` | Low | FUTURE |
| SEC-03 | `src/mlframe/training/feature_handling/providers.py:43-54` | Low | FUTURE |
| SEC-04 | `src/mlframe/training/feature_handling/providers.py:115-123`, `:126-150` (params handling) | Low | FUTURE |
| SEC-05 | `src/mlframe/training/feature_handling/providers.py:269-280` | Info | DOC |

## Details

### SEC-01 MLflow traceback leaks tracking-server userinfo into logs (Medium)

Kind: credential embedded in a tracking URI (`scheme://user:password@host`).  <!-- pragma: allowlist secret -->

Exposure: `mlflow.py:143-148`. The loop catches `Exception as e`. After more than five failed attempts it calls `log_throttle(..., logging.ERROR, "...", nfailed, exc_info=True)`. `exc_info=True` writes the full traceback, and the exception message in that traceback is the raw, unscrubbed `e`. The scrubbed value from `_strip_userinfo(e)` (line 150) is used only for the non-final log lines. The code comment at lines 160-162 shows the authors already know a bare `raise` leaks the credential. The same leak exists on the final-give-up log, which is the path a long outage hits.

Fix: on the final-give-up branch, log `_strip_userinfo(e)` as the message and format the traceback with the scrubbed text, or drop `exc_info=True` and log `traceback.format_exception` with `_strip_userinfo` applied to each line. Add a `caplog` test that makes `mlflow.start_run` raise `ConnectionError("...user:secret@host...")` six times and asserts the secret is absent from `caplog.text`.

Status FUTURE: real and small, but this is a read-only audit. The test in `tests/integrations/test_mlflow.py:237` covers only the single-attempt path and does not exercise the traceback path.

### SEC-02 CI embeds GitHub token in a git remote URL (Low)

Kind: `GITHUB_TOKEN` (ephemeral, `github.token`).

Exposure: `.github/workflows/pandas-matrix.yml:156-163` and `polars-matrix.yml:160-167` build `remote="https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git"` and pass it to `git ls-remote` and `git fetch`. If git prints an error that echoes the URL, the token would be in plain text. GitHub Actions masks the secret value in logs, so this is limited. The token is also short-lived and scoped to the run.

Fix: remove the token from the URL. Use `git -c http.extraheader="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"`, or set `GIT_CONFIG_*` env vars, so the credential never sits in a string that git may echo.

Status FUTURE. Low, because GitHub masks the value and the token is ephemeral.

### SEC-03 Secret-key heuristic misses common names (Low)

Kind: heuristic gap, not a leaked value.

Exposure: `providers.py:43-54` lists `key`, `token`, `secret`, `auth`, `authorization`, `bearer`, `credential(s)`, `password`, `passwd`. `_is_secret_field` (line 84-95) matches whole tokens after splitting on non-alphanumerics. So `api_key`, `X-Auth-Token`, and `db_password` are caught, but `apikey`, `pwd`, `passphrase`, `cookie`, `session_id`, and `sessionid` are not. Those values would appear in `__repr__`, `model_dump`, and the `signature` hash input unmasked. The signature issue is minor, since it is a hash input, not a log line.

Fix: add `apikey`, `pwd`, `passphrase`, `cookie`, `sessionid` to `_SECRET_KEY_PATTERNS`, and add a unit test per name that asserts `repr` shows `***`.

Status FUTURE.

### SEC-04 URL userinfo inside param values is not scrubbed (Low)

Kind: credential in a URL-valued param, e.g. `params={"base_url": "https://user:pass@host/"}`.  <!-- pragma: allowlist secret -->

Exposure: `_scrub_dict` (line 115-123) masks by key name only. A value under a non-secret key that contains `user:pass@` passes through `__repr__` (line 282-287), `model_dump` (line 269-280), and logs. `mlflow.py` already has `_USERINFO_RE` for this pattern, so the project has the regex; it is not applied here.

Fix: in `_scrub_value`, when a string value matches `_USERINFO_RE` (reuse one shared module-level regex), replace the userinfo with `***`. Add a test asserting `repr(EmbeddingProvider(kind="custom", model="m", params={"url": "https://u:p@h"}))` does not contain `p@h`.

Status FUTURE. The `"env:"` indirection (line 133-137) already keeps the documented route free of literal secrets, so this affects only users who paste a literal URL with userinfo.

### SEC-05 `scrub_secrets=False` opt-out (Info)

Kind: intentional unmasked dump for "resolved-secrets handoff".

Exposure: `providers.py:269-280`. The default is safe (`scrub_secrets=True`). A caller that passes `False` gets raw secrets. No caller in `src/` passes `False`; the only mention is the docstring. `hf_provider.py:202` calls `resolve_secrets()`, which resolves env values in memory and does not serialise them.

Status DOC: the opt-out is intended. Document in the docstring that it must never reach logs or YAML dumps. No code change required.

## Controls already in place (RESOLVED evidence)

- `.secrets.baseline` and `.pre-commit-config.yaml:81-93` run `detect-secrets` on every staged change, including `PrivateKeyDetector`, `GitHubTokenDetector`, and `JwtTokenDetector`.
- `providers.py:126-293` `EmbeddingProvider` masks secrets in `__repr__` and `__str__` (line 282-293), in `model_dump` (line 269-280), and in the cache signature (line 235). `__str__` is overridden because pydantic's own `__str__` would bypass the scrub; the comment at lines 289-292 records that.
- `providers.py:133-137` `"env:VAR"` indirection keeps secret values out of the stored config. `resolve_secrets` (line 246-263) fails loudly on a missing variable.
- `mlflow.py:34-41` `_strip_userinfo` is applied to the non-final retry log lines (line 150-159) and the re-raise path (lines 160-162).
- `hf_provider.py` contains no tokens. The pinned revisions at lines 68-70 are commit SHAs, not credentials.

## False-positive list (REJECTED)

1. `src/mlframe/training/core/_phase_train_one_target.py` and similar: `token` as a variable or dict key name, e.g. `_MISSING_LEVEL_TOKEN = "__missing__"` (`competition/train_test_union_frequency.py:40`) and `_NAN_TOKEN = "__nan__"` (`feature_selection/filters/cat_cooccurrence_svd.py:52`). These are sentinel strings, not credentials. REJECTED.
2. Tokenizer references (`tokenizer`, `AutoTokenizer`, `pad_token`, `eos_token`) in `hf_provider.py` and `text_encoder.py`. NLP tokenization, not an auth token. REJECTED.
3. `tests/integrations/test_mlflow.py:237`: `"https://user:supersecret@tracking.example.com/api"`. A fake credential used to test SEC-01's scrubbing behaviour; it is not a real secret. REJECTED.  <!-- pragma: allowlist secret -->
4. `.github/workflows/*.yml` `secrets.CODECOV_TOKEN`, `secrets.GITHUB_TOKEN`, `github.token`: references to GitHub-managed secrets, not literal values. REJECTED.
5. `.secrets.baseline` entries (`DiscordBotTokenDetector`, `GitHubTokenDetector`, etc.): detector configuration, not findings. REJECTED.
6. `src/mlframe/feature_selection/filters/_vendored/infonet/NOTICE.md`, `data/datasets/README.md`, and `_TRACKER.md` / `_VERIFICATION.md` benchmark notes: matches on the word "token" or "auth" in prose. REJECTED.
7. `# pragma: allowlist secret` on git commit SHAs in `hf_provider.py:68-70` and `.pre-commit-config.yaml:671`: pinned revisions, not secrets. REJECTED.
8. Logging of `config` / `params` objects across `src/mlframe/training` (grep for `logger.* config` and `__repr__`): the configuration dataclasses hold no credential fields. The only one that could hold a credential is `EmbeddingProvider`, which scrubs itself (see SEC-04 for the remaining userinfo gap). REJECTED for the other config classes.
9. `src/mlframe/training/feature_handling/providers.py:43-54` `auth` / `key` substrings in names like `monkey` or `author`: the whole-token matcher (line 84-95) was written specifically to avoid this. REJECTED.

## Coverage note

The first full-tree grep timed out on the broad pattern. Results came from narrower searches on `src`, `tests` (for `token` / `password` / `secret` / `api_key` assignments), `.github`, and the known credential-handling modules. Key-material regexes (`BEGIN ... PRIVATE KEY`, `AKIA`, `sk-`, `ghp_`, `xox`, `AIza`) returned no files under `src`. Tests and the remaining config and YAML files were not fully enumerated; a follow-up run of the key-material regexes over `tests` and `*.yml`/`*.json`/`*.env` is recommended.
