# Network, HTTP servers and external endpoints audit, round 2 (mlframe, src/mlframe)

Read-only. No source, test, or config file changed; no git state changed. Builds on `audits/2026-10-31/network.md` (NET-01..NET-09), which is not re-reported except where a claim is corrected.

## Summary

- Confirmed again: no listener in `src/mlframe` (no Flask/FastAPI/Dash/uvicorn/`http.server`/`socketserver`/`socket.bind`/`listen`/`start_server`/`serve_forever`/`app.run`/`launch(share=True)`). No `0.0.0.0` bind. No direct `requests`/`httpx`/`aiohttp`/`urlopen`/`urlretrieve` call site. No `verify=False`, `CERT_NONE`, `check_hostname=False`, `_create_unverified_context`, or `allow_redirects`/`follow_redirects` override. No telemetry SDK import.
- Outbound network is limited to: HuggingFace `from_pretrained` (`training/feature_handling/hf_provider.py`, `feature_selection/filters/_neural_mi.py:473`), MLflow tracking (`integrations/mlflow.py`, user-configured URI), and the `sklearn` demo fetch in `boruta_shap`. No `read_csv`/`scan_parquet`/`load_dataset` of a URL, and no `s3://`, `gs://`, `ftp://` source, in non-benchmark code.
- Two prior items are corrected by this pass (NET-02, and the "unverified" InfoNet note). One new Medium finding: the HF provider's `trust_remote_code` opt-in is a string test that inverts for the value `"False"`.
- Counts (new in this round): Medium 1, Low 1, Info 4.

## Corrections to the 2026-10-31 report

- **NET-02 (Low, FUTURE): the code-execution half is wrong.** The prior report said a tampered InfoNet checkpoint would run code if `torch.load` lacked `weights_only=True`. The load path does pass it: `feature_selection/filters/_vendored/infonet/infer.py:51` calls `torch.load(checkpoint_path, map_location=device, weights_only=True)`, and lines 49-50 refuse a fallback to an unrestricted unpickle on old torch. What remains is the integrity gap: no SHA-256 check on `infonet_pretrained.pt`, so a swapped checkpoint can silently change MI estimates (not execute code). Re-proposed as **NET2-03, Info, FUTURE**.
- **The "sys.path injection, unrated" note** is now verified. See NET2-04.

## Findings

| ID | File:line | Severity | Proposed disposition |
|---|---|---|---|
| NET2-01 | `src/mlframe/training/feature_handling/hf_provider.py:207` (fed by `src/mlframe/training/feature_handling/providers.py:207`) | Medium | FUTURE |
| NET2-02 | `src/mlframe/integrations/mlflow.py:36, 41` | Low | FUTURE |
| NET2-03 | `src/mlframe/feature_selection/filters/_neural_mi.py:280-300`, `src/mlframe/feature_selection/filters/_vendored/infonet/infer.py:51` | Info | FUTURE |
| NET2-04 | `src/mlframe/feature_selection/filters/_neural_mi.py:310-318, 425-431` | Info | DOC |
| NET2-05 | `src/mlframe/feature_selection/boruta_shap/__init__.py:823-829` | Info | DOC |
| NET2-06 | `src/mlframe/integrations/mlflow.py:71-96` | Info | DOC |

## Details

### NET2-01: `trust_remote_code` opt-in inverted by the string value "False" (Medium, FUTURE)

`HuggingFaceProvider` decides whether the hub repo's Python code runs with:

```
hf_provider.py:207  trust_remote = bool(params.get("trust_remote_code", False))
```

The params come from `EmbeddingProvider.from_uri` (`providers.py:175-225`), which keeps query values as raw strings: `parse_qs` returns `['False']` and line 209 flattens it to `'False'`. `bool('False')` is `True`. The same happens for a dict passed as `params={"trust_remote_code": "False"}`.

Failure scenario: a user or a shared config writes `hf://org/model?trust_remote_code=false` (or `=False`) to make the opt-out explicit. The provider then calls `AutoTokenizer.from_pretrained` and `AutoModel.from_pretrained` with `trust_remote_code=True`, so the repo's `modeling.py` executes in the training process. The attacker needs control of the repo content or of the model name in config. The code emits one `UserWarning` at `hf_provider.py:208-215`, but the user asked for the opposite, so the warning does not undo the mistake. The `revision` default is the hub `main` unless `PINNED_REVISIONS` covers the model (see NET-05), so the code run is not pinned either.

Why Medium: a security-control default is inverted by an ordinary input spelling, and the result is arbitrary code execution in-process. Not High because the trigger needs a hub repo that carries malicious code (or a config that points at one) plus an explicit `False` value.

Fix (not applied, read-only): parse the value as a boolean, accepting only `"true"`/`"1"`/`"yes"` as true and rejecting other strings, or parse the URI's boolean params in `from_uri`. A regression test should call `build_provider(EmbeddingProvider.from_uri("hf://x/y?trust_remote_code=False"))` and assert `_load_trust_remote is False`.

Disposition FUTURE: the fix is small, but it changes how URI params are typed for every provider. The owner should decide the typing rule first. This one is cheap, so it should not wait long.

### NET2-02: MLflow userinfo scrubber misses passwords containing `@` (Low, FUTURE)

`_USERINFO_RE = (?i)([a-z][a-z0-9+\-.]*://)[^/@\s]+@` (`integrations/mlflow.py:36`) and `_strip_userinfo` (line 41) redact the userinfo of a tracking URI before an error is logged. The class `[^/@\s]` excludes `@`, so the match stops at the first `@`.

Failure scenario: `MLFLOW_TRACKING_URI=https://user:p@ss@mlflow.internal` (an unencoded `@` in the password). `_strip_userinfo` returns `https://***@ss@mlflow.internal`, so `ss` is logged. Any caller who can read the log (a shared CI log, for example) gets part of the password. A `/` in an unencoded password has the opposite effect: the regex does not match at all, so nothing is redacted. Both need an unencoded special character in the password. The URI spec requires them to be percent-encoded, so exposure is limited to sloppy configs.

Fix: match up to the last `@` before the host, for example `[^\s/]*@` applied to the authority part, or use `urllib.parse.urlsplit(...).password` to rebuild the string.

Disposition FUTURE: a real leak of part of a credential, with a narrow trigger. Existing test `tests/integrations/test_mlflow_scrubbed_giveup_log.py` covers the simple case only; add a case with `@` in the password.

### NET2-03: InfoNet checkpoint has no integrity check (Info, FUTURE)

This is the surviving part of the prior NET-02. The checkpoint `infonet_pretrained.pt` is loaded from `_INFONET_CACHE_DIR` (`_neural_mi.py:266-268`, default `~/.cache/mlframe/infonet`, overridable by `MLFRAME_INFONET_CACHE`), and the user fills that directory by hand from a Google Drive folder (`_neural_mi.py:288-291`). There is no SHA-256 check, unlike the sidecar scheme in `utils/safe_pickle.py`. `weights_only=True` (`infer.py:51`) blocks code execution, so the impact is a silently different MI estimate if someone with write access to the cache directory swaps the file. That is a correctness and integrity issue, not a remote-code path.

Disposition FUTURE: pin the SHA-256 of the published checkpoint and check it in `_get_infonet_model` before `load_model`.

### NET2-04: InfoNet `sys.path` injection is scoped, but its imports stay in `sys.modules` (Info, DOC)

`_neural_mi.py:310-318` and `425-431` insert the vendored `_vendored/infonet` directory at `sys.path[0]` only for the duration of `from infer import ...`, and remove it in `finally`. This resolves the prior "unrated" note: the injection is correctly bounded. What is left is that `infer` imports `model.decoder`, `model.encoder`, `model.infonet`, and `model.query`, and those top-level names (`model`, `util`, `query`, `decoder`, `encoder`, `attention`) stay in `sys.modules` for the process lifetime. Any later `import model` in the process gets the vendored package. The code comment at `_neural_mi.py:299-306` already documents this. No attacker reach: the directory is inside the installed package, and nothing writes to it.

Disposition DOC: the behaviour is intended and commented. Optional: rename the vendored package to a namespaced name (for example `_infonet_model`) so it cannot shadow a common module name.

### NET2-05: Boruta `load_data("regression")` downloads a dataset on the default path (Info, DOC)

`feature_selection/boruta_shap/__init__.py:823-829`: `load_boston` was removed in sklearn 1.2, so its `ImportError` branch always runs on a current sklearn and calls `fetch_california_housing(as_frame=False)`, which downloads from a public host at runtime. The download is not pinned by mlframe, and no timeout is set (see NET-06). The data is public and the call is a documented demo fallback, with a comment at lines 826-827 explaining it.

Failure scenario: none that reaches a production path; the function is a usage example. A poisoned cache or a MITM on the sklearn download would alter the demo data only.

Disposition DOC: the comment already explains the fallback. Consider noting the network access in the docstring of `load_data`, which says only "Load Example datasets".

### NET2-06: `embed_website_to_mlflow` has no production caller (Info, DOC)

`integrations/mlflow.py:71-96` is public API. A grep of `src/mlframe` finds no caller; only `tests/integrations/test_mlflow.py` and `tests/test_meta/test_shared_uncalled_functions.py:59` use it. The prior NET-01 concern (sandbox `allow-same-origin allow-scripts`, unsanitised `fname` path) therefore has no reachable production path today. The docstring (lines 74-80) already states the trust boundary.

Disposition DOC: keep the docstring warning. If the helper is ever wired to user input, first drop `allow-same-origin` from the sandbox attribute on line 88.

## Out-of-class observation (not counted)

`integrations/mlflow.py:164`: in `_start_run_with_retries`, the `else:` clause of the `try` runs `mlflow.end_run()` immediately after a successful `mlflow.start_run`, and the returned run is therefore already ended. This is a correctness bug in the logging path, not a network finding. Route it to the correctness audit or fix it with the MLflow tests.

## Scope note

- Searched the whole `src/mlframe` tree (non-benchmark) for network imports, server APIs, TLS overrides, redirects, bind addresses, URL literals, tracking URIs, hub loads, and remote data reads. Benchmark code was searched only for the NET-07 claim, not reviewed.
- Not verified: the live HuggingFace and MLflow server behaviour; the sklearn checksum for `fetch_california_housing` (stated from memory, not checked against the installed version); the `mist_statinf` package's internal loader, which is third-party.
- Dependency and CI workflows (`.github/`) are out of this class and covered by the supply-chain and secrets reports.
