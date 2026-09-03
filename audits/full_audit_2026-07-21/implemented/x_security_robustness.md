# Cross-cutting: security & robustness audit -- mlframe audit

## Scope

Method: repo-wide `Grep` sweeps across `src/mlframe/` (excluding `feature_selection/filters/**` and
`feature_selection/shap_proxied_fs/**` per the task's exclusion list) for each of the six target patterns
(pickle/joblib dump+load, eval/exec/os.system/subprocess shell=True, path construction from
caller-controlled strings, HTML/JS building in reporting, hardcoded credentials, insecure YAML
deserialization), followed by a full/partial `Read` of every hit site to establish real context before
judging it a finding. 1806 non-excluded `.py` files exist under `src/mlframe/`; the grep sweeps covered
all of them for the six patterns above (zero `shell=True`, zero `yaml.load`/`yaml.unsafe_load`, zero
hardcoded-credential-shaped strings found anywhere in scope). The files below are the ones with actual
pattern hits that were opened and read for context.

Fully read (entire file):
- `src/mlframe/utils/disk_cache.py` (452 lines)
- `src/mlframe/training/io.py` (783 lines)
- `src/mlframe/inference/predict.py` (282 lines)
- `src/mlframe/utils/safe_pickle.py` (44 lines)
- `src/mlframe/reporting/report_html.py` (240 lines)
- `src/mlframe/reporting/output.py` (117 lines)
- `src/mlframe/reporting/renderers/_plotly_interactivity.py` (122 lines)
- `src/mlframe/data/datasets.py` (79 lines)
- `src/mlframe/training/suite_artefact_cache.py` (536 lines)

Partially read (specific line ranges around every pattern hit; file's remaining lines were not
inspected and are NOT covered by this report):
- `src/mlframe/training/core/_setup_helpers_pipeline_cache.py` (265 total; read ~180-266)
- `src/mlframe/training/feature_handling/cache.py` (542 total; read 1-120, 405-540)
- `src/mlframe/training/feature_handling/fingerprint.py` (437 total; read 235-273)
- `src/mlframe/feature_selection/wrappers/rfecv/_finalize.py` (271 total; read 230-272)
- `src/mlframe/feature_selection/wrappers/rfecv/_fit_fold.py` (397 total; read 296-380)
- `src/mlframe/feature_selection/wrappers/rfecv/__init__.py` (grep-context only, ~660-672)
- `src/mlframe/estimators/pipelines.py` (270 total; read 50-150)
- `src/mlframe/training/_trainer_train_and_evaluate.py` (1002 total; read 300-390, 625-695)
- `src/mlframe/training/core/predict.py` (731 total; read 610-709)
- `src/mlframe/training/core/_predict_main_suite.py` (559 total; read 100-200)
- `src/mlframe/feature_selection/_benchmarks/kernel_tuning_cache/auto_tune.py` (259 total; read 200-240)
- `src/mlframe/reporting/renderers/save.py` (288 total; grep-context ~207-225)
- `src/mlframe/training/neural/ranker.py` (953 total; grep-context ~645-660, 915-930)
- `src/mlframe/training/ranking/_ranker_suite_train.py` (929 total; grep-context ~880-896)
- `src/mlframe/utils/_param_oracle_store.py` (169 total; grep-context ~130-135)
- `src/mlframe/training/core/_phase_train_one_target_body.py` (955 total; grep-context ~516-598, 587-598, 697-713, 829-845)
- `src/mlframe/models/ensembling/score.py` (476 total; grep-context ~420-435)
- `src/mlframe/calibration/post.py` (928 total; grep-context ~895-915)
- `src/mlframe/reporting/renderers/plotly.py` (950 total; grep-context ~255-265)
- `src/mlframe/training/composite/cache_store.py` (699 total; grep-context, docstring-level only, ~line 10)
- `src/mlframe/training/core/_misc_helpers.py`, `src/mlframe/training/core/_setup_helpers_metadata.py`,
  `src/mlframe/training/_data_helpers.py` (grep hits were doc-comments only, one-line context each)

Total files reviewed (opened and read, full or partial): 27.
Total LOC actually read (sum of fully-read files + the specific line ranges read in partially-read
files, not the files' full sizes): 2655 (fully-read files) + 1082 (partial-read line ranges) = 3737 lines.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| SEC-1 | P1 | insecure-deserialization | `src/mlframe/training/_trainer_train_and_evaluate.py:346-365` | Cached-model reload (`use_cache=True`) does `joblib.load(model_file_name)` with only a path-containment (`trusted_root`) check -- no sha256 sidecar / integrity verification, unlike every sibling load site in the same codebase. |
| SEC-2 | P2 | silent-failure / error-swallowing | `src/mlframe/training/feature_handling/cache.py:480-502` | `_deserialize`'s `except Exception:` around `np.load(...)` funnels ANY numpy-load failure (truncated file, permission error, disk corruption) into the "legacy pickle" fallback branch whenever `allow_pickle=True`, not just genuine legacy-pickle payloads. |
| SEC-3 | P2 | pickle-of-live-object (defensive gap) | `src/mlframe/training/feature_handling/cache.py:99-127` (class `FeatureCache`) | `FeatureCache.__init__` stores `self._lock = threading.Lock()` with no `__getstate__`/`__setstate__` to strip it; the exact "caching live objects needs `__getstate__` exclusion" bug class the codebase has already fixed once (`training/neural/ranker.py:650-660`, `trainer_`/CUDA-tensor exclusion) is left unguarded here. |
| SEC-4 | P2 | test-coverage-gap | `tests/training/test_security_io_validation.py` | The test file for `_validate_trusted_path` covers only the path-containment branch; there is no test that plants a tampered/arbitrary pickle at the exact `model_file_name` a cached-model reload will hit (the scenario SEC-1 describes), so the documented residual risk has no regression guard. |
| SEC-5 | P2 | external-dependency / offline-robustness | `src/mlframe/reporting/renderers/plotly.py:263` | `fig.write_html(path, include_plotlyjs="cdn", ...)` embeds a `<script src="https://cdn.plot.ly/...">` reference; any HTML report produced this way is non-functional (blank charts) on an air-gapped host or one where an outbound-CDN network policy blocks `cdn.plot.ly`, and, in a hypothetical hostile-network scenario, trusts the CDN as a code-supply source for every viewer who opens the report. |

### SEC-1 (P1): `joblib.load` of cached models has no integrity check on the load side

`_trainer_train_and_evaluate.py:346-365`, when `use_cache=True` and a model file already exists at
`model_file_name`, loads it via plain `joblib.load(model_file_name)` after only checking that the
resolved path is inside `trusted_root` (defaulting to the file's own parent directory). The function's
own comment at line 351-356 names this exactly: *"RESIDUAL RISK (audit2 F2): path-containment cannot
catch a pickle an attacker plants AT the expected model_file_name (it's exactly where we look) -- only
integrity can... tracked as an owned follow-up."* Every sibling load path in the same repo already closed
this gap: `training/core/predict.py:load_mlframe_suite`, `training/core/_predict_main_suite.py`, and
`inference/predict.py:read_trained_models` all call `verify_sidecar`/`_vsidecar` (or route through
`safe_joblib_load`'s allowlisted unpickler) before deserializing. This one call site does neither -- it
uses plain `joblib.load` (the unrestricted `NumpyUnpickler`, not even the denylist-restricted
`_SafeJoblibUnpickler`) and performs no sha256 check at all. Failure scenario: a process (or another user)
with write access to the training run's model-output directory (a realistic multi-tenant CI / shared
training-host setup, which is exactly the threat model the sha256-sidecar infrastructure elsewhere in
this repo was built for) drops a crafted pickle at the exact `model_file_name` a subsequent `use_cache=True`
run will compute (deterministic from `model_name`/`model_name_prefix`/`data_dir`/`models_subdir`) --
the next training run executes it via `joblib.load`. Suggested fix direction: write a `.sha256` sidecar
at the save side for this specific artifact (the docstring already names this as the missing half) and
route the load through `mlframe.utils.safe_pickle.verify_sidecar` (fail-closed, mirroring
`load_mlframe_suite`), or switch to `safe_joblib_load` for defense-in-depth against the classic RCE gadgets
even before the sidecar work lands.

### SEC-2 (P2): overly broad except in `_deserialize`'s pickle-fallback gate

`training/feature_handling/cache.py:480-502`: `np.load(path, allow_pickle=allow_pickle, mmap_mode="r")`
is wrapped in a bare `except Exception:`. When `allow_pickle=False` (the default) this is correct --
any failure means refuse and raise `CachePickleRefusedError`. But when a caller has explicitly opted
into `allow_pickle=True`, the SAME broad except also catches non-pickle-related failures (a truncated
npz from a crashed write, an `OSError` from a locked file on Windows, a corrupt zip central-directory)
and routes them into the "legacy pickle fallback" branch, which then calls
`safe_load(path, allow_unverified=not _strict)` -- attempting to unpickle bytes that may not be a
pickle stream at all. Best case this raises a confusing `UnpicklingError` instead of the real I/O error;
worst case (low-probability but not provably impossible) a corrupted-but-pickle-shaped byte sequence gets
partially unpickled before failing. Suggested fix: narrow the except to the specific numpy exceptions that
indicate "this is a legacy pickle-format file, not a corrupt npz" (`ValueError`/`OSError` from `np.load`'s
own pickle-format rejection carries a distinguishable message), and let genuine I/O errors (`PermissionError`,
`zipfile.BadZipFile` from a truncated write) propagate as themselves.

### SEC-3 (P2): `FeatureCache` has no `__getstate__` guarding its `threading.Lock`

`training/feature_handling/cache.py:99-127` (`class FeatureCache.__init__`) sets `self._lock =
threading.Lock()` as a plain instance attribute. No `__getstate__`/`__setstate__` (or `__reduce__`)
excludes it. Grepping the codebase for `FeatureCache(` / `feature_cache` turned up no current call
site that stores a `FeatureCache` instance inside a structure that later gets `joblib.dump`'d or
`dill.dump`'d (the cache flows as a plain function parameter into `apply_preprocessing_extensions`-style
helpers and is never assigned into the `metadata` dict that `training/io.py:save_mlframe_model`
persists) -- so this is not confirmed as currently reachable. It is, however, exactly the bug class this
same file was already burned by once: `training/neural/ranker.py:650-660` had to add a defensive
`__getstate__` specifically to strip a live `lightning.pytorch.Trainer` (which itself carries a
`WarningCache`) before a downstream `joblib.dump` of a fitted `MLPRanker`, with the same "attribute
survives only because nobody currently pickles the parent object" precondition. Any future caller that
attaches a `FeatureCache` to a persisted object (or calls `pickle.dumps(fhc._cache)` directly, e.g. from
a debugger or a new caching layer built on top) hits `TypeError: cannot pickle '_thread.lock' object`
with no diagnostic pointing at the cache. Suggested fix: add a `__getstate__` that drops `_lock` (and
re-creates it in `__setstate__`), mirroring the `ranker.py` pattern, as a defensive guard even though no
active pickling path currently reaches it.

### SEC-4 (P2): no regression test for the SEC-1 gap

`tests/training/test_security_io_validation.py` tests `_validate_trusted_path` thoroughly (rejects paths
outside `trusted_root`, symlink-escape case, cross-drive `ValueError` re-raise) but has no test that
exercises the actual `use_cache=True` reload path in `_trainer_train_and_evaluate.py` with a
tampered/arbitrary pickle planted at the computed `model_file_name`. Per this repo's own convention
("every ML trick / bug-class fix gets a regression test"), the documented residual risk in SEC-1
currently has zero test coverage backing it -- there's nothing that would fail today to prove the gap
exists, nor anything that would catch a regression once it's fixed.

### SEC-5 (P2): plotly HTML reports depend on `cdn.plot.ly` at view time

`reporting/renderers/plotly.py:263`: `fig.write_html(path, include_plotlyjs="cdn", auto_open=False,
config=html_config())`. Every interactive HTML chart this renderer produces embeds a `<script
src="https://cdn.plot.ly/...">` tag rather than inlining plotly.js. On a host with no outbound internet
access (a common posture for on-prem ML training boxes / audited enterprise networks) the resulting
report renders as blank panels with no error surfaced to the viewer -- degraded-but-silently-wrong from
a report-consumer's point of view. This is very likely an intentional file-size tradeoff (`include_plotlyjs="cdn"`
keeps each HTML file tiny vs. the ~3-4 MB `include_plotlyjs=True` would add per report) rather than an
oversight, so flagging as a robustness/offline-availability finding rather than a vulnerability; a
CDN-substitution attack would additionally require compromising `cdn.plot.ly` itself or the viewer's DNS,
which is out of mlframe's control either way.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PROP-1 | test-coverage | `tests/training/test_security_io_validation.py` | Add the planted-pickle-at-`model_file_name` regression test described in SEC-4, both to document the current gap and to pin the eventual SEC-1 fix. |
| PROP-2 | consistency | `src/mlframe/training/_trainer_train_and_evaluate.py:360` | Route this one remaining plain `joblib.load` call through `safe_joblib_load` (already used by `inference/predict.py`) as a cheap first step -- it adds the RCE-gadget denylist immediately, ahead of the larger sidecar-writing work SEC-1's full fix needs. |
| PROP-3 | defensive-hardening | `src/mlframe/training/feature_handling/cache.py` | Add `__getstate__`/`__setstate__` to `FeatureCache` per SEC-3, even without a currently-reachable pickling path -- cheap now, and this exact bug class has already bitten the codebase once (`ranker.py`). |
| PROP-4 | offline-robustness | `src/mlframe/reporting/renderers/plotly.py` | Consider a config knob to switch `include_plotlyjs` between `"cdn"` (current default, small files) and `True`/a vendored local copy (works offline) for operators who need air-gapped report viewing; document the current CDN dependency in the renderer's own docstring so it's a documented tradeoff rather than an implicit one. |

## Coverage notes

- The two explicitly excluded packages (`feature_selection/filters/**`, `feature_selection/shap_proxied_fs/**`,
  ~375 files) were not read at all, per the task's exclusion list -- confirmed via `find`/`grep` file counts,
  not spot-checked.
- The `_benchmarks/` subtrees under `feature_selection/` and `training/` (dev-only profiling/benchmark
  scripts, not part of the public API) received only the pattern-grep pass, not a full read; every
  `subprocess`/`exec` hit found there was a fixed-arg, git-history-sourced, or `sys.executable -c` call with
  no externally-reachable input, so none were promoted to findings, but a line-by-line read of those ~50
  files' full bodies was not performed.
- `pyutilz.core.safe_pickle` (the actual implementation behind `mlframe.utils.safe_pickle`'s re-exports) is
  a separate first-party package outside this repo's `src/mlframe/` tree; its own `verify_sidecar`/`safe_load`
  internals (fail-open-on-missing-sidecar semantics, etc.) were read only through mlframe's re-export shim's
  docstrings, not by opening the pyutilz source itself -- out of this audit's repo scope.
- Given the scale (1806 in-scope `.py` files), the six target patterns were pursued exhaustively via
  repo-wide grep (every hit read for context), but the broader 12-point checklist (silent-except sweep,
  RNG/seed audit, GPU/CPU parity, memory-copy discipline, file-size/LOC-limit sweep, etc.) was applied only
  incidentally to the files opened while chasing the six security patterns, not as an independent
  file-by-file pass across all 1806 files -- that would need a dedicated pass beyond this cluster's stated
  scope and the available session budget.
- `training/composite/cache_store.py` (699 lines) was grepped for `pickle.load`/`joblib.load` (zero direct
  calls found; the module docstring documents it uses `safe_pickle.safe_dump`/presumably a matching safe-load
  wrapper) but was not read in full to confirm the load-side implementation matches the docstring's claim --
  flagged here rather than silently assumed correct.
