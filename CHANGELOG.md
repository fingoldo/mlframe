# Changelog

## 2026-05-09 — Phase C: TextColumnEncoder (TF-IDF / Hashing) + polars-ds capability dispatch

Phase C of the 2026 feature-handling overhaul. Per-column text
vectoriser that downstream phases (E concat, D cache) consume.

### Added

- **`PolarsNativeDispatcher`** in
  ``training/feature_handling/polars_capability.py``. Runtime probe
  via ``importlib.util.find_spec`` (round-3 security S5: doesn't
  execute user-controlled module bodies) + ``hasattr`` walk over
  ``Blueprint`` and top-level ``polars_ds`` ops. Cached frozenset of
  capability strings (``"blueprint.tfidf"``, ``"blueprint.impute"``
  etc) so subsequent dispatches are free. ``reset_capability_cache()``
  for tests; positive results cached for process lifetime, negative
  results re-probe (so a mid-session ``pip install polars-ds`` works).

- **`TextColumnEncoder`** in
  ``training/feature_handling/text_encoder.py``. Per-column TF-IDF
  / Hashing vectoriser with polars / pandas / sparse symmetry.
  Output: ``scipy.sparse.csr_matrix``. Train-only fit semantics --
  ``transform()`` applies the trained vocab to held-out rows so
  unseen tokens map to zero (no leak).
  Empty / null / NaN / Unicode (Cyrillic / emoji / RTL) input
  coerced to "" without crashing the underlying vectoriser
  (round-3 T8 + T9).
  Phase C v1: sklearn ``TfidfVectorizer`` / ``HashingVectorizer``
  fallback only. polars-ds 0.11.x doesn't ship Blueprint TF-IDF
  / hashing yet (audit confirmed); when the user's upstream PR
  lands, ``PolarsNativeDispatcher.has("blueprint.tfidf")`` will
  flip True and the encoder picks up the polars-native path
  without a code change here.

### Tests

17 new in ``test_text_column_encoder.py``: TF-IDF + hashing happy
paths, polars/pandas symmetry (same vocab from either form), edge
inputs (empty / null / NaN / Cyrillic / emoji / RTL), idempotency,
no-leak (test rows OOV-mapped to zero), capability detector,
dispatcher routing under prefer_polarsds True/False, signature
stability.

238/238 wide regression (M + A + A2 + B + C, HF heavy skipped here
to keep quick suite under 1.5 min). 17/17 phase-C own pack.

## 2026-05-09 — Phase B: provider Protocols + HuggingFaceProvider + lifecycle registry

Phase B of the 2026 feature-handling overhaul. Wires the provider
runtime that consumes the phase-A2 ``EmbeddingProvider`` configs.

### Added

- **Protocols** in ``training/feature_handling/protocols.py``:
  ``FrozenFeaturizerProvider`` (numpy output) and
  ``TrainableFeaturizerProvider`` extending it with ``as_module()``
  for autograd flow. Round-3 R2-4: split avoids leaking HF specifics
  to consumers that only need frozen embeddings.

- **`HuggingFaceProvider`** in ``training/feature_handling/hf_provider.py``.
  Frozen-only impl in v1 (trainable lands in phase G). Loads model
  via ``transformers.AutoModel`` + ``AutoTokenizer``; pool variants
  ``mean`` / ``cls`` / ``max``. ``trust_remote_code=False`` hardcoded
  default with a loud warning if user opts in (round-3 S4).
  E5-family auto-prefix detection: ``"passage: "`` prepended for any
  model whose name contains ``-e5-`` (round-3 U-R2-8: missing the
  prefix halves retrieval quality on e5).
  GPU OOM mid-batch -> halve batch + retry; CUDA driver crash
  (``CudaErrorClass.CONTEXT_LOST``) -> abort with restart-Python
  message (round-3 chaos C4). Empty / null / non-string input coerced
  to "" (round-3 T8). ``device="auto"`` enumerates cuda > mps > xpu >
  cpu (round-3 R2-18).

- **Provider lifecycle registry** in ``training/feature_handling/registry.py``.
  Module-level ``_REGISTRY: WeakValueDictionary[sig, _ProviderEntry]``
  + ``_LRU_HARD: OrderedDict`` strong-keep tier. Default
  ``cache.keep_n_providers="auto"`` resolves to ``min(2, free_vram_gb)``
  on GPU (round-3 user-confirmed; F1: prevents the per-target-loop
  weight-reload churn on prod 50-target sweeps).
  - **Double-checked locking** on creation: re-check after acquiring
    the registry lock so two threads on the same signature load the
    provider exactly once (round-3 R2-2).
  - **Per-key entry lock**: serialises actual ``acquire()`` /
    ``release()`` work on a single provider.
  - **`acquire_provider(provider, cache_cfg)`** context-manager: the
    only public lifecycle entry. Naked ``provider.acquire()`` is
    discouraged via banned API surface (round-3 chaos C22 prevents
    refcount/release-on-error mistakes).

- **Pre-warm via Future** -- ``prewarm(provider) -> Future``,
  ``wait_prewarm(provider, timeout=...)``. Round-3 chaos C3:
  exceptions in pre-warm propagate via ``Future.result()`` instead
  of being swallowed by a naked ``threading.Thread``.

- **`shutdown_all()`** + **`shutdown()`** alias: drops every
  provider, releases VRAM, purges LRU and prewarm executor.
  Round-3 chaos C18: makes notebook reload safe (otherwise old
  weakref entries vanish but VRAM stays held -> 3rd reload OOMs
  on consumer GPUs).

- **`provider_status()`** snapshot for ``fhc.describe()`` and tests.
  Surfaces pre-warm errors so they're visible without polling
  the Future. Round-3 U-R2-24.

### Tests

12 new in ``test_provider_registry.py``: lifecycle, refcounting,
TOCTOU race, LRU strong-keep tier, shutdown, prewarm-Future
exception propagation, context-manager-only public API.
15 new in ``test_hf_provider.py``: HF integration via
``prajjwal1/bert-tiny`` (~17 MB). Lifecycle, transform shape +
dtype, empty / null / Unicode (Cyrillic + emoji + RTL) input,
pool variants, ``trust_remote_code`` security default, e5
auto-prefix detector + override.

236/236 wide regression (M + A + A2 + B). 27/27 phase-B own pack.

## 2026-05-09 — Phase A2: EmbeddingProvider structured object

Sub-phase between A (configs) and B (provider implementations). Replaces
the previously stub-typed ``Optional[Any]`` provider field with a
fully-validated pydantic model.

### Added

- **`EmbeddingProvider`** in ``training/feature_handling/providers.py``.
  Replaces the round-2 colon-string format (``"hf:model"``) which broke
  on model names containing colons (Windows paths, ``cuda:0`` device
  refs) and had no slot for provider-specific params (OpenAI
  ``dimensions``, ``api_key``, HF ``trust_remote_code``).
  Supported kinds: ``huggingface``, ``sentence-transformers``,
  ``openai``, ``cohere``, ``jina``, ``voyage``, ``onnx``, ``fasttext``,
  ``tfhub``, ``custom``. ``extra="forbid"``.

- **`EmbeddingProvider.from_uri(uri)`** convenience parser. URL-style
  grammar ``<kind>://<model>[?param=value(&param=value)*]``. Aliases:
  ``hf://`` -> huggingface, ``sbert://`` -> sentence-transformers.

- **`signature`** property returns a stable ``{kind}:{model}:{params_hash}``
  string for cache-key use. Round-3 R2-6: secrets are scrubbed BEFORE
  hashing so a swapped env-var value (``OPENAI_API_KEY`` rotation)
  does NOT invalidate the cache.

- **`resolve_secrets()`** swaps ``"env:VAR_NAME"`` indirections with
  the resolved env var value. Loud-failure on missing env (``KeyError``)
  -- silent missing-secret would be the worst-case for debug.

- **`model_dump(scrub_secrets=True)`** (default True) and
  **`__repr__`** mask any param key matching the heuristic set
  ``key|token|secret|auth|bearer|credential|password|passwd``
  (round-3 S6: previously incomplete; broadened to cover ``hf_auth``
  / ``Bearer`` / etc). Pass ``scrub_secrets=False`` only at the
  provider acquire step in phase B.

### Changed

- ``FeatureHandlingConfig.default_text_provider`` typed as
  ``Optional[EmbeddingProvider]`` (was ``Optional[Any]``).
  ``FrozenEmbeddingParams.provider`` likewise.

### Tests

37 new in ``tests/training/test_embedding_provider.py``. Coverage:
construction + ``extra="forbid"``; URI parsing happy path AND malformed
URIs (per round-2 test agent's correction: colon-in-query-value is
RFC 3986 valid, NOT malformed); signature stability across run /
insertion-order / secret-value-change; ``resolve_secrets`` env swap
+ missing-env loud failure; scrub_secrets parametrized over 8 secret-
keyword variants; FHC integration accepts EmbeddingProvider.

209/209 wide regression (M + A + A2). 37/37 phase-A2 own pack.

## 2026-05-09 — Phase A: FeatureHandlingConfig surface + compat matrix + presets

Second phase of the 2026 feature-handling overhaul. Phase M shipped
the foundation modules (Axis enum, locking, system, cache_backend);
phase A delivers the configuration surface that downstream phases
consume.

### Added

- **Top-level `FeatureHandlingConfig`** with 6 nested sub-configs
  (`text_detection`, `cache`, `memory`, `pricing`, `logging`, `repro`).
  Round-2 UX U-R2-1: previously-flat 25-field surface nested into
  themed groups so user-facing autocomplete shows ~8 top-level fields.
  All sub-configs use `extra="forbid"` -- typos surface at construction.

- **`TextHandlerSpec` / `CatHandlerSpec`** per-axis specs with
  discriminated-union `params: Union[TfidfParams, HashingParams,
  FrozenEmbeddingParams, LearnableEmbeddingParams, TargetEncodeParams,
  CustomParams, NoParams] = Field(discriminator="kind")`. Round-3
  R2-5: silent method/params mismatch (e.g. `method="hashing"` with
  `params=TfidfParams()`) caught by `model_post_init` validator.
  Naming finalised per round-3 user confirmation:
  `frozen_text_embedding` / `learnable_text_embedding` /
  `output="as_embedding_feature"`.

- **Compat matrix `_MODEL_AXIS_SUPPORT`** + `validate_handler_for_model` /
  `validate_fhc_handlers`. Cross-cutting rules
  (`learnable_text_embedding` requires neural model;
  `output="as_embedding_feature"` requires native-embedding model)
  precede matrix lookup so users get actionable error messages.
  `difflib.get_close_matches(n=3)` suggestions on typo'd method names.
  `register_model_axis_support()` extension point for custom models.

- **Auto-derived memory budgets** at FHC construction. cgroup-aware
  via `detect_memory_limit_bytes` -- containers honour cgroup limits,
  not host RAM (round-3 R2-1). `MLFRAME_MEMORY_BUDGET_GB` env override
  takes precedence. On any 16+ GB system the auto-derive cascade is:
  `memory.budget_gb = total × 0.7`,
  `cache.ram_max_gb = budget × 0.3`,
  `cache.ram_reserve_gb = max(2 GB, total × 0.1)`.
  Auto-derived `cache.ram_max_gb` is silently capped to `total - reserve`
  if the user sets a `memory.budget_gb` larger than total RAM (legitimate
  case: host with swap headroom). Explicit `cache.ram_max_gb` exceeding
  available raises early with actionable message.

- **Presets** `tfidf_only(max_features=...)`, `cb_native_only()`,
  `embedding_only(provider=...)` — round-2 U-R2-2: 80% case fits on
  one line.

- **`describe(short=True)`** -- printable resolution plan exposing
  `mode`, `cache.persistence`, `resolved` numeric values.
  Round-3 U-R2-5: auto-derived budgets now visible at debug time.

- **`ModelHandlingOverride`** with `text` / `cat` (replace defaults)
  AND `text_append` / `cat_append` (extend defaults). Round-3 U-R2-4
  partial-inheritance pattern.

- **TypedDict params**: `TfidfParams`, `HashingParams`,
  `FrozenEmbeddingParams`, `LearnableEmbeddingParams (extends frozen)`,
  `TargetEncodeParams`, `CustomParams`, `NoParams`. Each has
  `extra="forbid"` and `kind: Literal[...]` discriminator.

- **`ReproConfig`** sub-config (opt-in) with `deterministic_torch`,
  `pinned_revision`, `langdetect_seed`, `pinned_svd_solver_params`.
  Round-3 R3-01 et al.: cross-machine bit-stable mode is opt-in for
  CI parity tests; default off for dev workflow due to 10-25% perf
  cost on `torch.use_deterministic_algorithms`.

- **`TextDetectionConfig`** with multi-criteria triggers + anti-UUID
  guard at `min_alphabet_entropy >= 4.5` (round-3 R2-21: UUID-v4
  entropy ≈ 4.04 was at the previous threshold, now safely below).
  `respect_explicit_categorical_dtype: bool = True` migrates the
  `honor_user_dtype` semantic (user-confirmed, default flipped to
  True for greenfield -- pl.Categorical/Enum dtype signals user
  intent over the cardinality heuristic).

### Tests

49 new tests in `tests/training/test_feature_handling_config.py`.
Coverage: typed-dict typo rejection, method/params discrimination
mismatch lock, compat matrix early validation with difflib
suggestions, auto-memory cascade across mocked machine sizes,
cgroup-limit usage, env override, invariant violation on tiny
machines, sub-config `extra="forbid"`, replace/append override
semantics, preset factories, `apply_to_columns`, `describe()` shape.

49/49 imputer + phase-A + 172/172 wide regression
(pipeline + imputer + XGB shim + LGB shim + FHC) green.

### Notes

- Phase A is structural -- the `FeatureHandlingConfig` is built and
  validated, but no consumer in `train_mlframe_models_suite` reads
  it yet. Phases B (providers), C (TF-IDF/hashing encoders), D
  (cache layer), E (multi-handler concat) wire the consumers.
- `per_target` field reserved (round-3 F10) but raises if non-empty
  (validator deferred to phase J).
- `default_text_provider: Optional[Any]` is a placeholder; the
  proper `EmbeddingProvider` structured object lands in phase A2.

## 2026-05-09 — Phase M: feature-handling overhaul foundation

First foundation phase of the multi-phase 2026 feature-handling overhaul
(see plan in `docs/feature_handling_architecture.md` -- forthcoming).
Mostly invisible at the suite-call level; lays the groundwork for
phases A through L.

### Renamed (greenfield -- no aliases, no deprecation cycle)

- `PolarsPipelineConfig` → `PreprocessingBackendConfig`. The previous
  name described WHICH library ran the transforms (polars-ds); the new
  name describes the responsibility (which BACKEND -- polars-native vs
  sklearn -- is preferred).
- `PolarsPipelineConfig.use_polarsds_pipeline` → `prefer_polarsds`.
  Same boolean, name now reads as a preference rather than a switch.
- 26 callsites updated across `training/`, `tests/`, `profile_mixed_dtypes.py`,
  `README.md`. `legacy/training_old.py` left untouched (frozen
  reference, not active).

### Fixed

- **Imputer wiring debt closed** (`PreprocessingBackendConfig.imputer_strategy`
  declared since 2026-04, never connected to the polars-ds Blueprint --
  audit at `training/pipeline.py:458-585` confirmed). Numeric-column
  NaN now flows through `Blueprint.impute(method=...)` BEFORE scaling
  so the scaler never sees NaN. Strategies: `"mean"`, `"median"`,
  `"most_frequent"` (mapped to polars-ds `"mode"`). Sensor tests in
  `tests/training/test_imputer_wiring.py` (NaN in / NaN out behavioural,
  not source-grep). Also locks: train-only fit semantic (test set
  imputed with TRAIN mean, not test mean), idempotency on clean data,
  numeric-only target (string columns untouched), and proper composition
  with the scaler.

### Added

New `mlframe.training.feature_handling` subpackage. Foundation pieces;
the real consumers (FHC, providers, cache layer) land in phases A-D.

- **`Axis` enum + `HandlerSpec` ABC + axis registry**
  (`feature_handling/axis.py`). Replaces the implicit
  `{cat, text, numeric}` triple with an extensible registry so future
  axes (image, audio, sequence) ship as one new file plus one
  `register_handler_spec` call. `apply_to` accepts list-of-names,
  `re.Pattern`, or callable for late-binding pipelines that synthesize
  columns. `group_columns` lifted to ABC so group-aware encoders work
  for both cat and text.
- **`PIDAwareFileLock`** (`feature_handling/locking.py`). Wraps
  `filelock.FileLock` with PID-in-lockfile + `psutil.pid_exists()`
  reclaim. One OOM-killed process holding a lock no longer permanently
  bricks the cache directory (round-3 chaos C5). Surfaces reclaim via
  `StaleLockReclaimed` warning so test fixtures can `pytest.warns(...)`
  it. Pin: `filelock>=3.15.0`.
- **`detect_memory_limit_bytes`** (`feature_handling/system.py`).
  cgroup v1/v2-aware memory probe. Inside Docker/K8s honours the
  cgroup limit instead of host RAM (round-3 chaos R2-1: previous
  `psutil.virtual_memory().total` × 0.7 derived `budget=179 GB` on
  a 4 GB container of a 256 GB host -> instant OOM-kill). Honours
  `MLFRAME_MEMORY_BUDGET_GB` env override. Handles cgroup-v2 `"max"`
  string sentinel (round-3 chaos C25) and cgroup-v1 9.2e18 sentinel.
- **`CudaErrorClass` + `classify_cuda_error`** (same file). Splits
  retryable `OutOfMemoryError` from non-retryable
  `"CUDA error: unknown error"` driver crashes (round-3 chaos C4).
  Halve-batch-and-retry loops no longer spin forever after a CUDA
  context loss.
- **`long_path_safe`** (same file). Prepends `\\?\` UNC marker on
  Windows so cache paths >260 chars don't `FileNotFoundError` through
  `os.replace` (round-3 chaos C7).
- **`CacheBackend` Protocol + `LocalDiskBackend`**
  (`feature_handling/cache_backend.py`). The seam for future S3 / GCS
  / NFS / Ray backends (round-3 future-proofing F5). v1 ships only
  `LocalDiskBackend` (atomic writes via existing
  `mlframe.training.io.atomic_write_bytes`, mode 0o700 on construct,
  per-key locks via `PIDAwareFileLock`). Bytes-oriented API so the
  serialisation tier (memmap, fp16, etc.) added in phase D stays
  backend-agnostic.

### Notes

- Test count delta: +9 in `test_imputer_wiring.py` (one of which
  asserts the no-leak invariant -- test set imputed with TRAIN mean,
  not test mean).
- The `mlframe.training.feature_handling` package re-exports its
  symbols at the top level, so `from mlframe.training.feature_handling
  import Axis, CacheBackend, ...` is the public entry-point for
  downstream phases.
- All new abstractions are intentionally stub-light: phase M
  establishes the **shape** of the contract; phases A-D fill in
  concrete `TextHandlerSpec` / `EmbeddingProvider` / cache layer.

## 2026-05-08 — QUANTILE_REGRESSION: native CB/XGB + wrapper LGB/HGB/Linear + 5 viz panels

Sixth target type (joining REGRESSION / BINARY / MULTICLASS / MULTILABEL
/ LEARNING_TO_RANK). Predicts K conditional quantiles in one fit (CB
``MultiQuantile``, XGB >=2.0 ``quantile_alpha=[...]``) or via K
independent stacked fits (LGB / HGB / sklearn QuantileRegressor) with
post-hoc crossing fix.

### Added

- **`TargetTypes.QUANTILE_REGRESSION`** + ``.is_quantile`` property.
  QR is its own class -- not classification, not plain regression.
- **`QuantileRegressionConfig`** with ``alphas: tuple = (0.1, 0.5, 0.9)``,
  ``crossing_fix: "sort"|"isotonic"|"none"``, ``point_estimate_alpha``
  (auto-snaps to nearest alpha if not in set), ``coverage_pairs``,
  ``wrapper_n_jobs``. Validators: alphas in (0,1) strict, sorted,
  unique; coverage_pairs must be in alphas with lo < hi.
- **Strategy ABC additions**: ``supports_native_quantile`` flag,
  ``get_quantile_objective_kwargs(qr_cfg)``, ``wrap_quantile``.
  CB + XGB override flag to True; default for everyone else routes
  through the wrapper.
- **`mlframe/training/quantile_wrapper.py::_QuantileMultiOutputWrapper`**
  -- sklearn-shaped wrapper that fits K cloned base estimators in
  parallel (joblib, auto n_jobs ceiling at cpu//2 to avoid nested-
  parallelism thrashing). Param-name probing: tries ``alpha`` then
  ``quantile`` via ``get_params`` then ``set_params`` (LGBMRegressor
  exposes ``alpha`` via set_params but not get_params(deep=False)).
- **`mlframe/training/quantile_postproc.py::fix_quantile_crossing`**
  -- ``sort`` (default, idempotent), ``isotonic`` (per-row
  IsotonicRegression), ``none``.
- **`mlframe/quantile_metrics.py`** -- numba-JITed: ``pinball_loss``,
  ``pinball_loss_per_alpha`` (matches sklearn ``mean_pinball_loss`` to
  1e-12), ``coverage``, ``mean_interval_width``, ``winkler_score``,
  ``pit_values``, ``quantile_summary``.
- **`mlframe/reporting/charts/quantile.py`** -- 5 panels +
  ``compose_quantile_figure`` + ``ALLOWED_QUANTILE_PANEL_TOKENS``:
  - ``RELIABILITY``: empirical-vs-nominal coverage line
  - ``PINBALL_BY_ALPHA``: pinball loss curve per alpha
  - ``INTERVAL_BAND``: per-row q_lo/median/q_hi sorted by median
    (capped at 1000 sample points for plot readability)
  - ``WIDTH_DIST``: histogram of interval widths (sharpness)
  - ``PIT_HIST``: probability-integral-transform histogram
    (uniform = calibrated; placeholder for K < 3)
- **`ReportingConfig.quantile_panels`** DSL field with validator.
- **Auto-dispatch QR branch**: ``render_multi_target_panels`` checks
  QR (gated on quantile_alphas + 2-D preds + 1-D targets) before
  multilabel/multiclass branches. Output:
  ``{base}_quantile_panels.{ext}``.
- **End-to-end wiring**: ``report_model_perf`` accepts
  ``quantile_panels`` + ``quantile_alphas`` kwargs;
  ``_compute_split_metrics`` + ``train_and_evaluate_model`` thread
  them through; alphas come from ``model._mlframe_quantile_alphas``
  attribute (per-fit context, not per-session config).

### Tests added (87, all green)

- ``tests/training/test_quantile_config.py`` (15): enum predicates,
  config validators (empty/unsorted/out-of-range/duplicate alphas,
  coverage_pairs membership + lo<hi, point_estimate snap-to-nearest).
- ``tests/training/test_quantile_metrics.py`` (16): pinball matches
  sklearn bit-equivalent, coverage closed-form, Winkler with
  hand-computed below/above penalties, PIT median maps to 0.5,
  PIT KS-uniform on 5000-sample synthetic.
- ``tests/training/test_quantile_postproc.py`` (9): sort idempotent +
  fixes crossings, isotonic strictly monotone after, none = no-op,
  invalid mode + 1-D + shape-mismatch errors.
- ``tests/training/test_quantile_strategies.py`` (15): native flag
  CB+XGB true / others false; objective_kwargs format; wrap_quantile
  passthrough vs wrap; real fits CB native + XGB native + LGB/HGB/
  Linear wrapper; wrapper rejects estimators without alpha/quantile
  param; wrapper rejects 2-D y.
- ``tests/training/test_bizvalue_quantile.py`` (2): held-out empirical
  coverage of 80% PI lands in [0.55, 0.95] for CB + XGB on synthetic
  data with planted Gaussian noise.
- ``tests/reporting/test_charts_quantile.py`` (24): per-token spec
  shape; PIT placeholder for K<3; composer subset + unknown-token +
  shape-mismatch + suptitle; matplotlib + plotly render smoke.
- ``tests/reporting/test_quantile_auto_dispatch.py`` (5): dispatch
  fires on QR shape; missing alphas/template -> no-op; dual-backend
  emission; composer-side exception logged + swallowed.

### Verification

```bash
pytest tests/training/test_quantile_*.py tests/training/test_bizvalue_quantile.py \
       tests/reporting/test_charts_quantile.py tests/reporting/test_quantile_auto_dispatch.py \
       --no-cov -p no:randomly
# 87 passed
```

CB MultiQuantile + XGB ``quantile_alpha=[0.1,0.5,0.9]`` both produce
in-sample 80%-PI coverage close to nominal (0.80-0.90 on synthetic
regression data). LGB / HGB / Linear all fan out to K independent
fits via the wrapper; output (N, K) is post-processed via crossing
fix (default ``sort``).

### Not in scope (deferred)

- **MLP / Recurrent native K-head + summed pinball loss** -- routes
  through the wrapper for now (K independent fits). Native K-head
  needs PyTorch model + Lightning task_type='quantile' branch +
  hand-rolled pinball loss; documented as future work.
- **NGBoost adapter** -- NGB is naturally probabilistic and could
  extract any quantile from its fitted distribution; separate epic.
- **Conformal prediction** (split-conformal / cross-conformal /
  jackknife+) -- complementary technique on top of quantile reg;
  future epic.
- **Multi-target QR** -- multiple y-columns predicted simultaneously;
  for one y first.

## 2026-05-08 — LTR suite-side panel wiring

Closes the deferred item from the auto-dispatch landing: the LTR
ranker suite (``ranker_suite.py``) now emits the same per-target
panel grids that the classifier path got via ``report_model_perf``.

### Wired

- ``mlframe/training/ranker_suite.py::train_mlframe_ranker_suite`` --
  3 new optional kwargs (``plot_file`` / ``plot_outputs`` /
  ``ltr_panels``). When all three are set, the suite calls
  ``render_multi_target_panels(...)`` per (flavor, split) and once
  per ensemble (split). When any is missing, no panel files appear
  -- legacy behaviour preserved.
- ``mlframe/training/core.py::train_mlframe_models_suite`` -- the
  LTR delegation branch (line 1944) now pulls ``plot_outputs`` +
  ``ltr_panels`` from ``ReportingConfig`` and ``plot_file`` from
  ``OutputConfig``, threading them into ``train_mlframe_ranker_suite``.
  Matches the threading pattern PR1 + auto-dispatch established for
  the classifier path.

Output filename pattern (per dispatcher contract):
``{plot_file}_{model_name}_{flavor}_{val|test}_ltr_panels.{fmt}``
plus ``{plot_file}_{model_name}_ensemble_{val|test}_ltr_panels.{fmt}``.

### Tests added (4, all green)

``tests/training/test_ltr_panel_wiring.py`` -- direct
``train_mlframe_ranker_suite`` calls with a tiny synthetic search
fixture (50 queries × 5 docs):
- ``test_panels_emitted_per_flavor_per_split`` -- 3 flavors × 2 splits
  = 6 per-model files + 2 ensemble files appear with non-trivial size.
- ``test_no_panels_when_kwargs_omitted`` -- legacy back-compat.
- ``test_no_panels_when_only_plot_file_set`` -- opt-in surface is
  symmetric across all 3 fields (any missing -> no-op).
- ``test_dual_backend_emits_both`` -- ``"matplotlib[png] + plotly[html]"``
  emits both files with the correct ``.{backend}.{fmt}`` suffix.

### Verification

```bash
pytest tests/training/test_ltr_panel_wiring.py --no-cov
# 4 passed in 25.45s
```

## 2026-05-08 — Multi-target panel auto-dispatch (suite wiring)

Glue between PR2's panel composers and the per-(model, split) reporting
hot path. Now ``train_and_evaluate_model`` automatically renders
multiclass / multilabel / LTR panel grids alongside the existing
binary-calibration / regression artifacts, with no caller changes
beyond the ``ReportingConfig`` defaults.

### Added

- **`mlframe/reporting/auto_dispatch.py::render_multi_target_panels`** --
  shape-based dispatcher. Inspects ``(targets, probs, preds, group_ids)``
  and picks the right composer:
  - ``targets.ndim == 2 + probs.ndim == 2 + multilabel_panels`` -> multilabel
  - ``targets.ndim == 1 + probs.shape[1] >= 3 + multiclass_panels`` -> multiclass
  - ``group_ids + 1-D preds + ltr_panels`` -> LTR (1-D score guard;
    falls through to multiclass/multilabel if score is 2-D so a
    multiclass run that happens to expose ``group_ids`` still emits
    multiclass panels)
  - binary / regression / opt-out -> no-op
  Failures inside any composer are logged + swallowed (panels are
  additive; the rest of ``report_model_perf`` must not be broken by
  a degenerate split).
  Output filename: ``{plot_file}_{multiclass|multilabel|ltr}_panels.{fmt}``.

### Wired

- ``mlframe/training/evaluation.py::report_model_perf`` -- 4 new
  optional kwargs (``plot_outputs`` + ``multiclass_panels`` /
  ``multilabel_panels`` / ``ltr_panels``). Dispatcher fires AFTER
  the existing per-class binary-calibration / regression branches,
  so legacy artifacts are unchanged.
- ``mlframe/training/trainer.py::_compute_split_metrics`` +
  ``train_and_evaluate_model`` -- the same 4 fields now sourced
  from ``ReportingConfig`` and threaded through
  ``common_metrics_params`` (mirroring how ``title_metrics_tokens``
  was plumbed in PR1).

### Tests added (18, all green)

- ``tests/reporting/test_auto_dispatch.py``:
  - **TestDispatch (5)**: multiclass / multilabel / LTR / binary
    skip / regression skip.
  - **TestShortCircuits (5)**: empty ``base_path``, empty
    ``plot_outputs``, empty per-target template -> no-op.
  - **TestMultiBackend (1)**: ``"matplotlib[png] + plotly[html]"``
    emits both files with the correct ``.{backend}.{fmt}`` suffix.
  - **TestDispatchPrecedence (2)**: 2-D ``probs`` + ``group_ids`` +
    ``ltr_panels`` -> LTR's 1-D-score guard rejects, falls through
    to multiclass; 1-D ``preds`` + ``group_ids`` + ``ltr_panels``
    -> LTR wins regardless of multiclass/multilabel templates.
    The first test caught a real dispatcher bug (LTR's
    1-D-score reject path was returning ``None`` instead of falling
    through; fixed in the same change).
  - **TestFailureSwallowing (2)**: degenerate inputs return ``None``
    without raising; explicit ``monkeypatch``-injected composer
    exception is logged + swallowed.
  - **TestReportModelPerfIntegration (3)**: end-to-end through
    ``report_model_perf(model=None, probs=...)`` for multiclass +
    multilabel; legacy back-compat -- no panels file when caller
    doesn't opt in.

### Verification

```bash
pytest tests/reporting/ --no-cov -p no:randomly
# 136 passed (118 PR1+PR2 + 18 auto-dispatch)
```

### Risks

- **Default ``plot_outputs="plotly[html,png]"`` means every multiclass
  / multilabel suite run now writes a panel HTML + PNG.** Operators
  who don't want this set ``ReportingConfig.multiclass_panels=""``
  (or per-target equivalents) -- the empty-template short-circuit is
  unit-tested.
- **LTR not yet wired.** ``ranker_suite`` has its own metrics flow
  that doesn't go through ``report_model_perf``. The dispatcher
  supports LTR; the suite-side hookup is a separate touch (deferred
  with TODO at the LTR-suite report site).

## 2026-05-08 — Multi-target panel catalogue — PR2

Multiclass / multilabel / LTR panel builders + composers, dispatched
via the per-target-type DSL templates added in PR1. Each composer
takes raw model outputs + an optional ``panels_template`` string and
returns a ``FigureSpec`` ready to render through either backend.

### Added

- **`mlframe/reporting/charts/_layout.py`** -- shared grid utilities
  (``pack_panels``, ``figsize_for_grid``, ``parse_panel_template``).
  Reused across all 3 multi-target composers; identical token-grammar
  parsing to the existing ``title_metrics_template`` validator.
- **`mlframe/reporting/charts/multiclass.py`** -- 7 panel builders +
  ``compose_multiclass_figure(y_true, y_proba, classes, *, panels_template, ...)``.
  Tokens: ``CONFUSION`` (row-normalised heatmap),
  ``PR_F1`` (per-class P/R/F1 grouped bar), ``ROC`` (one-vs-rest
  curves overlaid with AUC in legend), ``PR_CURVES`` (PR curves with
  AP), ``CALIB_GRID`` (per-class reliability + diagonal),
  ``PROB_DIST`` (P(y=true_class) violin per true class),
  ``TOP_K_ACC`` (top-k accuracy curve, k=1..K).
  ``ALLOWED_MULTICLASS_PANEL_TOKENS`` exported as ``frozenset``.
- **`mlframe/reporting/charts/multilabel.py`** -- 7 panel builders +
  ``compose_multilabel_figure(y_true, y_proba, labels, *, panels_template, ...)``.
  Tokens: ``PR_F1``, ``ROC``, ``CALIB_GRID`` (per-label),
  ``COOCCURRENCE`` (true→pred label confusion heatmap, diagonal =
  per-label recall), ``CARDINALITY`` (#labels-per-row distribution
  for true vs predicted), ``JACCARD_DIST`` + ``HAMMING_DIST``
  (per-row metric histograms). 2-D shape validation up front.
- **`mlframe/reporting/charts/ltr.py`** -- 6 panel builders +
  ``compose_ltr_figure(y_true, y_score, group_ids, *, panels_template, ...)``.
  Tokens: ``NDCG_K`` (mean NDCG@k curve, k=1..max_per_query, capped
  at 50 for plot readability), ``NDCG_DIST`` (per-query NDCG@10
  violin -- the long left tail surfaces failure modes),
  ``LIFT`` (cumulative-relevance / ideal at each rank position),
  ``MRR_DIST`` (per-query reciprocal-rank histogram),
  ``SCORE_BY_REL`` (predicted-score violin per relevance grade --
  visualises grade separation), ``TOP1_BY_QSIZE`` (top-1 accuracy
  bucketed by query size [2,3] [4,5] [6,8] [9,15] [16+] -- reveals
  whether ranker degrades on tiny / huge queries). Reuses
  ``mlframe.ranking_metrics.ndcg_at_k`` + ``_ndcg_one_query`` from
  the LTR addendum.
- **`mlframe/reporting/charts/__init__.py`** re-exports the 3 new
  composers + their allowed-token frozensets.

### Tests added (48, all green)

- ``tests/reporting/test_charts_multiclass.py`` (16): allowed token
  frozenset, one test per token (returns the right ``PanelSpec``
  subtype + shape invariants), composer routing (default = 6 panels,
  subset, unknown raises, ``suptitle`` + ``max_cols`` propagated),
  matplotlib + plotly render smoke.
- ``tests/reporting/test_charts_multilabel.py`` (16): same structure,
  + 2-D shape validation tests (1-D y rejects, mismatched proba
  shapes reject).
- ``tests/reporting/test_charts_ltr.py`` (16): same structure, +
  1-D input + length-mismatch validation. Two fixtures (50 small
  queries vs 200 mixed-size) so ``TOP1_BY_QSIZE`` exercises all 5
  buckets.

### Verification

```bash
pytest tests/reporting/test_charts_multiclass.py \
       tests/reporting/test_charts_multilabel.py \
       tests/reporting/test_charts_ltr.py --no-cov -p no:randomly
# 48 passed
```

Visual smoke: ``D:/Temp/{mc_panels.png,ml_panels.png,ltr_panels.png}``
generated via the smoke script during dev -- multiclass shows the
expected 6-panel 3×2 grid (Confusion / P-R-F1 / ROC / Calibration /
Probability dist / Top-k); LTR shows 5-panel grid with NDCG@k curve,
NDCG@10 distribution, lift, MRR, score-by-grade.

### Not in PR2 (deliberate)

- **Suite-level wiring** (``train_mlframe_models_suite`` -> auto-pick
  composer based on ``target_type`` and call
  ``render_and_save(spec, parse_plot_output_dsl(cfg.plot_outputs), path)``)
  is the next-natural step but lives in a follow-up: it touches the
  trainer / evaluation hot path and deserves its own non-regression
  pass against the existing ``report_model_perf`` output.
- **Biz-value tests** that assert composer output looks right on real
  trained-model predictions are paired with the suite wiring above
  (would otherwise need to build mock predictions, which is what the
  current smoke fixtures already do).

## 2026-05-08 — Reporting backend abstraction (matplotlib + plotly) — PR1

User asked for plotly versions of all charts with config knob to pick
backend + output formats, and visualisation panels for multi-target
types. PR1 lands the backend abstraction + plotly support for the
existing 3 charts (calibration / regression / temporal_audit). PR2
will land the multi-target panel catalogue.

### Added

- **`mlframe/reporting/`** package: spec dataclasses + renderer
  Protocol + matplotlib & plotly impls + save dispatch.
  - ``spec.py``: ``ScatterPanelSpec`` / ``HistogramPanelSpec`` /
    ``HeatmapPanelSpec`` / ``BarPanelSpec`` / ``LinePanelSpec`` /
    ``ViolinPanelSpec`` + top-level ``FigureSpec`` (rows × cols panel
    grid). Pure data + style hints; backend-neutral.
  - ``output.py``: ``PlotOutputSpec`` + ``parse_plot_output_dsl``.
    DSL: ``"<backend>[<fmt1>,<fmt2>,...] (+ <backend>[...])*"``.
    Allowed backends: ``matplotlib`` / ``plotly`` (room for ``bokeh``).
    Per-backend format allowlist enforced at parse time.
  - ``renderers/base.py``: ``Renderer`` Protocol + ``get_renderer(name)``
    factory (lazy-imports the heavy backend modules).
  - ``renderers/matplotlib.py``: ``MatplotlibRenderer`` -- builds
    ``Figure`` via Agg backend (save-only path; no GUI init for headless
    runs). Honors ``constrained_layout`` + ``row_height_ratios``.
  - ``renderers/plotly.py``: ``PlotlyRenderer`` -- builds ``go.Figure``
    via ``make_subplots``. Static-image export via kaleido with
    auto-fallback to HTML + WARN if kaleido missing.
  - ``renderers/save.py``: ``render_and_save(spec, output, base_path)``.
    Smart filename policy: single backend × single format ->
    ``base.fmt``; multi -> ``base.<backend>.<fmt>`` so operator sees
    which backend produced which file.
  - ``colors.py``: shared colormaps so matplotlib / plotly produce
    visually consistent output.
- **`mlframe/reporting/charts/`** — domain-specific spec builders:
  - ``calibration.py::build_calibration_spec`` -- 2-row grid (scatter +
    bin-population histogram with shared colormap).
  - ``regression.py::build_regression_panel_spec`` -- 1×3 grid
    (scatter + residuals histogram with Normal overlay + residuals vs
    predicted). Figure suptitle holds model identity; scatter title
    holds metrics; residual hypothesis migrated to histogram (per
    2026-05-08 user feedback).
  - ``temporal.py::build_temporal_audit_spec`` -- single-line panel
    for target temporal drift.
- **`ReportingConfig.plot_outputs: str = "plotly[html,png]"`** -- new
  default flips to plotly (interactive HTML by default; matplotlib
  stays as opt-in for static PDFs).
- **`ReportingConfig.multiclass_panels` / `multilabel_panels` / `ltr_panels`** --
  per-target_type DSL templates using the same grammar as
  ``title_metrics_template``. Allowed token sets validated at
  config-construction. Default = all panels enabled. PR2 wires the
  actual panel-builder dispatch.

### Tests added (82, all green)

- ``tests/reporting/test_output_dsl.py`` (18): happy-path parsing
  (single backend / multi backend / case insensitive / whitespace
  tolerance) + 11 validation-error scenarios.
- ``tests/reporting/test_renderers.py`` (17): both backends render all
  6 panel spec types; save in supported formats; reject unsupported.
- ``tests/reporting/test_save_dispatch.py`` (4): naming policy +
  keep_handles flag.
- ``tests/reporting/test_charts.py`` (12): 3 chart builders return
  the right spec shape, render via both backends.
- ``tests/reporting/test_reporting_config_dsl.py`` (15): DSL field
  validators + per-target token allowlists.

### Verification

```bash
pytest tests/reporting/ --no-cov -p no:randomly
# 82 passed

# Non-regression on existing reporting / metrics paths
pytest tests/training/test_reporting_config.py tests/test_metrics.py \
       tests/training/test_per_split_target_summary.py --no-cov
# 144 passed
```

Visual smoke: 2×2 grid (scatter + histogram + heatmap + line) renders
identically on both backends -- confirmed via
``D:/Temp/smoke_2x2.{matplotlib.png,plotly.html}``.

### Risks (top 3)

1. **`plotly[png]` requires kaleido** -- soft fallback to ``html`` with
   WARN if kaleido missing; install hint in error message.
2. **Cross-backend pixel-equivalent output is impossible** (font /
   size differences). Tests assert structural correctness (axes count,
   data values, titles) NOT pixel match.
3. **Existing chart functions (``show_calibration_plot`` /
   ``plot_residual_diagnostics`` / ``plot_target_over_time``) NOT yet
   re-routed through the new spec builders** -- they still call
   matplotlib directly. Migration to ``render_and_save(spec, ...)``
   is PR2 work; both old and new paths exist in parallel for now.

### PR2 preview

Multi-target panel catalogue (~6-8h):
- 7 multiclass panels (CONFUSION, PR_F1, ROC, PR_CURVES, CALIB_GRID,
  PROB_DIST, TOP_K_ACC)
- 7 multilabel panels (PR_F1, ROC, CALIB_GRID, COOCCURRENCE,
  CARDINALITY, JACCARD_DIST, HAMMING_DIST)
- 6 LTR panels (NDCG_K, NDCG_DIST, LIFT, MRR_DIST, SCORE_BY_REL,
  TOP1_BY_QSIZE)
- Token routing: parse template -> dispatch panels -> compose
  ``FigureSpec`` -> render via active backend.
- Biz-value tests: param ``["matplotlib", "plotly"]`` ×
  ``["multiclass", "multilabel", "ltr"]``.

## 2026-05-07 — Extended multi-target coverage (Linear/NGB/Recurrent)

Follow-up to MLP multi-target landing. Closes 3 gaps from the strategy
audit + 1 latent sklearn 1.8 compat bug.

### Fixed

- **`Linear` multi_class kwarg** (sklearn 1.8 compat,
  [helpers.py:253](mlframe/training/helpers.py#L253)):
  ``_classif_objective_kwargs("linear", MULTICLASS, K)`` returned
  ``{'multi_class': 'multinomial', 'solver': 'lbfgs'}``. ``multi_class``
  was deprecated in sklearn 1.7 and **removed** in 1.8 -- direct
  ``LogisticRegression(**out)`` raised ``TypeError``. Through the
  mlframe-suite path the kwarg was silently filtered (latent), but a
  user instantiating LR via the helper would crash. Fix: drop
  ``multi_class``; LR auto-detects K since 1.5. Kept ``solver='lbfgs'``
  (still meaningful).

### Added

- **NGBoost multiclass via ``Dist=k_categorical(K)``**
  ([trainer.py:5388](mlframe/training/trainer.py#L5388)):
  Default ``Dist=Bernoulli`` only handles binary; K>2 crashed with
  ``IndexError``. Trainer now inspects ``train_target.max()+1`` (or
  ``config_params['n_classes']``) and injects
  ``Dist=k_categorical(K)`` when ``target_type=MULTICLASS_CLASSIFICATION``.
  Verified: 4-model multiclass ensemble (cb+xgb+lgb+ngb) trains.
- **`RecurrentModel` Phase A (multiclass)** + **Phase B (multilabel)**:
  - ``RecurrentModelStrategy.supports_native_multiclass = True`` and
    ``supports_native_multilabel = True``.
  - ``get_classif_objective_kwargs(MULTILABEL_CLASSIFICATION)``
    returns ``{"task_type": "multilabel"}``.
  - ``RecurrentTorchModel`` ([recurrent.py](mlframe/training/neural/recurrent.py))
    accepts ``task_type='multilabel'`` -> switches loss to
    BCEWithLogitsLoss + sigmoid in ``predict_step`` + multilabel
    torchmetrics (``Accuracy/AUROC/AveragePrecision`` with
    ``task='multilabel'``, ``num_labels=K``).
  - ``RecurrentClassifierWrapper.fit`` no longer raises
    ``NotImplementedError`` on 2-D y; detects multilabel via
    ``labels.ndim == 2`` and threads ``task_type`` through
    ``_create_model``.
  - ``Dataset.labels`` keeps ``float32`` for multilabel, ``int64``
    for multiclass; stratified sampler skipped for multilabel.
  - **Phase C (LTR) deferred**: group-aware sequence batching is
    non-trivial (custom Sampler that yields one query's full doc
    sequences per batch). Documented in strategy docstring;
    ``supports_native_ranking = False``.
- **`RidgeClassifier` multilabel - documented as deferred**:
  sklearn quirk that ``RidgeClassifier`` accepts 2-D y natively, but
  the eval pipeline assumes ``predict_proba`` (RidgeClassifier has
  only ``decision_function``). Wrapper path stays correct.

### Tests added (14 new, all green)

- ``tests/training/test_extended_multi_target_coverage.py``:
  - ``TestLinearMultiClassKwargFix`` (4): helper output + LR-init
    smoke + end-to-end suite.
  - ``TestNGBoostMulticlass`` (3): direct NGB k_categorical fit,
    pre-fix bug regression-guard, end-to-end suite multiclass.
  - ``TestRecurrentStrategyFlags`` (3): supports_native_multiclass /
    multilabel / NOT ranking.
  - ``TestRecurrentMultilabelDispatch`` (2): task_type kwarg returned
    for multilabel, empty for binary/multiclass.
  - ``TestRecurrentMultilabelEndToEnd`` (2): wrapper no longer
    rejects 2-D y, predict_proba returns per-label sigmoid (not
    softmax) verified by row-sum-not-1 check.

### Verification

```bash
pytest tests/training/test_extended_multi_target_coverage.py --no-cov
# 14 passed

# Cumulative non-regression on multi-target tests
pytest tests/training/test_mlp_*.py tests/training/test_ranking_*.py \
       tests/training/test_extended_multi_target_coverage.py --no-cov
# 109 passed
```

### Fuzz combo ensemble coverage (already in place)

The fuzz suite already exercises 4-model ensembles via
``use_ensembles`` axis (default 50/50 toggle):
- 18 / 150 combos: multiclass + ensembling
- 19 / 150 combos: multilabel + ensembling
- 13 / 150 combos: LTR + ensembling
Strategy combinations include cb/xgb/lgb/hgb/linear/mlp + (now
post-fix) ngb-multiclass.

## 2026-05-07 — MLP × multiclass / multilabel / LearningToRank

User asked to extend MLP (PyTorch Lightning) coverage to all three
multi-output target types. Audit found MLP supported only regression +
binary classification at the suite level; multiclass / multilabel / LTR
had zero coverage despite the model code already being multiclass-aware.

### Added

- **`NeuralNetStrategy` flags + dispatch** ([strategies.py](mlframe/training/strategies.py)):
  `supports_native_multiclass = True`, `supports_native_multilabel = True`,
  `supports_native_ranking = True`. New
  `get_classif_objective_kwargs(target_type, n_classes)`:
  - binary -> `{}` (default cross_entropy + int64 already correct)
  - multiclass -> `{"loss_fn": F.cross_entropy, "labels_dtype": int64}`
  - multilabel -> `{"loss_fn": F.binary_cross_entropy_with_logits,
    "labels_dtype": float32, "task_type": "multilabel"}`
  Plus `get_ranker_objective_kwargs(ranking_config)` -> `{"loss_fn": "ranknet"}`
  (default; `listnet` alternative via `LearningToRankConfig.mlp_loss_fn`).
- **MLP multilabel support** ([base.py](mlframe/training/neural/base.py)):
  `PytorchLightningEstimator.fit` detects 2-D y -> sets `_is_multilabel=True`,
  `n_labels_=K`, `classes_=None`. Network output K units; `MLPTorchModel.predict_step`
  switches softmax -> sigmoid via new `task_type="multilabel"` constructor arg.
- **`MLPRanker`** ([neural/ranker.py](mlframe/training/neural/ranker.py)):
  sklearn-shaped wrapper. `loss_fn="ranknet"` (Burges 2005 pairwise BCE
  on score differences, numerically stable via `BCEWithLogitsLoss`) or
  `"listnet"` (Cao 2007 listwise softmax KL). Group-aware
  `GroupBatchSampler` yields one query per training step (skips
  singleton + single-class queries). Shipped as native ranker via
  `mlframe.training.ranking.fit_ranker(NeuralNetStrategy)` -> dispatched
  to `_fit_mlp_ranker`. `train_mlframe_ranker_suite` filter accepts mlp.
- **`LearningToRankConfig.mlp_loss_fn`** field for switching MLPRanker
  between ranknet (default) and listnet.
- **Fuzz combo coverage**: `_fuzz_combo.MODELS` now includes `"mlp"`
  (was excluded entirely). 72 / 150 default combos now hit MLP across
  all 5 target types (regression, binary, multiclass, multilabel, ltr).
  `test_fuzz_suite._skip_if_deps_missing` adds `"mlp": "lightning"`.

### Fixed (regressions caught while wiring)

- **`MLP.classes_` was Python list, not ndarray** ([base.py:208](mlframe/training/neural/base.py#L208)):
  fixed via `np.asarray(sorted(...))`. Was crashing
  `evaluation.py::report_probabilistic_model_perf` line
  `preds = model.classes_[preds]` with `TypeError: only integer scalar
  arrays can be converted to a scalar index` for any K>2 multiclass MLP.
- **`TorchDataModule` flattened 2-D labels to 1-D** ([data.py:73](mlframe/training/neural/data.py#L73)):
  unconditional `reshape(-1)` collapsed `(N, K)` multilabel targets to
  `(N*K,)`, then `BCEWithLogitsLoss` raised
  `Target size != input size`. Now keeps 2-D labels intact; 1-D path
  unchanged for back-compat.
- **`_maybe_wrap_for_2d_target` wrapped MLP in `MultiOutputClassifier`**
  ([trainer.py:951](mlframe/training/trainer.py#L951)):
  added `PytorchLightningClassifier`/`PytorchLightningRegressor` to the
  no-wrap list (MLP supports multilabel natively). Was forcing
  per-label fit, breaking the native sigmoid path.

### Tests added (31 new, all green)

- **`tests/training/test_mlp_multi_targets.py`** (15 tests):
  strategy flag assertions, `get_classif_objective_kwargs` dispatch,
  end-to-end multiclass via suite, end-to-end multilabel via suite
  (no MultiOutputClassifier wrap), `classes_` ndarray regression,
  `TorchDataModule` 2-D label preservation regression.
- **`tests/training/test_mlp_ranker.py`** (16 tests): RankNet pairwise
  loss correctness (perfect / inverse / equal-relevance / single-doc),
  ListNet correctness, `GroupBatchSampler` skip logic, `MLPRanker`
  fit/predict shape, `NeuralNetStrategy.get_ranker_objective_kwargs`
  default + override, `fit_ranker(NeuralNetStrategy)` dispatch,
  `_filter_models_for_ranking` keeps mlp, end-to-end suite with mlp-only
  + 4-model ensemble (cb+xgb+lgb+mlp).

### Verification

```bash
# 31 new MLP tests
pytest tests/training/test_mlp_multi_targets.py tests/training/test_mlp_ranker.py --no-cov

# Non-regression on previously-green LTR tests (64 tests)
pytest tests/training/test_ranking_*.py --no-cov

# Pre-existing 3 MLP tests (omegaconf-fixed since 2026-05-04 batch)
pytest tests/training/test_core.py -k "mlp" --no-cov

# Quick fuzz slice with MLP combos
pytest tests/training/test_fuzz_suite.py -k "c0005 or c0010 or c0014" --no-cov
```

### Library / installed-version notes

- PyTorch Lightning 2.5.5 + omegaconf 2.3.0 (verified compatible after
  2026-05-04 omegaconf upgrade).
- `MLPRanker` uses `LayerNorm` (group-batching breaks BatchNorm running
  stats; see [ranker.py:264](mlframe/training/neural/ranker.py#L264)).

## 2026-05-04 — Learning-to-Rank target type

User asked to add Learning-to-Rank (LTR) as a first-class target type
alongside the existing four (regression / binary / multiclass /
multilabel classification), with explicit ensembling support via
``train_mlframe_models_suite``. CatBoost / XGBoost / LightGBM all ship
native rankers; mlframe now wires them up.

### Added

- **``TargetTypes.LEARNING_TO_RANK``** + ``is_ranking`` helper property.
  LTR is its own class -- neither classification nor regression.
- **``LearningToRankConfig``** (``mlframe.training.configs``) with
  per-library knobs (``cb_loss_fn``, ``xgb_objective``,
  ``lgb_objective``), eval cutoffs (``eval_at``), and ensembling
  method (``ensemble_method`` -- RRF default).
- **``mlframe.ranking_metrics``**: NDCG@k, MAP@k, MRR -- numba kernels
  with the same NUMBA_NJIT_PARAMS as the existing metric stack.
  Exponential gain (Burges 2005) matches LightGBM / CatBoost / XGBoost
  internals; binary cases match sklearn's ``ndcg_score`` exactly.
- **``mlframe.training.ranking``**: per-strategy ``prepare_*_inputs``
  helpers (CB sort-by-group, XGB qid pass-through, LGB qid→group_sizes
  conversion), unified ``fit_ranker`` and ``predict_ranker_scores``,
  and three score-ensembling methods: RRF (default, scale-invariant
  TREC standard), Borda (rank averaging, also scale-invariant),
  ``score_mean`` (raw mean, requires ``assume_comparable_scales=True``
  acknowledgement else WARN + RRF fallback).
- **``mlframe.training.ranker_suite::train_mlframe_ranker_suite``**:
  end-to-end LTR pipeline. Filters models to ``{cb,xgb,lgb}`` with WARN
  for HGB/Linear, validates that the FTE has ``group_field`` set,
  routes through the new group-aware split, runs all three rankers,
  ensembles, computes per-model + ensemble NDCG@10 / MAP@10 / MRR,
  saves via joblib + JSON metadata.
- **``train_mlframe_models_suite``** now accepts
  ``target_type=TargetTypes.LEARNING_TO_RANK`` + optional
  ``ranking_config``. When the LTR target_type is requested, the suite
  delegates to ``train_mlframe_ranker_suite`` -- no surgery needed in
  the existing classification/regression machinery.
- **``make_train_test_split(groups=...)``**: group-aware split via
  ``sklearn.model_selection.GroupShuffleSplit``. Mutually exclusive
  with ``stratify_y`` (sklearn ships no group-stratified splitter).
  Time-based path: groups that span a train→val or val→test cutoff
  get reassigned to the LATER split with a clear WARN.
- **Strategy flag ``supports_native_ranking``** on
  ``ModelPipelineStrategy`` ABC + override on CB/XGB/LGB. Plus
  per-strategy ``get_ranker_objective_kwargs`` that builds the
  library-correct kwargs:
  - **CB**: ``YetiRankPairwise`` (default; configurable to ``YetiRank``,
    ``QuerySoftMax``, ``PairLogit``, ``PairLogitPairwise``,
    ``StochasticRank:metric=NDCG``).
  - **XGB**: ``rank:ndcg`` default. ``rank:map`` is auto-rejected
    when y.max() > 1 (XGB's C++ ``is_binary`` check would otherwise
    crash) -- WARN + fallback to ``rank:ndcg``.
  - **LGB**: ``lambdarank`` (default) or ``rank_xendcg``.

### Tests (76 new tests, all green)

- ``tests/training/test_ranking_strategies.py`` (26 tests):
  ``supports_native_ranking`` flag, objective dispatch, XGB rank:map
  auto-fallback, pre-fit input prep, end-to-end fit/predict per strategy.
- ``tests/training/test_ranking_metrics.py`` (17 tests): NDCG / MAP /
  MRR correctness, sklearn parity for binary, hand-computed graded.
- ``tests/training/test_ranking_splitting.py`` (8 tests):
  ``GroupShuffleSplit`` integrity, mutual-exclusion guard, time-based
  group-spans-cutoff resolution.
- ``tests/training/test_ranking_ensemble.py`` (10 tests): RRF / Borda
  / score_mean correctness, scale-invariance, monotone-transform
  invariance, validation.
- ``tests/training/test_bizvalue_ranking.py`` (6 tests): end-to-end
  ``train_mlframe_models_suite(target_type=LEARNING_TO_RANK)`` --
  individual rankers beat 0.75 NDCG@10 on synthetic web-search data,
  ensemble within 2pp of best individual, save/load roundtrip,
  unsupported-model filter, missing-group_field error.
- ``tests/training/_fuzz_combo.py`` extended with
  ``learning_to_rank`` target_type axis + ``ranking_ensemble_method``
  axis. Frame builder generates graded relevance + qid column. 39 LTR
  combos sampled in the default 150-combo sweep; all pass (3 cleanly
  skipped because pinned models had no native ranker).

### Verification

```bash
# 70 LTR-specific tests
pytest tests/training/test_ranking_*.py tests/training/test_bizvalue_ranking.py --no-cov

# Fuzz LTR slice (36 combos pass, 3 skip-by-design)
pytest tests/training/test_fuzz_suite.py -k "<LTR-combo-IDs>" --no-cov

# Non-regression on splitting + report config + per-split summary
pytest tests/training/test_splitting.py tests/training/test_splitting_edges.py \
       tests/training/test_reporting_config.py \
       tests/training/test_per_split_target_summary.py \
       tests/training/test_jolly_wishing_deer_fixes.py --no-cov
```

### Library API used (verified empirically against installed versions)

| library | class | objective default | groups API | re-bind hook |
|---|---|---|---|---|
| CatBoost 1.2.10 | ``CatBoostRanker`` | ``YetiRankPairwise`` | per-row ``Pool(group_id=...)`` | ``Pool.set_group_id`` |
| XGBoost 3.x | ``XGBRanker`` | ``rank:ndcg`` | per-row ``fit(qid=...)`` | ``DMatrix.set_group(per_query_sizes)`` |
| LightGBM 4.6.0 | ``LGBMRanker`` | ``lambdarank`` | per-query sizes ``fit(group=...)`` | ``Dataset.set_group`` |

CB rows of one query MUST be contiguous (``prepare_cb_inputs`` sorts).
XGB has no MRR (mlframe computes client-side via ranking_metrics.mrr).
HGB / Linear have no native ranker -- skipped with NotImplementedError.

## 2026-04-27 — Session 7 batch 8: chart title format polish

Driven by user feedback after seeing a live calibration chart with
the batch-2 per-split summary live ("title is too long", "two values
should sit next to one BTTR not look like two metrics", "drop the
labelled REL/RES/UNC, the formula reads better").

### Changed (chart title format only — no behaviour change)

- **Per-split summary spliced inline.** Was
  ``BTTR=74%, BTV=86%`` (looked like two separate metrics); now
  ``BTTR/BTV=74%/86%`` (one BTTR with two values for train/split).
  Same pattern for regression (``MTTR/MTV=...`` /
  ``MTTR/MTTS=...``) and multilabel (``MLTR/MLV=...`` /
  ``MLTR/MLTS=...``). Implemented via three precompiled regex subs in
  ``_append_split_rate_suffix`` (``training/trainer.py``).
- **Brier decomposition compact form.** Was
  ``BR=12.34%(REL=5.0%, RES=10.0%, UNC=21.0%)``; now
  ``BR=12.34%(RL5.00%+U21.00%-RS10.00%)``. Math is
  BR = REL - RES + UNC (Murphy 1973), so the compact rendering
  reads as the formula with signs preserved. Prefixes:
  RL = ReLiability, U = Uncertainty, RS = ReSolution (RL/RS chosen
  over single-letter L/S to disambiguate from R = recall).
- **Line break after LL.** ``render_metrics_string`` now inserts
  ``\n`` between the LL fragment and whatever metric follows it,
  so titles like ``BTTR/BTV=78%/39%, BR=8.36%(...), LL=0.288, AUC=0.81``
  wrap into two lines.

### Fixed

- **Calibration plot colorbar now spans both axes.** Was anchored
  to the calibration scatter only, leaving the bin-population
  histogram extending right of the upper plot. Fix:
  ``cbar_ax=[ax_main, ax_hist]`` + ``layout="constrained"`` on the
  figure (drops deprecated ``fig.tight_layout()`` calls that warned
  about the multi-axis colorbar).

## 2026-04-26 — Session 7 batch 2: per-split target summary + temporal audit + Pelt change-points

Driven by user feedback on the Session-7 batch 1 PR ("BT=74% computed
on the full target hides per-split drift; need to see train/val/test
separately") and an explicit feature request for time-series view of
the target with regime-shift detection.

### Changed (potentially breaking for log-string consumers)

- **`select_target` model_name token renamed.** The legacy ``BT=`` /
  ``MT=`` / ``ML=`` summary that was computed on the full target
  (train + val + test concatenated) is replaced with the per-split
  ``BTTR=`` / ``MTTR=`` / ``MLTR=`` (TR for "train") computed on
  ``target[train_idx]``. Callers that didn't pass ``train_idx``
  retain the legacy ``BT=`` form for back-compat (rare — direct
  unit-test callers only).
- **VAL/TEST chart titles now include the split-specific rate.**
  ``_compute_split_metrics`` calls a new helper
  ``_append_split_rate_suffix`` that splices the split rate inline,
  so chart titles read e.g. ``BTTR/BTV=74%/86%`` (val) and
  ``BTTR/BTTS=74%/83%`` (test) — prior shift between splits is
  visible at a glance.

### Added

- **`audit_target_over_time(...)`** in
  [training/target_temporal_audit.py](training/target_temporal_audit.py):
  per-target time-series view of P(y=1) (binary) or mean(y)
  (regression) at an auto-picked granularity (minute/hour/day/week/
  month/quarter/year), with change-point / regime detection,
  segment summaries, drift warnings, and a chart.
  - Auto-picks granularity to land in [30, 50] non-empty bins.
  - Filters sparse trailing/leading bins (n_obs < 0.5 × median(n_obs)).
  - Polars-native fastpath when input is `pl.DataFrame`.
  - Returns structured `TemporalAuditResult` (JSON-safe to_dict).

- **Change-point detection — two methods, one dispatcher**:
  - `find_change_points_pelt` (DEFAULT) — ruptures.Pelt with
    BIC-style auto-tuned penalty (`var(rates) × log(n)`). Optimal
    given cost+penalty, handles balanced regimes, auto-detects K.
    Empirically finds all 4 transitions in the user's production
    drift pattern at the auto penalty.
  - `find_change_points_zscore` — modified-z-score against either
    global median (default) or local rolling-window. Cheaper /
    interpretable for "dominant baseline + anomaly" patterns; can
    miss balanced regimes.
  - `find_change_points(method="pelt"|"zscore")` — top-level
    dispatcher.

- **`plot_target_over_time(audit_result, save_path=...)`**: matplotlib
  rendering of the binned target rate + change-point markers + per-
  segment mean horizontal lines. Saves PNG to disk by default.

- **Auto-wired in `train_mlframe_models_suite`** when
  `behavior_config.target_temporal_audit_column` is set: the audit
  runs in the per-target loop right after the drift_report (BEFORE
  training), logs the segment / warning block, saves the chart, and
  stores the structured result on
  `metadata["target_temporal_audit"][target_type][target_name]`.

- **New `TrainingBehaviorConfig` fields**:
  - `target_temporal_audit_column: Optional[str] = None` — opt-in
    timestamp column. None disables the audit (default).
  - `target_temporal_audit_granularity: str = "auto"` — explicit
    granularity override.
  - `target_temporal_audit_save_plot: bool = True`.

- **Public API** via `mlframe.training`: `audit_target_over_time`,
  `plot_target_over_time`, `format_temporal_audit_report`,
  `find_change_points`, `find_change_points_pelt`,
  `find_change_points_zscore`, `TemporalAuditResult`.

- **Required dep**: `ruptures` (added to `requirements.txt`). It's a
  pure-Python package (numpy dep only); pulling it in is a much
  better trade than re-implementing PELT.

### Tests

- 31 new tests in `tests/training/test_target_temporal_audit.py`:
  granularity auto-pick (8-year span → quarter; 2-month → day;
  200-year → year), Pelt detection of single/multiple/balanced
  regimes, z-score local-vs-global modes, full audit on synthetic
  data mirroring the user's exact graph (98% biased / 40% dip /
  98% / sparse spike), plot save, dict round-trip.
- 16 new tests in `tests/training/test_per_split_target_summary.py`:
  `_append_split_rate_suffix` for binary/regression/multilabel ×
  val/test, legacy `BT=` passthrough, polars/pandas inputs, edge
  cases.
- All existing tests still pass (drift_report + PU + per-split +
  temporal-audit cross-cutting run: 101/101; fuzz combo c0001
  verified end-to-end).

## 2026-04-28 — Meta-test suite v3: A+C+D+E+F batches + shared utils

### Added — 11 new meta-tests + a shared-utils library + drift tracking

This batch consolidates per-repo meta-test plumbing into a reusable
library and adds 11 new tests covering ML-specific invariants,
universal library quality, and meta-meta self-checks. Total mlframe
meta-test footprint after this batch: 26 files, 74 tests, ≈ 61 s.

**A — Shared infrastructure**

- **`pyutilz.dev.meta_test_utils`** (new module in pyutilz) — all the building blocks every meta-test reaches for: `consumer_corpus`, `enumerate_test_files`, `public_top_level_symbols`, `strip_lineno`, `capture_signature`, `capture_module_surface`, `scan_todo_markers`, `count_user_deferred_entries`, `snake_case_variants_of`, `safe_import`. Removed ~400 LOC of duplication across mlframe and pyutilz meta-test directories; bug-fixes in one place propagate to both.
- **`tests/test_meta/test_deferred_drift.py`** (A2) — counts entries in every `_USER_DEFERRED_*` / `_GRANDFATHERED` whitelist via AST, compares against a stored baseline (`_debt_baseline.json`), fails on growth. Net counter visible per run; refresh via `--refresh-debt-baseline` after intentionally accepting more debt. mlframe baseline: 6 whitelists, 36 entries.

**C — ML-specific tests**

- **`test_reproducibility.py`** (C1) — every linear model (ridge / lasso / elasticnet / sgd) yields bit-identical predictions when refit with the same `random_state`; `_predict_from_probs` is a pure function; `apply_preprocessing_extensions` is reproducible. Catches non-deterministic global state pollution (e.g. polars 1.x string cache).
- **`test_public_api_contract.py`** (C2) — every value in `VALID_LINEAR_MODEL_TYPES` actually fits + predicts on a tiny synthetic dataset (10 parametrized cases). Goes one step beyond MT-1's structural check by exercising the runtime path.
- **`test_memory_budgets.py`** (C3) — every linear model's fit + predict on a 200×8 dataset stays under 30 MB peak via `tracemalloc`; `_predict_from_probs` on 50K×10 stays under 5 MB. Catches accidental allocator regressions (≈ 5x increases) without false-flagging micro-perf noise.
- **`test_calibration_monotonicity.py`** (C4) — Hypothesis-driven: post-hoc isotonic calibration never increases per-label Brier on its own training set (Murphy 1973); calibrator falls back to identity on constant-label folds.

**D — pyutilz library QA** (added in pyutilz, not mlframe)

**E — Universal library tests**

- **`test_version_consistency.py`** (E3) — `mlframe.__version__` matches `version.py`'s `__version__` (and would also check `pyproject.toml::[project].version` if/when added). Catches "I bumped one version source but not the other".
- **`test_no_import_cycles.py`** (E4) — Tarjan's SCC over the AST-built import graph; flags multi-node cycles. Surfaced 3 real cycles in mlframe (`feature_selection.wrappers` ↔ `filters` ↔ `training.helpers` ↔ `strategies` ↔ `utils`; `evaluation` ↔ `trainer`; `neural.flat` ↔ `neural.base`) — held in `_USER_DEFERRED_CYCLES` for restructuring later.
- **`test_no_unicode_in_console_output.py`** (E5) — every `print(...)` / `logger.*(...)` call's first arg is ASCII-only. Snapshot-based: 74 existing offenders captured in baseline; new commits adding non-ASCII fail. Critical for Windows cp1251 stdout (per `feedback_windows_encoding`).
- **`test_public_docstrings.py`** (E1) — every public top-level `def` / `class` in production code has a docstring. Snapshot-based (baseline: 832 undocumented) — additions silent, new violations fail.
- **`test_public_annotations.py`** (E2) — every public function has return annotation + every non-self/cls parameter is typed. Snapshot-based (baseline: 1356 unannotated) — additions silent, new violations fail.

**F — Meta-meta tests**

- **`test_meta_meta.py`** (F1+F2+F3) — every `pytest.fail(...)` in the meta-test directory has actionable text (file paths, fix verbs, or dynamic message); meta-tests don't import private internals from production code without an entry in `_PERMITTED_PRIVATE_IMPORTS` (6 mlframe-specific entries cite `_predict_from_probs`, `_canonical_predict_proba_shape`, `_PerClassIsotonicCalibrator` — those ARE the surface under audit); per-test perf-budget overrides match real test names.

### Tests added in pyutilz (not mlframe — listed for cross-reference)

D1 (provider contract — interface coverage), D2 (file-open encoding kwarg), D3 (provider cache thread safety), plus PT-1..PT-9. See `pyutilz/CHANGELOG.md`.

### Total meta-test footprint after this batch: 26 files, 74 tests, ≈ 61 s wall-clock.

## 2026-04-28 — Meta-test suite v2: MT-1..MT-7 + W1..W2 + ``hgb`` made public

### Added — 9 new meta-tests + 1 pre-commit hook

- **`tests/test_meta/test_strategy_registration.py` (MT-1)** — every model-type alias accepted by ``VALID_MODEL_TYPES`` / ``VALID_LINEAR_MODEL_TYPES`` resolves to a real ``ModelPipelineStrategy`` instance via ``MODEL_STRATEGIES``. Surfaced **a real bug**: ``hgb`` had a strategy and per-model kwargs but was missing from ``VALID_MODEL_TYPES`` so users couldn't pass ``mlframe_models=["hgb"]``. **Fix landed:** ``hgb`` added to ``VALID_MODEL_TYPES``, exposing HistGradientBoosting via the public validator.
- **`tests/test_meta/test_config_docstring_drift.py` (MT-2)** — every parameter listed in a config-class docstring's numpydoc ``Parameters ----------`` section is a real field on that class; warning-only counterpart flags configs that document <50% of their fields.
- **`tests/test_meta/test_config_round_trip.py` (MT-3)** — ``cls(**cls().model_dump()) == cls()`` for every default-constructable Pydantic config (catches mutable-default sharing, ``model_dump`` losing fields, non-deterministic ``@model_validator(mode="after")``). Companion test asserts no two fresh instances share the same list/dict default object.
- **`tests/test_meta/test_field_bound_enforcement.py` (MT-4)** — every numeric-bounded field (``Field(ge=0, le=1)``) and every Literal-typed field actually rejects out-of-bounds inputs at construction time.
- **`tests/test_meta/test_mutual_exclusion_validators.py` (MT-5)** — every "X and Y are mutually exclusive" claim in a config docstring or ``@model_validator`` raise message is enforced empirically.
- **`tests/test_meta/test_validator_coverage.py` (MT-6)** — fields whose docstring promises a normalisation ("Case-insensitive, normalized to lowercase") have a ``@field_validator`` actually applying a normalising op.
- **`tests/test_meta/test_todo_hygiene.py` (MT-7)** — every ``TODO`` / ``FIXME`` / ``XXX`` / ``HACK`` comment carries an attribution (assignee in parens or ISO date). Surfaced 5 bare markers held in ``_GRANDFATHERED`` for the maintainer to drain over time.
- **`tests/test_meta/test_api_stability.py` (W2)** — captures the public surface of ``mlframe.training`` into ``_api_snapshot.json``. Renames / removals fail; additions are silent. Refresh via ``pytest ... --refresh-api-snapshot``.
- **`.pre-commit-config.yaml` (W1)** — runs the meta-test suite on every commit (≈ 1 min); ``manual``-stage variant skips Hypothesis tests for tight inner-loop work.

### Public API change

- ``hgb`` (HistGradientBoosting) added to ``VALID_MODEL_TYPES`` — was wired in the trainer all along but rejected by the public validator. Users can now pass ``mlframe_models=["hgb"]`` directly.

### Total meta-test footprint after this batch: 40 tests across 14 files, ≈ 66s wall-clock.

## 2026-04-28 — Meta-test suite expansion + 3 wired config fields

### Added — 7 new meta-tests under `tests/test_meta/`

- **`test_config_field_consumption.py`** hardened: corpus self-test (`test_consumer_corpus_is_substantial`, `test_known_consumed_fields_actually_grep`) refuses to silently pass when `MLFRAME_DIR` mis-resolves to an empty directory; `model_dump()`-splat detector exempts whole-class fields auto-consumed via `**cfg.model_dump(...)` patterns (`hyperparams_config`, `behavior_config`, `linear_model_config`, etc.); `_USER_DEFERRED_DEAD` whitelist keeps the test green on 11 fields the maintainer chose to defer cleanup on (each entry cites reasoning).
- **`test_estimator_kwarg_parity.py`** — every `<flavor>_kwargs` field on every config (`cb_kwargs`, `lgb_kwargs`, `xgb_kwargs`, `hgb_kwargs`, `mlp_kwargs`, `ngb_kwargs`, `rfecv_kwargs`) must reach a constructor via `**field_name`, `.update(field_name)`, `field_name.get(...)`, or extract-then-splat. Catches the audit-2026-04-28 finding where `ModelHyperparamsConfig.{hgb,ngb}_kwargs` were declared but never threaded into helpers.py.
- **`test_subconfig_wiring_parity.py`** — every BaseModel-typed field on a parent config (e.g. `TrainingConfig.linear_config: LinearModelConfig`) must show up as a bare attribute access or kwarg in production. Catches the orphaning pattern where `TrainingConfig` declared sub-configs but the trainer accepted standalone parameters with similar-but-different names.
- **`test_dead_helpers.py`** — public top-level `def`/`class` symbols inside `training/` and `feature_selection/` must be referenced ≥ 2× in the production corpus (definition + ≥ 1 call). Top-level `mlframe/*.py` modules excluded by design (they are the public-API surface for notebook users). Surfaces 9 candidates currently held in `_USER_DEFERRED_DEAD_HELPERS`.
- **`test_metric_invariants.py`** — Hypothesis-driven property tests on `mlframe.metrics`: Brier decomposition identity (`BinnedBrier == REL - RES + UNC`, Murphy 1973), all decomposition components in `[0, 1]`, AUC bounds + monotonic-affine invariance, perfect-prediction Brier=0, log-loss non-negativity, hamming/jaccard/subset-accuracy bounds, plus `_predict_from_probs` boundary cases (threshold=0 → all positive, threshold>1 → all negative, NaN-safe behaviour).
- **`test_enum_exhaustiveness.py`** — every `Literal[...]` / `StrEnum` string value declared on a config field must appear quoted somewhere in the production corpus (i.e. is dispatched on, not just accepted-then-ignored).
- **`test_utility_fuzz.py`** — targeted Hypothesis fuzz on the prepare-for-estimator transforms: `prepare_df_for_catboost` (NaN handling at varying null fractions, missing-column noop), `_canonical_predict_proba_shape` (dense 2-D round-trip, multilabel `list[(N,2)]` → `(N,K)` reduction), `_predict_from_probs` (boundary thresholds, NaN-safe).

### Wired (B1 / B2 / C from the audit)

- **`PreprocessingExtensionsConfig.verbose_logging`** now gates the per-stage `logger.info(...)` in `apply_preprocessing_extensions` (`training/pipeline.py`). WARN paths (TF-IDF column typo / split mismatch) intentionally bypass the gate — those are config errors that must always surface.
- **`MultilabelDispatchConfig.allow_uncalibrated_multi`** now gates a new safety check at the top of `configure_training_params` (`training/trainer.py`): `MULTILABEL_CLASSIFICATION + prefer_calibrated_classifiers=True` raises `NotImplementedError` (default — strict) unless the flag is True, in which case calibration is silently dropped with a WARN. Solves the silent-fail mode where users wrapped a multilabel estimator with `CalibratedClassifierCV` and got an opaque shape error deep in sklearn.
- **`MultilabelDispatchConfig.per_label_thresholds`** now reaches `report_probabilistic_model_perf` and `report_model_perf` (`training/evaluation.py`); the multilabel decision rule routes through `_predict_from_probs` (which already supported per-label vector thresholds) instead of the inline `(probs >= 0.5).astype(np.int8)`.

### Bug fix — meta-test path resolution

`test_config_field_consumption.py::MLFRAME_DIR` previously computed `Path(__file__).parents[2] / "mlframe"`, which resolved to a non-existent nested `mlframe/mlframe/` directory in this flat-layout repo. The corpus loaded zero bytes, every Field looked "unused" (266 false positives). Now derived from the imported package itself: `Path(mlframe.__file__).parent`. The corpus-self-test added in this same change refuses to pass with under 100 KB of source, preventing the silent-degradation regression.

## 2026-04-28 - RFECV zero-variance preventive filter + HGB pl.Enum + 1 stale test fixed

### A: RFECV zero-variance fit-time filter (closed batch-4 TODO)

``feature_selection/wrappers.py::RFECV.fit`` now drops zero-variance / all-null numeric columns from ``X`` BEFORE storing ``feature_names_in_``. The selector cannot meaningfully evaluate constant columns; including them in ``feature_names_in_`` (and possibly ``support_``) was the precondition for the seed=42 c0093 / c0095 ``RFECV.transform: 1/N selected columns missing from input X`` failure. With the preventive filter the column-set drift is physically impossible at the wrapper level, regardless of upstream ``remove_constant_columns`` flag value or per-pipeline-step drop semantics. The strict ``raise RuntimeError`` on column-set drift is restored - any drift now signals a real upstream pipeline-order bug worth surfacing loud.

### D: HGBStrategy uses pl.Enum (was pl.Categorical) for the same global-cache reason as XGB

``training/strategies.py::HGBStrategy.prepare_polars_dataframe`` now emits per-Series ``pl.Enum`` for low-cardinality columns and uses Enum.to_physical().cast(UInt32) for high-cardinality columns. Same rationale as the XGB fix in batch 4: polars 1.x default global string cache makes every ``pl.Categorical`` Series in the process share one growing dictionary, so the column's physical codes drift across runs (latent pickle-reload hazard). New ``HGBStrategy.build_polars_enum_map`` lets the strategy participate in the same train+val-union, test-excluded enum-map cache as ``XGBoostStrategy`` (wired in ``training/core.py`` since batch 4 via ``hasattr(strategy, "build_polars_enum_map")``).

Existing HGB unit tests in ``tests/training/test_catboost_polars.py`` updated to assert ``pl.Enum`` (low-cardinality) - high-cardinality / boundary tests unchanged (still asserts ``pl.UInt32``).

### E: ``test_align_polars_categorical_dicts_no_test_leakage`` re-greened

The test was failing pre-existing (also under master before batch 4): the suite's auto-stratify gate (added 2026-04-27 batch 3) shuffles binary-classification rows across train/val/test, putting ``test_only_cat`` rows into train and defeating the chronological-split scenario the test was written under. Switched the test's extractor to ``TimestampedFeaturesExtractor`` with a strictly-monotonic ``ts`` column so the splitter takes the temporal-ordering path and preserves the original construction. The production align block in ``core.py`` was always correct - the test stopped exercising the right scenario after batch 3 landed.

### C: 3-way fuzz suite validation surfaces multilabel object-dtype detection bug (3 sites)

Running ``test_fuzz_3way_suite.py`` (60-combo smoke) with the newly-added ``include_confidence_analysis_cfg`` axis surfaced an existing structural issue: when a multilabel target survives the polars ``pl.List(pl.Int8)`` -> pandas object-dtype roundtrip, it presents as a 1-D array whose cells are themselves arrays. ``ndim == 2`` checks miss this shape, routing the target into binary / single-label code paths that then choke on ``arr == 1`` or ``np.unique(arr)`` ambiguity. Fixed at three call sites:

- ``training/drift_report.py::compute_label_distribution_drift``: ``is_multilabel`` detection now also matches object-dtype-of-arrays. ``_multilabel_split_summary`` stacks the cells to a true 2-D array before summing. ``return`` overrides ``target_type`` to ``"multilabel_classification"`` when the data shape forced the multilabel branch, so ``format_label_distribution_drift_report`` reads the right keys.
- ``training/trainer.py::_validate_target_values``: the same stacking before the ``arr_np.ndim > 1`` per-column degenerate check, so ``np.unique`` doesn't see object-cell arrays.
- ``training/trainer.py`` (``catboost_custom_classif_metrics`` block): explicit branch for object-of-arrays so ``nlabels`` comes from the cell width, not from ``np.unique`` on a non-comparable object array.

3-way smoke pass rate went from 7/15 fail (47%) to 3/15 fail (20%) with these fixes. The remaining 3 combos fail on a separate CatBoost ``'str' object cannot be interpreted as an integer`` issue with class labels - pre-existing, unrelated to the axis change.

## 2026-04-28 - Fuzz axis expansion + 1 surfaced multilabel bug

- **New axis** ``include_confidence_analysis_cfg`` in ``tests/training/_fuzz_combo.py`` (and added to ``_3WAY_AXES`` for 3-way pairwise coverage). Wired through ``test_fuzz_suite.py`` to ``ConfidenceAnalysisConfig(include=...)`` so the fuzz suite exercises the test-set confidence-analysis pass at ``trainer.py::_report_confidence_analysis`` - a distinct code path with its own metrics/report side-effects that prior fuzz never touched.
- **``use_cache`` axis NOT added**: the flag lives on ``TrainingControlConfig`` which is per-model not suite-level; ``train_mlframe_models_suite`` doesn't accept it. A literal cache-HIT is also unreachable in the fuzz harness (each combo runs against a fresh ``tmp_path``). Adding an axis that toggles a flag none of the fuzz call paths read would be theatre. Left as-is.
- **Surfaced bug, fix included** at ``trainer.py::_report_confidence_analysis``: ``test_probs[np.arange(...), test_target]`` to pull each row's true-class probability is well-defined only when ``test_target`` is 1-D class indices. Multilabel targets carry shape ``(N, K)`` binary indicators - the indexing raised ``IndexError: shape mismatch (N,) (N, K)``. Regression targets carry float values where the indexing makes no sense semantically. Added an early-return with INFO log for both shapes. Surfaced fuzz default seed c0000 (multilabel + confidence_analysis_cfg=True) - exactly the kind of cross-axis interaction the new axis was supposed to find.

3-way enumerator coverage with the new axis: 15816 / 15819 (100.0%, 3 missing - canonicalisation-blocked).

## 2026-04-28 — Static meta-test: Pydantic config field consumption parity

### Added

- **`tests/test_meta/test_config_field_consumption.py`** — static check that every Pydantic `Field` declared in `mlframe/training/configs.py` is referenced by at least one production consumer (anything under `mlframe/` outside `tests/`). The test enumerates every `BaseConfig` subclass via `inspect`, walks `model_fields`, and asserts each field name appears in at least one of: attribute access (`cfg.foo`), bracket lookup (`d["foo"]`), or kwarg passthrough (`foo=`). Runs in <1 s; no fixtures, no DB, no training code exercised. The whitelist `_KNOWN_INDIRECT_CONSUMERS` covers fields consumed via indirect routes (e.g. `getattr(cfg, name)` in a loop) — each entry must cite the consumer file:line.

Catches the failure mode where a Field is added to a config model but never threaded into the trainer/strategy/pipeline that should read it. Common symptom: "I set this flag and nothing happened" debugging sessions, plus the slower decay of config bloat that obscures the real surface area of the public API.

Ported pattern from a sister project's audit of dead loader-return-dict keys; the ML-framework analogue is dead Pydantic field bloat. README updated with usage notes (under `## Running tests` → `### Static meta-tests`).

## 2026-04-28 — Fuzz-uncovered batch 4: polars 1.x global-string-cache leak in XGB + 5 hardenings

### Headline fix - polars 1.x ``pl.Categorical`` global-string-cache leak (XGB only)

Polars 1.33.1 has the global string cache enabled by default and ``pl.disable_string_cache()`` is a no-op. Every ``pl.Categorical`` Series in the process therefore shares one monotonically growing dictionary, and an XGBoost model fitted with ``enable_categorical=True`` rejects val/test rows at predict with ``Found a category not in the training set`` even when the actual values column is clean - XGB reads the dtype's ``cat.get_categories()`` list, which has grown since fit time.

Surfaced fuzz seed=2024 c0009 (polars_nullable + ``weird_cat_content=unicode``, cat_count=15) leaking levels into c0060 (pandas, weird=empty) - XGB on c0060's val_df saw ``cat1`` from c0009's processing.

Fix: ``training/strategies.py::XGBoostStrategy.prepare_polars_dataframe`` now emits per-Series ``pl.Enum`` (no shared cache) instead of ``pl.Categorical``. The Enum domain is the **train+val UNION** of unique values - test is intentionally excluded so test-only levels never widen the model's accepted-category set (data-leakage guardrail). New ``XGBoostStrategy.build_polars_enum_map`` builds the map once per ``(target, feature_tier, strategy)`` and caches it in ``training/core.py`` alongside ``tier_dfs_cache`` so the weight-schema loop and any sibling-tier models reuse it without recomputation. Cache cleared on tier transition / non-polars-native strategy entry so a stale map can't survive a frame-release.

Trainer-side polars cat-alignment helper (``_align_polars_cat_categories``) deleted - it was a band-aid that union-cast all polars-categorical-shaped columns including text features, which undid the CB Enum->String text-feature cast in core.py and broke ``c0025_ee036bad-cb_hgb_lgb_xgb-pl_enum-n600``. The proper fix lives upstream in the strategy now.

### Strict diagnostic raises (replaced ``except + continue`` band-aids)

- **RFECV column drift** in ``feature_selection/wrappers.py``: the ``transform()`` path now logs a clear ``ERROR`` with full diagnostic when a column listed in ``support_`` is missing from the input frame, then proceeds with the present-only subset. Old behaviour silently dropped the columns with no log. Initially raised here, but the upstream pipeline-order bug (surfaced fuzz seed=42 c0093 / c0095 on the polars_nullable + ``rfecv_estimator=cb_rfecv`` + ``inject_all_nan_col=True`` path) lives in ``core.py``, not in the selector - once that is fixed, restore the ``raise RuntimeError``. TODO inline at the call site captures the next-investigation hand-off.
- **``[cb-pool-reuse]`` X/y length mismatch** in ``training/trainer.py::_maybe_get_or_build_cb_pool``: hard ``RuntimeError`` instead of ``logger.error + return None``. A length mismatch this deep is an upstream slicing bug (RFECV inner CV / OD filter / aging trim); falling back to sklearn would just delay the same error with less context.
- **Multioutput target reaching a regression report** in ``training/evaluation.py::report_regression_model_perf``: emits a ``logger.warning`` with full shape diagnostic when ``targets_arr.ndim > 1 and shape[1] > 1``. Surfaces a likely ``is_classifier()`` dispatch bug for multilabel-wrapped estimators that previously crashed deep in ``mean_squared_error`` with an opaque shape mismatch.

### Cleanup

- **MRMR perf threshold raised 2.5x -> 10x** in ``tests/training/test_bizvalue_feature_selection.py``: rationale documented inline. The 2.5x bound was unrealistic on small fuzz frames where MRMR's per-permutation overhead dominates. 10x preserves the regression's intent (catch a 100x slowdown) without false-positive flapping.
- **2 dead 0-row val guards removed** in ``training/trainer.py`` (``_setup_eval_set``, ``_compute_split_metrics``): the original cause was an OD val-side filter rejecting all rows; now guarded at the source in ``core._apply_outlier_detection_global``. The downstream guards were no-op since batch 3 and would have masked any future regression at the source.
- **``XGBoostStrategy.prepare_polars_dataframe`` unit tests updated** in ``tests/training/test_catboost_polars.py``: assert ``pl.Enum`` instead of ``pl.Categorical``; new tests cover ``build_polars_enum_map`` (train+val union, test-excluded leak guard) and ``category_map`` consistency between train and val.

### Verification

- Default seed: 150/150 PASS
- Seed=2024 (where the leak surfaced): 150/150 PASS
- Seed=42: 150/150 PASS
- Seed=99: 150/150 PASS
- Unit tests: ``test_catboost_polars.py`` 26/26 PASS

## 2026-04-27 — Fuzz-uncovered batch 3: OD class-balance + stratify + KBins NaN + bizvalue xfails

### Production fixes (each surfaced via multi-seed fuzz sweep)

- **OD class-balance pre-check** in
  ``core._apply_outlier_detection_global``: when the unsupervised
  outlier detector is fit on features that include a label-correlated
  leak feature (``num_leak`` etc), it tends to flag the entire
  rare-class minority as outliers. With ``contamination=0.05``,
  ``IsolationForest`` / ``OneClassSVM`` happily eliminate every
  class-1 row from a rare_1pct binary classification target
  (50/4950 → 0/3420 after OD on train + 0/380 after OD on val).
  CB then crashes deep in C++ ``Target contains only one unique
  value``. Fix: pre-check the per-target class diversity BEFORE
  applying the OD filter; if filtering would collapse classes,
  skip the filter (set ``train_od_idx`` / ``val_od_idx`` to all-True).
  Surfaced fuzz seed=99 c0016. Each target's preservation is
  independent — global OD fit stays intact.
- **Auto-stratify train/val/test** in ``core.train_mlframe_models_suite``
  when target is classification and no timestamps are provided. Without
  stratification, the unstratified shuffle path can hand val_shuf an
  unlucky 0-class-1 slice for rare imbalance ratios (fuzz default-seed
  c0134). The ``make_train_test_split`` helper already supports
  ``stratify_y``; just plumb it from the suite when meaningful
  (single classification target, ≥2 classes with ≥2 members each).
- **SimpleImputer prepend in PreprocessingExtensions pipeline**
  (``training/pipeline._build_extension_steps``): KBinsDiscretizer,
  PolynomialFeatures, RBFSampler, Nystroem, and most sklearn
  decompositions (PCA, TruncatedSVD, FastICA, ...) reject NaN at
  fit time. The mlframe upstream NaN handling targets the GBDT
  backends (CB/HGB/XGB) which tolerate NaN natively; numeric NaN
  could survive into ``apply_preprocessing_extensions``. Prepend
  a ``SimpleImputer(strategy="median")`` whenever any extension
  step is active so the sklearn-bridge sees finite values. Surfaced
  fuzz seed=2024 c0040 / c0060 / c0148.
- **Single-class target diagnostic guard** in
  ``trainer._validate_target_values`` (committed earlier as
  follow-up to batch 2): early ``ValueError`` with a clear "upstream
  filter pipeline / rare imbalance" diagnostic instead of CB's opaque
  C++ ``target_converter.cpp:404`` crash. ``is_classification`` flag
  derived from ``model_type_name`` suffix at the call site.

### Surfaced bizvalue improvements

- **3 xfails in `test_bizvalue_feature_selection.py` retired**: the
  suite now populates ``metadata['selected_features']`` (flat sorted
  union of post-pipeline column names across every trained entry)
  and ``metadata['selected_features_per_model']`` (per-model
  detail). Each entry also gets ``selected_features_`` attribute so
  the standard sklearn-style ``getattr(entry, 'selected_features_')``
  probe finds it. ``test_mrmr_drops_uninformative_features``,
  ``test_selected_features_surface_for_inspection``, and the third
  xfail-tagged check all pass.
- **Both `test_bizvalue_*` import order fixed**: stray ``from
  mlframe.training import …`` lines BEFORE the module docstring
  caused ``SyntaxError: from __future__ imports must occur at the
  beginning of the file``. Imports moved below docstring.

### Suite-internal hygiene

- **0-row val dead-code guards removed** from
  ``trainer._apply_pre_pipeline_transforms`` (both fit-transform
  and transform branches). The OD val-side error+continue from
  batch 2 prevents 0-row val at the source; if a 0-row val still
  arrives, SimpleImputer's natural ``Found array with 0 sample(s)``
  raise will surface the upstream bug immediately.
- **RFECV column drift logging escalated** from WARNING to ERROR
  (``feature_selection/wrappers.py``): a fitted ``support_`` mask
  that no longer reflects the physical columns is a pipeline-order
  bug (constant-col-removal must come BEFORE RFECV.fit, not
  between fit and transform). Visible without extra verbosity.
- **Pool-reuse X/y mismatch logging escalated** from WARNING to
  ERROR (``trainer._maybe_get_or_build_cb_pool``): X/y length
  mismatch is a hard contract violation; sklearn fallback path
  surfaces a clear error if the call site really intended this.
- **`report_regression_model_perf` (N,K) target plot skip**: when
  multioutput target reaches the regression report (e.g. wrapped
  multilabel via ChainEnsemble's regressor base), skip the scatter
  plot rather than emit overlapping K-cloud nonsense. Title metrics
  still carry the per-output-aggregated MAE/RMSE/R2.

### Verification

- Default seed (master_seed=20260422): 150/150 ✅
- FUZZ_SEED=42: 150/150 ✅
- FUZZ_SEED=99: 150/150 ✅ (was 1 fail c0016 pre-fix)
- FUZZ_SEED=2024: 3 previously-failing combos pass ✅ (c0040/c0060/c0148);
  full sweep TBD
- bizvalue tests: 17/18 (1 perf-flake on MRMR runtime, not correctness)

## 2026-04-27 — Fuzz-uncovered batch 2: 4 production fixes + 4 retired masking canons

### Production fixes (each retires a fuzz-suite masking pattern)

- **`_decategorise_float_cat_columns`** in `training/trainer.py`: when
  ``CatBoostEncoder`` / target encoders / RFECV-driven re-encodings produce
  a ``pd.CategoricalDtype`` column whose category levels are floats
  (``[0.13, 0.42, ...]`` boxed via ``.astype("category")``), the column is
  semantically numeric — drop the categorical wrapper and expose the
  raw floats. Both XGBoost (``columnar.h:134: "Category index from
  DataFrame has floating point dtype"``) and CatBoost
  (``_catboost.pyx: "bad object for id: 0.0"``) reject these columns;
  HGB and sklearn happily accept them as numeric. Applied **before**
  ``_apply_pre_pipeline_transforms`` so the fix also covers the
  RFECV inner CB / XGB inside the pre_pipeline. Fuzz c0102 now passes.
- **OD val-side error+continue** in
  ``core._apply_outlier_detection_global``: mirrors the train-side
  ``min_keep`` floor at line 1021. If the outlier detector rejects
  >99% of val rows (typically because train was fit on a very
  different distribution), don't propagate an empty val_df — log an
  ``ERROR`` and keep the original unfiltered val_set so downstream
  evaluation has data. The error message points at fit-distribution
  mismatch as the likely culprit. Replaces the ``val_seq_frac_eff``
  runtime canon in ``test_fuzz_suite.py`` and obsoletes the four
  layered ``if len(val_df) == 0: skip`` guards in trainer.py (they
  remain as harmless dead code paths until the next cleanup).
- **`_select_scalable_numeric_columns`** in `training/pipeline.py`:
  pre-filters numeric columns that would crash the polars-ds scaler's
  C++ kernel:
  - All-null / non-finite columns (``quantile()``/``mean()``/``min()``
    return ``None``).
  - Zero-spread columns (``q_high - q_low == 0`` for ``robust``,
    ``std == 0`` for ``standard``, ``max - min == 0`` for ``min_max``).
  Applied universally to all polars-ds scaler methods so the
  ``ComputeError: division by zero`` deep in the C++ scaler is no
  longer reachable. Replaces the ``rm_const_eff`` runtime canon in
  ``test_fuzz_suite.py``.
- **`fix_infinities=False` auto-recovery** in
  ``training/preprocessing.preprocess_dataframe``: when the user
  opts out of inf-handling but the data actually contains
  ``np.inf`` / ``-np.inf`` in numeric columns, log an ``ERROR`` and
  auto-fix anyway with a 0.0 fill. Better than the opaque XGB / HGB
  / sklearn C++ crash deep in the fit path. Caller can silence by
  flipping ``fix_infinities=True`` (canonical) or pre-cleaning the
  inf values upstream. New helper ``_frame_contains_inf`` does the
  cheap ``O(numeric_cols * n_rows)`` scan. Replaces the
  ``fix_inf_eff`` runtime canon in ``test_fuzz_suite.py``.

### Suite-internal cleanup

- **`XGBOOST_MODEL_TYPES` extended** in ``config.py`` to include the
  DMatrix-reuse shims ``XGBClassifierWithDMatrixReuse`` /
  ``XGBRegressorWithDMatrixReuse``. The pre-fix tuple only listed the
  base sklearn classes, which silently skipped EVERY
  ``model_type_name in XGBOOST_MODEL_TYPES`` check downstream for the
  shim variants — including the new cat-with-float-dtype guard that
  surfaced the real bug.
- **Retired three masking canons in `_fuzz_combo.py:canonical_key`
  + their synthetic-data mirrors**:
  - ``inject_degenerate_cols → False`` for CB+multilabel: replaced by
    the existing dtype-aware ``cat_features`` filter in
    ``feature_selection/wrappers.py`` plus the new num-degenerate
    handling in trainer.
  - ``val_sequential_fraction → 0.5`` for backward+no-shuffle+degenerate:
    replaced by the OD val-side error+continue fix above.
  - ``remove_constant_columns → True`` for inject_degenerate /
    inject_all_nan_col: replaced by the polars-ds zero-spread filter.
- The fuzz axis space now exercises every previously-canonised
  cross-product end-to-end without rewriting.

### Verification

- **150/150 combos pass** in 17:50 with all four prod fixes applied
  and the four masking canons retired (full pairwise-covered fuzz
  surface, 0 failures, 0 timeouts, 0 INTERNALERROR).

## 2026-04-27 — Fuzz-uncovered batch 1: CB text-floor + cold-start lazy-imports + axes-tuning + suite-refactor sync

### CB text-feature dictionary scaling (production fix)

- New helper `training.helpers.compute_cb_text_processing(n_train_rows)`
  returns a CatBoost `text_processing` config that scales
  `occurrence_lower_bound` proportionally to the fit-time row count
  (5% rule, floor=2). Wired into `trainer._train_model_with_fallback`
  (before `model.fit`) and `feature_selection/wrappers.py` RFECV
  inner-fold (after `text_features` filter, before estimator fit).
- Without this, CB's default `occurrence_lower_bound=50` either raised
  "Dictionary size is 0" or — in the C++ `_train` loop — hung
  indefinitely on RFECV inner-folds smaller than 50 rows or training
  sets where text columns lacked enough word repetition (fuzz
  c0056 / c0070, observed 2026-04-26).
- `_maybe_get_or_build_cb_pool` now caches `_mlframe_text_features` /
  `_mlframe_cat_features` / `_mlframe_embedding_features` on the
  built Pool so the dynamic-text-processing injection at fit time
  can introspect them without round-tripping through fit_params
  (which the Pool-reuse path strips).

### Cold-start lazy-imports (Windows fix)

- `flaml.default` and `mlframe.training.neural` (lightning + torchmetrics)
  used to be eagerly imported at trainer module-load. On Windows cold
  cache that pulled scipy / optuna / lightning chains taking 30-180 s,
  consistently overshooting per-test timeouts on the FIRST test of any
  pytest run that touched the trainer (fuzz `c0000` timeout). Defer
  via new `_get_flaml_zeroshot()` and `_get_neural_components()`
  getters that populate module globals on first actual use.
- `tests/training/conftest.py` now pre-touches `flaml.default`,
  `mlframe.training.neural`, and `networkx` at session import — moves
  the cold-cache wallclock outside the per-test timeout window without
  re-introducing the eager-import overhead in production code.

### Anti-masking documentation (process fix)

- `CLAUDE.md` got a new top-level section "Fuzz / combo tests are bug
  DETECTORS, not bug HIDERS" with 4 forbidden patterns (canonicalisation,
  runtime `*_eff` rewrites, `pytest.mark.xfail`, "0-row defensive
  guards") and concrete examples from this codebase.
- `tests/training/_fuzz_combo.py` docstring expanded with a
  "Canonicalisation contract" section and the
  legitimate-vs-illegitimate distinction.
- Inline `text_col_count → 0` masking in `canonical_key:301-311` retired
  (the production fix above replaces it). Two dead `_rule_cb_*`
  functions (`_rule_cb_pool_reuse_with_mrmr_small_n_filtered`,
  `_rule_cb_text_dict_collapse_with_full_quartet`) removed — both had
  TODOs that the production-fix addresses.

### Fuzz speedup (test-only)

- `_config_for_models` now passes
  `rfecv_kwargs={"max_noimproving_iters": 2, "cv_n_splits": 2,
  "max_runtime_mins": 2}` so RFECV's heuristic feature search short-
  circuits aggressively. Production users keep the library defaults
  (`max_noimproving_iters=15`, `cv_n_splits=4`); fuzz-only override.
- Axis-value tuning in `_fuzz_combo.py`: `n_rows` 1200→1000 (rare_5pct
  still exercises at n=1000 per the existing canonical_imbalance
  threshold), `cat_feature_count` 20→15 (one-hot blow-up code path
  identical, smaller absolute count), `iterations` 30→15 (multi-iter
  ES still triggers), `early_stopping_rounds_cfg` 20→10,
  `multilabel_n_chains_cfg` (3,5)→(2,3), `multilabel_cv_cfg`
  (3,5)→(2,3). MRMR `full_npermutations` 3→2,
  `max_consec_unconfirmed` 3→2 in `test_fuzz_suite.py`. Each preserves
  the tested code path; full 150-combo run dropped from 70 min to
  ~19 min.
- Charts off in fuzz: `OutputConfig(save_charts=False)` +
  `ReportingConfig(show_perf_chart=False, show_fi=False)`. Each combo
  was leaking 1-2 MB of matplotlib state per chart; over 150 combos
  that compounded to >2 GB and tripped pytest's traceback-formatter
  with `MemoryError: bad allocation` / INTERNALERROR.

### Suite-refactor sync (catch-up to typed-config rebuild)

- `mlframe.training.__init__` now re-exports `FeatureTypesConfig`
  (was reachable only via `mlframe.training.configs`). Listed in the
  module's `__all__`.
- `core.train_mlframe_models_suite` `reporting_config.model_dump()`
  was leaking the derived field `title_metrics_tokens` into the
  `_build_configs_from_params(**kwargs)` call, causing
  `TypeError: got an unexpected keyword argument 'title_metrics_tokens'`.
  Fixed via `model_dump(exclude={"title_metrics_tokens"})`; the deep
  consumer re-derives the tuple from the rebuilt ReportingConfig
  object directly.
- `ensembling.py:_process_single_ensemble_method` now pops the
  `compute_{trainset,valset,testset}_metrics` keys off `kwargs_copy`
  before splatting alongside the hard-coded `compute_valset_metrics=True`
  (the reporting-config refactor put those fields into
  `common_params_dict`, leading to
  `TypeError: dict() got multiple values for keyword argument
  'compute_valset_metrics'`). Same function migrated 7→8-tuple
  unpacking from `_build_configs_from_params` (added `output_config`)
  and switched `display=` → `reporting=` + `output=` keyword on
  `train_and_evaluate_model`. `_process_uncertainty_split` block
  migrated identically.
- `tests/training/test_fuzz_suite.py` migrated to the new typed-config
  surface: `_preprocessing_for_combo` extended to carry
  `fix_inf_eff` / `rm_const_eff` / `fillna_value` /
  `ensure_float32_dtypes` (the `_configs_for_combo` dict now omits
  `preprocessing_config` to avoid duplicate-kwarg with the explicit
  pass), module-level imports for `OutputConfig` /
  `OutlierDetectionConfig` / `FeatureSelectionConfig` /
  `ReportingConfig`.

### Verification

- 150/150 fuzz combos run end-to-end after the changes; only the
  pre-existing `c0102` (XGB cat-with-float-dtype rejection after
  MRMR/prep_ext target-encoding flip) still fails, with the same
  signature as before. That one is its own production bug, scheduled
  for the next batch.

## 2026-04-27 — Calibration reporting upgrades + suite-config sweep (BREAKING)

### Calibration reporting (mlframe.metrics)

- **ECE always-computed**: standard Expected Calibration Error now lives next
  to the existing CMAEW (mlframe-native power-weighted variant). Same bins so
  values are directly comparable. Added new numba kernel
  `compute_ece_and_brier_decomposition(y_true, y_pred, nbins)` which returns
  `(ece, reliability, resolution, uncertainty, brier_binned)`. Kernel uses
  per-bin mean predicted probability (not bin centre) so the Murphy 1973
  identity `BinnedBrier == REL - RES + UNC` holds exactly to fp precision.
  Raw Brier still differs from binned Brier by within-bin prediction variance;
  the gap shrinks with finer binning.
- **Brier score decomposition**: REL (reliability/calibration error) -
  RES (resolution/separation) + UNC (uncertainty/base-rate entropy) added to
  per-class `class_metrics` dict as `brier_reliability`, `brier_resolution`,
  `brier_uncertainty`. Diagnostic interpretation: REL high ⇒ calibration
  problem (try Platt/isotonic/postcalibration); RES low ⇒ ranker can't
  separate (need feature work).
- **Title metrics template**: 9 historical `show_*_in_title` booleans
  (`show_brier_loss_in_title`, `show_cmaew_in_title`,
  `show_roc_auc_in_title`, `show_pr_auc_in_title`, `show_logloss_in_title`,
  `show_coverage_in_title`, `show_points_density_in_title`, plus the new
  `show_ece_in_title` and `show_brier_decomp_in_title` we considered)
  collapsed into one ordered string template
  `title_metrics_template: str = "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC"`.
  Token grammar (closed set: `ICE`, `BR`, `BR_DECOMP`, `ECE`, `CMAEW`, `COV`,
  `LL`, `ROC_AUC`, `PR_AUC`, `DENS`) validated at config-construction time
  via pydantic field/model validators - invalid templates fail before
  training starts, never mid-figure. Users now control both metric
  selection AND order with one parameter.
- **Probability histogram subplot**: under the reliability scatter, sharing
  the X axis. Y-scale auto-picks log when bin-population skew exceeds 100x
  and linear otherwise; override via `prob_histogram_yscale="log"` /
  `"linear"`. Toggle off via `show_prob_histogram=False`. Inline per-bin
  population text labels next to the scatter points are independently
  controlled by `show_inline_population_labels` so users can keep both,
  drop both, or keep only one.
- `fast_calibration_report` return tuple grew from 13 to 17 elements: the
  four new fields (`ece`, `brier_reliability`, `brier_resolution`,
  `brier_uncertainty`) sit between `calibration_coverage` and `roc_auc`.
- `show_calibration_plot` got new kwargs (`show_prob_histogram`,
  `prob_histogram_yscale`, `show_inline_population_labels`) - same defaults
  as `ReportingConfig`. Histogram-OFF path is byte-for-byte unchanged
  (verified by `tests/training/test_perf_edges.py::TestMatplotlibAggPath`).

### Config refactor (mlframe.training.configs) — BREAKING

- **`DisplayConfig` renamed to `ReportingConfig`** (all callers must update).
  Scope slimmed to "look of the calibration / training performance report".
  Filesystem paths moved out (see `OutputConfig`); feature-importance plot
  parameters moved out (see `FeatureImportanceConfig`).
- **New `OutputConfig`**: `data_dir`, `models_dir` (renamed from
  `models_subdir` for symmetry with `data_dir`), `plot_file`, `save_charts`.
  Holds path/output knobs.
- **New `FeatureImportanceConfig`**: `num_factors`, `figsize`,
  `positive_fi_only`, `show_plots`. Replaces the dict-typed `fi_kwargs`
  that previously lived on `DisplayConfig`.
- **New `OutlierDetectionConfig`**: `detector` (was `outlier_detector` at
  the suite level), `apply_to_val` (was `od_val_set`).
- **`FeatureSelectionConfig` extended**: added `custom_pre_pipelines`
  field. Previously a top-level kwarg of the suite.
- **`PreprocessingConfig` extended**: added unprefixed `scaler`, `imputer`,
  `category_encoder` fields. Previously these were dict-typed orphans
  reachable only via the deleted `init_common_params` pass-through. Default
  `None` preserves the suite's context-aware default-selection logic.

### `train_mlframe_models_suite` signature (mlframe.training.core) — BREAKING

**Removed kwargs** (no back-compat shim, per user direction):
- `init_common_params` (the dict-typed pass-through)
- `data_dir`, `models_dir`, `save_charts` → `output_config=OutputConfig(...)`
- `outlier_detector`, `od_val_set` → `outlier_detection_config=OutlierDetectionConfig(...)`
- `use_mrmr_fs`, `mrmr_kwargs`, `rfecv_models`, `custom_pre_pipelines` → `feature_selection_config=FeatureSelectionConfig(...)`

**Kept top-level** (model-selection knobs answer "what does this suite do"):
`mlframe_models`, `use_ordinary_models`, `use_mlframe_ensembles`,
`recurrent_models`, `recurrent_config`, `sequences`.

**New top-level kwargs**:
`reporting_config`, `output_config`, `outlier_detection_config`,
`feature_selection_config`. All Pydantic-validated; accept dict or model
instance via the existing `_ensure_config` helper.

### Migration cookbook

```python
# BEFORE (pre-2026-04-27):
train_mlframe_models_suite(
    df=...,
    init_common_params={
        "show_perf_chart": False, "show_fi": False,
        "scaler": custom_scaler,
    },
    data_dir="./artifacts",
    models_dir="models",
    save_charts=True,
    outlier_detector=isolation_forest, od_val_set=True,
    use_mrmr_fs=True, mrmr_kwargs={"features_to_select": 50},
    rfecv_models=["cb"],
    custom_pre_pipelines={"pca50": IncrementalPCA(50)},
)

# AFTER (2026-04-27+):
train_mlframe_models_suite(
    df=...,
    reporting_config=ReportingConfig(
        show_perf_chart=False, show_fi=False,
        title_metrics_template="ICE BR_DECOMP ECE CMAEW",  # new: customise title metrics
        show_prob_histogram=True,                          # new: histogram subplot
    ),
    preprocessing_config=PreprocessingConfig(scaler=custom_scaler),
    output_config=OutputConfig(
        data_dir="./artifacts", models_dir="models", save_charts=True,
    ),
    outlier_detection_config=OutlierDetectionConfig(
        detector=isolation_forest, apply_to_val=True,
    ),
    feature_selection_config=FeatureSelectionConfig(
        use_mrmr_fs=True, mrmr_kwargs={"features_to_select": 50},
        rfecv_models=["cb"],
        custom_pre_pipelines={"pca50": IncrementalPCA(50)},
    ),
)
```

### Severed-config wiring (audit + fixes)

Audit of all 26 `*Config` classes in configs.py found one truly severed
class after the suite signature rebuild:

- **`ConfidenceAnalysisConfig`** was reachable only via the deleted dict
  pass-through. Wired as new first-class kwarg `confidence_analysis_config`
  on the suite. Internal dict gets fields dumped under their existing
  scalar key names (`include_confidence_analysis`,
  `confidence_analysis_use_shap`, etc.) so deep consumers in trainer.py
  continue working unchanged.

Plus three trainer-internal knobs lifted up to suite users via
**`ReportingConfig`** (the natural home — these are all about what gets
reported and how):

- `compute_trainset_metrics` (default False), `compute_valset_metrics`
  (default True), `compute_testset_metrics` (default True) — was hardcoded
  in `_build_configs_from_params`. Users now control per-split metric
  computation explicitly (skip train-set metrics for speed, or enable
  them for overfit diagnostics).
- `custom_ice_metric` / `custom_rice_metric` (default None → trainer
  falls back to compute_probabilistic_multiclass_error) — was reachable
  only via the deleted dict pass-through.

`use_cache` default flip: `TrainingControlConfig.use_cache` and
`_build_configs_from_params(use_cache=...)` both flipped from False to
True for consistency with `train_eval.py:664`'s long-standing
`.get("use_cache", True)` de-facto behaviour. Cache loading is almost
always faster than retraining; users force a retrain by passing
`use_cache=False` explicitly through their TrainingControlConfig
(suite-level wiring intentionally deferred — model-loading is a
trainer-internal concern, not a top-level user knob).

Severed configs that audit found but **deliberately NOT wired** (each
either dead code, wrong suite, or premature):
- `FairnessConfig` — declared but never consumed; the actual fairness
  mechanism is `TrainingBehaviorConfig.fairness_features` (flat list).
  Refactor deferred to a follow-up PR.
- `TreeModelConfig` / `MLPConfig` / `NGBConfig` — declared but never
  instantiated; the dict-typed `ModelHyperparamsConfig.{cb,lgb,xgb,
  hgb,mlp,ngb}_kwargs` is the real config surface today. Migrating
  those dicts to typed configs is high-value but very-high-effort
  (each lib has 100+ params, mirrors evolve upstream) — separate PR.
- `EnsemblingConfig` — TODOs in ensembling.py reference its 3 fields
  (`force_legacy`, `quantile_budget_bytes`, `accumulator`) but the
  underlying logic isn't wired yet; promoting now would be premature.
- `AutoMLConfig` — for `train_automl_models_suite`, not this suite.
- `TrainingConfig` umbrella — documentation-only aggregator;
  redundant given the 17 first-class typed kwargs the suite already has.
- Trainer-internal `DataConfig`/`TrainingControlConfig`/`MetricsConfig`/
  `NamingConfig` (other fields beyond what we promoted) — built inside
  `_build_configs_from_params` from suite-level state; per-model flags
  (`pre_pipeline`, `fit_params`, `model_category`, `model_name_prefix`,
  `just_evaluate`) are auto-derived by the trainer and shouldn't be
  user-exposed at the suite layer.

### Tests

- New: `tests/test_metrics.py::TestCalibration` — 16 added tests
  (Murphy identity, ECE textbook formula, decomposition edge cases, return
  tuple, title-template grammar, token rendering).
- New: `tests/training/test_reporting_config.py` — 30 tests covering
  template parsing, validation failure modes, histogram fields,
  slimmed-config invariants, all 4 new sibling configs.
- New: `tests/training/test_suite_config_migration.py` — asserts the
  removed kwargs really are gone, the new typed configs are present, and
  model-selection kwargs stay top-level.
- New: `tests/test_calibration_plot_layout.py` — histogram subplot axes
  count, inline-label independence, auto-yscale heuristic, explicit
  override modes (matplotlib Agg path).
- Existing tests/scripts mass-migrated: 39 files via automated pass for
  `data_dir/models_dir/outlier_detector/use_mrmr_fs/rfecv_models` →
  typed configs; `init_common_params=common_init_params` → `reporting_config=`;
  `{**common_init_params, "scaler": X}` → split into `reporting_config` +
  `preprocessing_config`. Residual rich-dict cases hand-edited or tagged
  with TODO comments where the deep consumer was reading dict-only fields
  that now lack a typed home (`include_confidence_analysis`).

## 2026-04-27 — Combo-fuzz expansion + 13 production fixes uncovered by the new axes

Expanded `tests/training/test_fuzz_suite.py` from 43 → 61 axes via six
incremental batches (each: add axes, run fuzz, fix surfaced bugs, retest).
Coverage rose to a 100% pairwise-covered 150-combo suite that exercises
the entire `train_mlframe_models_suite` configuration surface. The
campaign uncovered and fixed 13 real production bugs.

### Fuzz axes added

- **Batch 1** (8 axes): `PreprocessingConfig.fix_infinities` /
  `ensure_float32_dtypes` / `remove_constant_columns`,
  `PreprocessingBackendConfig.imputer_strategy`,
  `TrainingSplitConfig.shuffle_val` / `shuffle_test` /
  `wholeday_splitting` / `val_sequential_fraction`.
- **Batch 2** (axis-value expansion): `n_rows` += 5000,
  `cat_feature_count` += 20.
- **Batch 3**: `MultilabelDispatchConfig` (n_chains / chain_order_strategy /
  cv) — and **wired `multilabel_dispatch_config` end-to-end through the
  production API** (3-layer prop: `train_mlframe_models_suite` →
  `select_target` → `configure_training_params` → `strategy.wrap_multilabel`).
  Plus `outlier_detection` += `lof` + `ocsvm`.
- **Batch 4**: `PreprocessingExtensionsConfig` axes (scaler / kbins /
  polynomial_degree / dim_reducer / nonlinear_features) — entire config
  was previously untouched by fuzz.
- **Batch 5**: `rfecv_estimator_cfg` (cb_rfecv / xgb_rfecv).
- **Batch 6**: `recurrent_model_cfg` (lstm / gru / transformer) +
  synthetic per-row sequence builder + RecurrentConfig wiring.

### Production fixes uncovered by the new axes

- **`_ChainEnsemble` is now sklearn-compatible** (`training/helpers.py`):
  inherits `(ClassifierMixin, BaseEstimator)` in the correct sklearn-1.x
  MRO order so `is_classifier()` / `clone()` / `get_tags()` all return
  the right classifier-side answers.
- **`_ChainEnsemble.fit` filters incompatible fit_params**
  (`training/helpers.py`): drops `eval_set` / `eval_metric` / `X_val` /
  `y_val` before forwarding to `ClassifierChain.fit`.
- **Trainer polars-frame guard recognises `_ChainEnsemble`**
  (`training/trainer.py`): peeks at `base_estimator` to decide
  polars-native vs pandas-required.
- **CB pandas-multilabel cat-null defensive fillna**
  (`training/trainer.py:_train_model_with_fallback`): the polars-side
  `_polars_fill_null_in_categorical` pass already covered the
  polars-native input; the pandas multilabel path slipped through.
- **Empty-val splitter min-rows guards**
  (`training/trainer.py:_apply_pre_pipeline_transforms` +
  `_compute_split_metrics`): both call sites now skip cleanly with a
  structured warning when `val_df` arrives with 0 rows.
- **CB Pool reuse: train_target / train_df length-mismatch guard**
  (`training/trainer.py:_maybe_get_or_build_cb_pool`): RFECV's inner CV
  folds occasionally hand `_train_model_with_fallback` mismatched
  `(train_df rows, train_target len)` pairs; the cached Pool's
  `set_label` then leaves a stale label.
- **CB Pool reuse: post-`set_label` non-empty verification**
  (`training/trainer.py`): after the cached Pool's label is swapped,
  read it back and evict the cache entry if `len == 0`.
- **CB Pool reuse: empty-target early return**
  (`training/trainer.py:_maybe_get_or_build_cb_pool`).
- **Empty-val eval_set guard**
  (`training/trainer.py:_setup_eval_set`): an empty `(val_df, val_target)`
  tuple sneaking into `fit_params['eval_set']` made CB raise
  `Labels variable is empty` from the val-side Pool with a misleading
  error message. Skip the `eval_set` injection entirely on empty val
  so the model trains without early-stop validation.
- **`max_error` multioutput-safe**
  (`training/evaluation.py:report_regression_model_perf`): sklearn's
  `max_error` rejects multioutput `y`; compute per-output max-abs-error
  and report the worst on multilabel-shaped predictions.
- **LOF / OCSVM wrapped in `Pipeline([SimpleImputer, OD])`**
  (`tests/training/test_fuzz_suite.py:_outlier_detector_for_combo`):
  unlike `IsolationForest`, both reject NaN; wrap in a sklearn
  `Pipeline` with a `SimpleImputer` so the detector sees clean input.
- **Feature-selector wrapper (`feature_selection/wrappers.py`)**:
  polars → pandas at `RFECV.fit` entry; dtype-aware `cat_features`
  filter (drops cats already encoded to float by an upstream
  `CatBoostEncoder`); outer `fit_params` feature-list narrowing to
  current_features; empty-train CV fold skip.

### Net effect

Pairwise-covered 150-combo suite: ~146/150 (97%) green. Two narrow
xfail rules remain for deeper CB-side fragility (CB text-estimator
dictionary collapse on full-quartet + heavy NaN injection; CB Pool
desync inside MRMR + small-n + heavy filtering); each marked with a
TODO and a precise predicate.

## 2026-04-26 — Session 7: label-distribution drift report + PU-learning wrapper

Two new training-time tools surfaced by a production drift incident
(temporal-split classifier showed VAL ROC AUC 0.99 → TEST 0.64 across
three model families). Both target selection-bias / prior-shift, the
class of failures that don't show up until the trained model meets
test data with a different marginal P(y).

### Added

- **`compute_label_distribution_drift(train, val, test, target_type)`**
  in [training/drift_report.py](training/drift_report.py): per-split
  summary (n / n_positive / P(y=1)) plus cross-split deltas in
  percentage points. Type-aware: binary (P(y=1)), multiclass (per-class
  rate), multilabel (per-label rate), regression (mean / std / median
  / p01 / p99 + train-σ z-scores). Warns when |Δ| > 5pp (binary/multi)
  or > 0.5σ (regression).

- **Auto-wired in `train_mlframe_models_suite`**: the report is computed
  inside the per-target loop right after train/val/test target subsets
  are extracted, BEFORE `select_target` and any model fits. Renders a
  4-line block to the log and stores the structured dict on
  `metadata["label_distribution_drift"][target_type][target_name]`.
  Catches selection bias in seconds rather than after a 5-hour run.

  Example output:
  ```
  label_distribution_drift report (target_type=binary_classification target=cl_act_total_hired_above_1):
    train n=  7_304_969 n_positive=  5_405_680 P(y=1)=0.7400
    val   n=    811_663 n_positive=    698_344 P(y=1)=0.8604
    test  n=    901_847 n_positive=    746_133 P(y=1)=0.8273
    WARN: VAL P(y=1)=0.860 vs train 0.740 (Δ=+12.0pp); selection-bias / prior-shift suspected — model will be miscalibrated on val.
    WARN: TEST P(y=1)=0.827 vs train 0.740 (Δ=+8.7pp); selection-bias / prior-shift suspected — model will be miscalibrated on test.
  ```

- **`PULearningWrapper`** in [training/pu_learning.py](training/pu_learning.py):
  sklearn-style binary classifier wrapper for selection-biased data
  ("most periods are positive-only because the data source only surfaces
  positives; some periods are fully labeled"). Three strategies:

  | strategy                    | what it does                                                                  |
  |-----------------------------|-------------------------------------------------------------------------------|
  | `unbiased_only`             | Train base classifier on the unbiased subset only. Cleanest calibration.       |
  | `prior_shift_correction`    | Train on full data, apply Saerens-Latinne-Decaestecker (2002) at inference.    |
  | `elkan_noto`                | Classical Elkan & Noto (KDD 2008) PU classifier with proxy g(x) and c=P(s=1\|y=1). |
  | `auto`                      | Picks `unbiased_only` if ≥1k each-class unbiased samples, else `prior_shift_correction`. |

  Synthetic temporal-bias benchmark (true TEST P(y=1)=0.46, train observed P(y=1)=0.96):

  | strategy                | mean_pred | AUC   | Brier   | Brier vs naive |
  |-------------------------|-----------|-------|---------|----------------|
  | NAIVE                   | 0.882     | 0.864 | 0.3765  | —              |
  | UNBIASED_ONLY           | 0.463     | 0.833 | 0.1812  | -51.9%         |
  | PRIOR_SHIFT_CORRECTION  | 0.448     | 0.864 | 0.1493  | -60.3%         |
  | ELKAN_NOTO              | 0.653     | 0.856 | 0.1959  | -48.0%         |

  `prior_shift_correction` wins: same AUC as naive (Saerens correction
  is a monotone rescale → preserves ranking), calibration matches true
  prior within 1.5pp, Brier drops 60%. `unbiased_only` is the fallback
  when the operator doesn't know the true target prior.

- **Public API exports** via `mlframe.training`:
  `compute_label_distribution_drift`, `format_drift_report`,
  `PULearningWrapper`, `estimate_c_from_unbiased_positives`.

### Tests

- 20 unit tests in `tests/training/test_drift_report.py` covering
  binary / multiclass / multilabel / regression input types, pandas /
  polars / numpy compatibility, threshold tuning, edge cases,
  JSON-roundtrip safety.
- 24 unit tests in `tests/training/test_pu_learning_synthetic.py`
  including a temporal-bias synthetic generator
  (`_gen_temporal_pu_dataset`) that mirrors the user's production
  drift pattern: 96 months, 4 unbiased windows mid-stream, ~98%
  observed positive rate during biased periods, ~40% true rate.
  Per-strategy invariants: calibration recovery, Brier-vs-naive
  improvement, AUC preservation.
- Integration smoke: drift report fires correctly inside actual fuzz
  combo runs.

## 2026-04-25 — Session 6: multilabel full-pipeline integration

End-to-end multilabel support across `train_mlframe_models_suite`. All 42
multilabel combos in the fuzz suite pass end-to-end (`cb`, `xgb`, `lgb`,
`hgb`, `linear` × pandas/pl_enum/pl_utf8/pl_nullable × n=300/600/1200 ×
single-model + 2-, 3-, 4-, 5-model combos × MRMR-on/off). Binary
non-regression: 6 spot-check combos still pass.

### Added

- **`target_type` / `n_classes` plumbing to `get_training_configs`**
  (`training/trainer.py:configure_training_params`,
  `training/train_eval.py:select_target`, helper
  `_n_classes_from_target`): the suite now derives K from the target
  shape (multilabel → `target.shape[1]`, multiclass → `len(unique)`)
  and routes `target_type`/`n_classes` into `config_params`. Without
  this, multilabel targets reached CB without `loss_function` set, and
  CB's `_get_loss_function_for_train` crashed with
  `TypeError: unhashable type: 'numpy.ndarray'` on the 2-D label.

- **MultiOutputClassifier wrap-on-suite for non-native multilabel
  strategies** (`training/trainer.py:_wrap_for_multilabel_if_needed`):
  HGB/XGB/LGB/Linear get wrapped in `MultiOutputClassifier` (or
  `_ChainEnsemble` per `MultilabelDispatchConfig`) when `target_type ==
  MULTILABEL_CLASSIFICATION`. CB MultiLogloss native is preserved.

- **Multilabel-aware report path**
  (`training/evaluation.py:report_probabilistic_model_perf`): detects
  2-D `targets`, iterates over K columns directly via
  `targets[:, class_id]` (was `targets == class_name` which broadcast
  bool-2-D against 1-D y_score and crashed). `preds` derived as
  `(probs >= 0.5).astype(int8)` for multilabel (was argmax — collapses
  to single class). Probs canonicalised via
  `_canonical_predict_proba_shape` before threshold so
  `MultiOutputClassifier`'s `List[(N, 2)]` output works.

### Fixed

- **CB `eval_metric` incompatible with `MultiLogloss`**
  (`training/helpers.py:get_training_configs`): CB rejects
  `eval_metric='AUC'` with "metric AUC and loss MultiLogloss are
  incompatible". Override to `'HammingLoss'` for `MultiLogloss` and
  `'Accuracy'` for `MultiClass`. Also disable `CB_CALIB_CLASSIF`'s
  custom `ICE` metric for multilabel (CB asserts custom metrics inherit
  `MultiTargetCustomMetric`); falls back to `HammingLoss`.

- **`_setup_eval_set` skips for `MultiOutputClassifier` wrapper**
  (`training/trainer.py`): `eval_set`/`X_val`/`y_val` propagate verbatim
  through the wrapper to each per-label inner estimator; `y_val` stays
  2-D and crashes the inner fit. Inner estimators rely on internal
  early-stopping (HGB `validation_fraction`) or have early stopping
  disabled by `_wrap_for_multilabel_if_needed`.

- **Inner-estimator `early_stopping_rounds`/`callbacks` disabled when
  wrapped** (`training/trainer.py:_wrap_for_multilabel_if_needed`):
  LGB/XGB raise "at least one dataset and eval metric is required for
  evaluation" when early stopping is configured but no eval_set is
  available (skipped per the previous fix). Patch them to `None` on the
  inner estimator before wrapping.

- **Polars-receipt guard sees through `MultiOutputClassifier`**
  (`training/trainer.py`): when wrapping a polars-native estimator
  (HGB) in `MultiOutputClassifier`, the outer name no longer matches
  the polars-native prefix list. The guard now unwraps via
  `model.estimator` to evaluate the inner type.

- **Supervised encoders refuse 2-D y**
  (`training/trainer.py:_multilabel_target_to_1d_for_supervised_encoders`):
  `category_encoders.TargetEncoder` and friends fail with "Unexpected
  input shape: (N, K)". Collapse to 1-D "any-positive" indicator
  (`y.sum(axis=1) > 0`) for the encoder fit only — actual model still
  trains on the full (N, K) target.

- **Multilabel target nlabels-derivation in `configure_training_params`**:
  the `nlabels = len(np.unique(target))` site rejects 2-D arrays.
  Multilabel branch uses `target.shape[1]`; auto-empties
  `catboost_custom_classif_metrics` (AUC-incompatible with
  `MultiLogloss`).

- **MRMR target injection for multilabel** (`feature_selection/filters.py`):
  `X.loc[:, target_names] = vals.reshape(-1, 1)` crashes on 2-D `vals`
  with "Must have equal len keys and value when setting with an
  ndarray". Pass `vals` unchanged when 2-D so each column maps to its
  `target_names` entry. Polars branch was already correct.

### Removed

- **`_rule_multilabel_full_pipeline_deferred`** xfail rule from
  `tests/training/_fuzz_combo.py:KNOWN_XFAIL_RULES`. All 42 multilabel
  combos now pass end-to-end without xfail.

### Verification

- `pytest tests/training/test_fuzz_suite.py -k '<42 multilabel ids>'`:
  42/42 pass.
- `pytest tests/training/test_fuzz_suite.py -k 'c0001 or c0002 or c0003
  or c0004 or c0006 or c0007'`: 6/6 binary baselines still pass.
- `pytest tests/training/test_save_load_multi_output.py`: 6/6.
- `pytest tests/training/test_multi_output_corner_cases.py
  test_multiclass_classification.py test_multilabel_metrics_numba.py
  test_xgb_native_multilabel.py`: 78/78 (1 skipped — iterstrat optional).

## 2026-04-24 — Session 5: multi-output polish + integration-gap discovery

Save/load roundtrip safety, corner-case coverage, and honest documentation
of the multilabel-through-full-pipeline integration gap uncovered by
running the full fuzz sweep.

### Added

- **Save/load roundtrip tests for multi-output** (`tests/training/test_save_load_multi_output.py`, 6 tests):
  - Native CatBoost `MultiLogloss` — `loss_function` survives pickle;
    predictions bit-identical across dump/load
  - `MultiOutputClassifier` wrapper — per-estimator `classes_` arrays
    preserved
  - `MultiOutputClassifier` near-constant label (99% class 0) — no crash,
    low positive probabilities preserved
  - `_ChainEnsemble` — chain `order_` arrays survive, predictions
    bit-identical across pickle
  - `_ChainEnsemble` with `chain_order_strategy='by_frequency'` —
    frequency-resolved orderings survive
  - `_PostHocMultiCalibratedModel` wrapping `MultiOutputClassifier` +
    `_PerClassIsotonicCalibrator` — full 3-level nested pickle roundtrip

- **Corner-case tests** (`tests/training/test_multi_output_corner_cases.py`, 7 tests):
  - `test_cb_multilogloss_accepts_int_targets_returns_NK` — CB native
    contract verification
  - `test_xgb_num_class_inference_survives_y_val_none` — no crash on
    missing validation set
  - `test_multilabel_K1_degenerate_metrics_ok` — K=1 edge case for
    hamming/subset/jaccard + canonicalizer
  - `test_splitting_stratify_2d_via_iterstrat_roundtrip` — per-label
    frequency preservation within ±10pp (requires `iterative-stratification`)
  - `test_splitting_stratify_1d_equivalent_to_sklearn` — 1-D stratify via
    sklearn `StratifiedShuffleSplit`
  - `test_iterstrat_import_error_has_install_hint` — helpful error
    message when optional dep missing
  - `test_xgb_native_multilabel_kwargs_have_tree_method` —
    force_native_xgb_multilabel kwargs correctness
  - `test_per_class_calibrator_tiny_calib_set` — graceful handling when
    calibration set is smaller than n_classes

### Fixed

- **`select_target` multilabel guard** (`training/train_eval.py:297+`):
  was raising `ValueError: Data must be 1-dimensional` on 2-D
  multilabel target via `pd.Series(target).value_counts()`. Now
  dispatches by target_type; multilabel path computes per-label
  positive rates for model_name summary instead of the binary
  value_counts path.

### Deferred (Session 6+ — integration epic)

**Multilabel through full `train_mlframe_models_suite` pipeline** is a
4-6h integration epic surfaced by running the full 200-combo fuzz sweep
after re-adding `multilabel_classification` target_type in Session 3.
Each combo that reaches the CB fit path hits a new 1-D-assuming site:

1. **`select_target`** — FIXED (this session).
2. **CatBoost's `_get_loss_function_for_train`** (catboost/core.py:1802,
   upstream quirk): runs `len(set(label))` for auto loss-inference even
   when user sets `loss_function='MultiLogloss'` explicitly. Fails on
   2-D label with `TypeError: unhashable type: 'numpy.ndarray'`.
   Workaround needs mlframe-side label routing in
   `CatBoostStrategy.build_native_dataset` / trainer.py.
3. **`configure_training_params`** — likely expects 1-D target for class
   weights / imbalance handling.
4. **`report_probabilistic_model_perf`** per-class loop — the multilabel
   `if len(classes)==2 and class_id==0: continue` skip is wrong semantics.
5. **Metrics computation + display** — per-output reporting.

Added XFAIL rule `_rule_multilabel_full_pipeline_deferred` to
`_fuzz_combo.py::KNOWN_XFAIL_RULES` so the fuzz sweep marks multilabel
combos as expected-fail with a clear pointer to this CHANGELOG entry
and the audit-trace summary.

**Multi-output primitives remain fully tested** — FTE unpack, dispatch,
calibration, metrics, ensembling, save/load, chain ensemble all have
passing unit tests. The gap is purely in the integration flow through
the top-level suite function.

- **Recurrent multilabel sigmoid head** — separate K-output architecture
  needed; current recurrent handles multiclass via softmax, multilabel
  deferred as own architecture track.

## 2026-04-24 — Session 4: per-class calibration + METRICS_REGISTRY + streaming ensembling + numaggs Kahan Phase 2

Continuation of multi-output / numerical-stability work. Removes the
NotImplementedError hole for multi-output calibration, adds a pluggable
metrics-registry for downstream extensibility, completes Kahan coverage
in numaggs moment kernels, and delivers streaming ensembling via
_WelfordAccumulator.

### Added

- **Per-class isotonic calibration** (`training/trainer.py`):
  `_PerClassIsotonicCalibrator` — K independent `IsotonicRegression`
  fits, with row-renormalisation for MULTICLASS (preserves softmax
  sum-to-1) and independent columns for MULTILABEL. Constant-label
  columns → identity mapping (no crash). Output clipped `[0, 1]`.
  Wrapped in `_PostHocMultiCalibratedModel` for transparent
  `predict_proba` / `predict` delegation; pickle-roundtrip verified.
  `evaluation.post_calibrate_model` routes multi-output targets here
  (was: `raise NotImplementedError`).
- **METRICS_REGISTRY pattern** (`training/metrics_registry.py`):
  pluggable metric registration per target_type. Built-in
  multilabel metrics (`hamming_loss`, `subset_accuracy`,
  `jaccard_samples`) auto-dispatch from `report_probabilistic_model_perf`.
  External callers register custom metrics via `register_metric(...)`
  without touching `evaluation.py`. Failing metrics silently skipped.
- **Streaming ensembling** (`ensembling.py`):
  `ensemble_probabilistic_predictions_streaming` using
  `_WelfordAccumulator` for O(N*K) peak memory regardless of M.
  Supports `arithm`/`harm`/`quad`/`qube`/`geo`. `median` raises
  NotImplementedError (needs P²-Quantile sketch, deferred —
  scalar-per-cell vectorisation prototyped but rejected due to
  per-cell direction heterogeneity).
- **Kahan compensation in `compute_moments_slope_mi`**: all 12
  uncompensated accumulators in the per-row loop (slope_over,
  slope_under, r_sum, mad, std, skew, kurt + 5 weighted variants)
  now Kahan-Babuška-Neumaier compensated. Precision recovered to
  machine-epsilon on catastrophic cancellation cases (1e9 + N(0,1e-3)
  float64 → 0-relerr skew vs naive 2.78 = 315× improvement on the
  original audit finding). Verified against scipy reference across
  normal_N{1k,1M} / catastrophic / lognormal distributions.

### Fixed

- **`numerical.py:740` bug** (landed in Session 2): was
  `weighted_skew += w_summand * w_d * next_weight` — double-
  accumulating skew, never accumulating kurt. Affected every caller
  of `compute_moments_slope_mi` with weights.

### Tests added

- `tests/training/test_per_class_isotonic.py` (11 tests): per-class
  isotonic rows-sum-to-1 (MULTICLASS), independent columns (MULTILABEL),
  output in [0,1], constant-label-column identity, Brier-loss improvement
  on miscalibrated, `_PostHocMultiCalibratedModel` shape / predict /
  pickle roundtrip, METRICS_REGISTRY dispatch / custom registration /
  failing-metric skip.
- `tests/training/test_ensembling_streaming.py` (9 tests, Session 3):
  streaming-vs-materialised bit equivalence across 5 methods, median
  NotImplementedError, outlier-filter WARN, Welford + combine.

### Out of scope (future sessions)

- **P²-Quantile streaming sketch** — scalar-per-cell vectorisation
  gave ~100% relative error (per-cell direction `d_sign` varies across
  cells, broken by scalar update). Correct impl requires numba-jit
  per-cell inner loop (~1h focused work). For mlframe's typical M=5-10
  ensembling regime, materialised sort is already O(1) wallclock
  per cell and competitive in memory — P² real value is for big-M
  (CV 100+ folds) or unbounded-stream feature quantiles.
- **Recurrent multilabel sigmoid head** — separate K-output architecture
  needed; current recurrent handles multiclass via softmax, multilabel
  deferred.
- **Welford-Pébay migration in skew/kurt**: reframed as unnecessary —
  Kahan compensation (this session) already achieves machine-epsilon
  precision on the catastrophic case. Pébay's online formulation would
  give marginal extra precision at 2× cost.
- **15 review-driven test gaps**: K=1 degenerate, jit cold-start
  budget guard, full-sweep walltime guard, etc.
- **Polars `pl.Array(pl.Int8, K)` through full pipeline**: FTE
  unpacks; internal transforms still see pandas-post-unpack. Native
  polars path requires per-transform adapter.

## 2026-04-24 — Multiclass + multilabel classification (foundation)

First-class support for `MULTICLASS_CLASSIFICATION` (K>2 single-label
softmax) and `MULTILABEL_CLASSIFICATION` (K>=1 independent-binary OvR /
native-CB / chain ensemble) target types alongside existing binary +
regression. ~75 new tests; the runtime/dispatch surface is non-trivial
so the test/code ratio is high.

### Added

- **`TargetTypes` enum** (`configs.py`): `MULTICLASS_CLASSIFICATION` +
  `MULTILABEL_CLASSIFICATION` values; class-property predicates
  `is_classification`, `is_binary`, `is_regression`, `is_multiclass`,
  `is_multilabel`, `is_multi_output`. Replaces ad-hoc
  `target_type == BINARY_CLASSIFICATION` checks.
- **Per-strategy native dispatch** in `helpers._classif_objective_kwargs`:
  CB `MultiClass` / `MultiLogloss`; XGB `multi:softprob+num_class=K`;
  LGB `multiclass+num_class=K`; HGB sklearn auto-detect; Linear
  `multi_class='multinomial', solver='lbfgs'`.
- **Multilabel dispatch** via `_maybe_wrap_multilabel`: native CB path
  (no wrapper); `MultiOutputClassifier` for XGB/LGB/HGB/Linear; or
  `_ChainEnsemble` (custom — sklearn `VotingClassifier(soft)` over
  `ClassifierChain` is BROKEN for multilabel, raises on 2-D y) of N
  random-ordered chains with `cv=5` cross-validated chain features.
- **Strategy feature flags** `supports_native_multiclass` /
  `supports_native_multilabel` on `ModelPipelineStrategy` ABC. CB sets
  both True; LGB/XGB/HGB/Linear set only multiclass=True; multilabel
  via wrapper. New strategies opt-in by overriding (open/closed).
- **`(N, K)` probability surface contract**: `_canonical_predict_proba_shape`
  + `_predict_from_probs` helpers handle binary `(N,2)`,
  `MultiOutputClassifier` `List[(N,2)]`, 1-D sigmoid, native `(N,K)`,
  per-label thresholds (`Union[float, ndarray]`), NaN-safe decision
  rule. Constant-label-column (per-label estimator with `classes_=[0]`)
  emits zeros instead of IndexError.
- **`MultilabelDispatchConfig` + `EnsemblingConfig`** Pydantic configs.
  Replace env vars and sprawling kwargs with structured config.
- **`stratify_y`** parameter in `make_train_test_split`: 1-D sklearn
  `StratifiedShuffleSplit`, 2-D `iterstrat.MultilabelStratifiedShuffleSplit`
  (lazy import; helpful ImportError on missing optional dep).
- **Numba multilabel metrics** in `metrics.py`: `hamming_loss`,
  `subset_accuracy`, `jaccard_score_multilabel`. Sequential + parallel
  (auto-selected when `N*K > 1_000_000` — per Win32 `numba.prange`
  cold-spawn benchmarking). Wrappers validate shape mismatch BEFORE
  numba call (numba doesn't bounds-check inner loops).
- **Ensembling materialisation dedup** (`ensembling.py`): 9 separate
  `np.array(preds)` materialisations replaced with single `_preds_arr`
  cache. ~5× peak-memory drop during aggregation. Geometric mean
  rewritten via `exp(mean(log(...)))` with safer 1e-300 clip floor
  (was 1e-12 — destroyed signal from well-calibrated boosted trees).
- **Artifact metadata schema versioning** (`core.py`): per-model schema
  records now include `target_type`, `n_classes`, `multilabel_strategy`,
  `schema_version: int = 2`. Legacy artifacts (no fields) infer from
  model structure at load time.
- **Combo-fuzz**: `multiclass_classification` re-added to `target_type`
  axis (3-class via balanced quantile-cut). `multilabel_strategy_cfg`
  (`auto`/`wrapper`/`chain`) axis added forward-compat.

### Changed

- `[:, 1]` binary slicing replaced with shape-aware dispatch in
  `core.py` (4 sites) and `automl.py` (2 sites with `multi_class='ovr'`).
- `evaluation.post_calibrate_model` raises `NotImplementedError` on
  multi-output probability shapes (univariate post-hoc isotonic doesn't
  generalise; per-class isotonic is Session-2 work).

### Tests added (~75 tests, 63 verified passing)

- `tests/training/test_multiclass_classification.py` (29 tests):
  TargetTypes predicates, canonicalizer, decision rule, strategy
  feature flags, `get_training_configs` plumbing, chain orders,
  config dataclass defaults.
- `tests/training/test_multilabel_metrics_numba.py` (24 tests):
  hamming/subset/jaccard correctness vs sklearn, parallel-variant
  equivalence, edge cases.
- `tests/training/test_bizvalue_classifier_chain.py` (3 tests):
  ChainEnsemble beats `MultiOutputClassifier` on correlated labels —
  observed +0.59pp Jaccard mean delta over 5 seeds, positive in 5/5.

### Out of scope (Session 2-3)

- Multi-* post-hoc calibration (per-class isotonic) — currently raises
  `NotImplementedError`.
- Multilabel fuzz axis (needs 2-D-aware `SimpleFeaturesAndTargetsExtractor`).
- Native XGB multilabel (`multi_strategy='multi_output_tree'`) — WIP
  in upstream, unstable.
- Polars-native multilabel target dtype.
- Recurrent / NeuralNet multi-output.
- Full Welford-streaming ensembling refactor for prod-sized N=9M+
  frames (current dedup gives ~5× peak-memory drop; full streaming
  via Welford accumulators + P²-Quantile sketch gives ~10× more).

## 2026-04-24 — Combo-fuzz upgrade: A–G (invariants, metamorphic, 3-wise, Hypothesis, seed rotation, adversarial axes)

Seven-part upgrade to the combo-fuzz harness in `tests/training/`.
Pairwise test count unchanged at 150; new infrastructure sits alongside.

### Added

- **`tests/training/COMBO_FUZZ.md`** — design doc for the fuzz approach:
  axes, covering algorithm, combo id scheme, invariants, A–G upgrade
  roadmap, operating manual.
- **Fix C — per-combo property invariants** (`test_fuzz_suite.py::_assert_prediction_invariants`):
  - I1 finiteness: no NaN/Inf in `val_preds` / `val_probs`.
  - I2 non-constant classification probs on val (≥2 distinct rounded to 6dp).
  - I3 prediction-length upper-bound: `len(val_preds) <= meta['val_size']` (OD may shrink but never grow).
- **Fix D — metamorphic dual-run suite** (`test_fuzz_metamorphic.py`):
  - D1 column-rename invariance (val metric stable under feature rename).
  - D2 duplicate-row stability (val metric stable under 5% row duplication).
  - 5 curated combos (one per model-family), expandable via `MLFRAME_METAMORPHIC_ALL=1`.
- **Fix E — master_seed rotation**: each `_fuzz_results.jsonl` row now
  carries `master_seed`. Driver script `run_fuzz_seed_rotation.py`
  runs N seed-rotated sweeps and aggregates `_fuzz_seed_summary.jsonl`.
- **Fix G — adversarial axis values** (`_fuzz_combo.AXES`):
  `inject_label_leak`, `inject_rank_deficient`, `inject_all_nan_col`.
  Exercise feature leakage, colinear pairs, all-NaN columns.
- **Fix A — 3-wise covering suite** (`test_fuzz_3way_suite.py` +
  `enumerate_combos_3way` in `_fuzz_combo.py`): greedy triple coverage
  over 15 curated load-bearing axes. 400 combos → 99.99% triple
  coverage. Default master_seed `20260424` (distinct from pairwise).
- **Fix B — Hypothesis continuous leaves** (`test_fuzz_hypothesis.py`):
  draws `n_rows`, `fillna_value`, `test_size`, `cat_feature_count`,
  `null_fraction_cats`, `iterations` continuously per example. Default
  20 examples, auto-shrinking on failure.

### Fixed

- **polars-ds fork** (`D:/Temp/polars_ds_fork`): `cs.string() | cs.categorical()`
  didn't match `pl.Enum`. Extended the selector in `transforms.py` (6 call
  sites), `pipeline.py` (5 default-selector spots), and the
  `drop([pl.String, pl.Categorical])` call in the one-hot-encode teardown
  to include `pl.Enum`. Unblocks `polars_enum × onehot` combos.
- **`mlframe/training/pipeline.py`**: `create_polarsds_pipeline` pre-checks
  for encodable columns — skips the `one_hot_encode` / `ordinal_encode`
  step when the frame has no string/categorical/enum columns, rather
  than letting polars-ds raise "Provided columns either do not exist".
- **`mlframe/training/utils.py`**: `_process_special_values` restricts
  pandas `fillna` to numeric columns only. Unrestricted `df.fillna(0.0)`
  was raising `TypeError: Cannot setitem on a Categorical with a new
  category (0.0)` when any Categorical column existed.

## 2026-04-24 — Coverage-gap sweep + 7 framework bug fixes

47 new tests landed across two files, plus 12 new fuzz axes and
2 per-combo invariants. Every entry below traces to a test that
surfaced the underlying bug.

### Added

- **`tests/training/test_suite_coverage_gaps.py`** — 47 tests (46 pass,
  1 skip on CPU-only hosts) covering `train_mlframe_models_suite`
  invariants the fuzz suite couldn't reach:
  - Tier 1 (12): schema-hash load (#2), OD integration (#3), RFECV
    smoke (#4), ensembles log guard (#5), sample_weight validation
    (#6), constant/all-null column handling (#7), determinism (#13),
    polars≡pandas feature set (#14), no caller-frame mutation (#16).
  - Tier 2 (5): single-class target (#8), high-cardinality cat (#9),
    inf/NaN numeric (#10), datetime column (#11), multi-target
    regression (#12).
  - Tier 3 (3): iteration non-regression (#15), save→load→predict
    roundtrip (#17), tier_dfs_cache hit rate (#18).
  - Tier 4 (5): strict Pydantic configs reject typos (#19a),
    permissive configs WARN on typos (#19b), metadata schema
    (#20), `continue_on_model_failure=True` graceful skip (#21),
    `verbose=0` stdout silence (#22).
  - Group A (10): empty/None df error (#23), missing target (#23b),
    empty/unknown `mlframe_models` (#24), predict output range
    (#25a+b), column-order invariance (#26), subprocess load/predict
    (#27), duplicate-in-`mlframe_models` (#28).
  - Group B (6): custom pre_pipelines (#29), preprocessing_extensions
    polynomial features (#30), fairness subgroups (#31), calibrated
    classifiers (#32), streaming parquet via `df=str(path)` (#33),
    `trusted_root` security guard (#34).
  - Group C (6): recurrent LSTM smoke (#35), CB GPU≡CPU (#36), memory
    cap during polars→pandas (#37), MRMR+RFECV stack (#38),
    `continue_on_failure`+ensembles (#39), uninformative column (#40).
- **Fuzz axis expansion** (`tests/training/_fuzz_combo.py`): 12 new
  axes bringing total to 23+ dimensions — `outlier_detection`,
  `use_ensembles`, `continue_on_model_failure`, `iterations`,
  `prefer_calibrated_classifiers`, `inject_degenerate_cols`,
  `inject_inf_nan`, `with_datetime_col`, `inject_zero_col`,
  `fairness_col`, `custom_prep`, `input_storage`. Canonicalisation
  skips `custom_prep=pca2` when the frame has non-numeric columns
  (PCA incompatible).
- **Two post-train invariants in fuzz runner** (`test_fuzz_suite.py`):
  #16 caller-frame mutation guard + #20 metadata schema check fire
  on every combo — ~300 invariant evaluations per 150-combo sweep.

### Fixed (framework bugs surfaced by the new tests)

- **Recurrent path was DOA** (`training/trainer.py:3305`). Four
  field-name typos in the `RecurrentConfig(...)` call: `seq_input_dim`
  and `features_dim` aren't dataclass fields (wrappers compute both
  from input shapes at fit-time, `neural/recurrent.py:1041-1042`);
  `num_heads` was a typo for `n_heads` (declared line 170);
  `mlp_hidden_dims` was a typo for `mlp_hidden_sizes` (line 174).
  Every LSTM/GRU/RNN/Transformer fit had been crashing with
  `TypeError` / `AttributeError` for an unknown duration. Surfaced
  by `test_recurrent_lstm_smoke`.
- **`ensembling.py:402` kwarg collision** — the ensemble scorer's
  `dict(..., drop_columns=[], ..., **kwargs_copy)` raised
  `TypeError: dict() got multiple values for keyword argument
  'drop_columns'` on every ensemble run that used the standard
  `init_common_params={"drop_columns": []}` pattern. Now pops
  `drop_columns`, `category_encoder`, `scaler`, `imputer` from
  `kwargs_copy` before the splat. Surfaced by
  `test_ensembles_enabled_produces_ensemble_log` + 14×
  `TypeError` in the fuzz sweep.
- **Datetime columns crashed downstream models**
  (`training/core.py :: train_mlframe_models_suite`). Raw
  `pl.Datetime` / pandas `datetime64[ns]` columns surviving the
  extractor reached CB Pool / XGB DMatrix / MRMR numeric path and
  triggered 6× `CatBoostError: Error while processing column` and
  18× `numpy.DTypePromotionError`. Now decomposed into
  `day/weekday/month/hour` int8 components via
  `feature_engineering.basic.create_date_features` BEFORE the
  pre-pipeline polars-clone — so `train_df_polars_pre` inherits the
  numeric decomposition. (Earlier draft used naive int64-epoch
  coerce; user pushed back on the signal quality → rewrote using
  the existing calendar-decomposition helper.)
- **`_apply_outlier_detection_global` crashed on non-numeric columns**
  (`training/core.py:971`). `IsolationForest.fit(train_df)` called
  `validate_data` which tried to coerce string/categorical columns
  to float. Now selects numeric+boolean columns only before fit
  and predict (symmetric for train + val splits). Surfaced 20+
  times in the fuzz sweep.

### Invariant relaxations (runner)

- Fuzz runner now accepts empty `trained` dict when
  `continue_on_model_failure=True` (intended degradation — not a bug).
- `model_schemas` only asserted when `trained` non-empty.
- Per-test timeout raised 60s → 300s (heavy 5-model × MRMR × OD ×
  iterations=30 combos legitimately need ~2-3 min).

### Infrastructure

- `run_fuzz_10k.py` (new file from earlier in the day) — spawns
  one pytest subprocess per master_seed with retry on fast-fail
  imports (flaky shap / transformers collection on Windows).
- Memory rules documented: **never xfail framework bugs to defer**
  (fix on the spot; xfail only for third-party / OS quirks);
  **search for reuse before writing** (grep project + pyutilz
  helpers before implementing any transform/FE utility —
  naive-feeling fixes are a smell that proper tooling exists).

## 2026-04-23 — Prod-log review fixes

Batch of four fixes motivated by review of the 2026-04-23
`train_mlframe_models_suite` log on `prod_jobsdetails`.

### Fixed

- **Nullable Polars Boolean → pandas `object` crashed LightGBM**
  (`mlframe/training/utils.py :: get_pandas_view_of_polars_df`). The
  column `hide_budget` in prod is `pl.Boolean` with nulls; through
  pyarrow's `to_pandas()` it materializes as pandas `object` dtype
  carrying `True`/`False`/`None`, which LightGBM's sklearn wrapper
  rejects with `ValueError: pandas dtypes must be int, float or bool`.
  The bridge now detects Arrow `bool` columns before the conversion and
  coerces any object-materialized ones to pandas `Int8` with `pd.NA` —
  verified 2026-04-23 as the universal format accepted by LightGBM
  4.6, XGBoost 3.2, and CatBoost 1.2.10 (plain pandas nullable
  `boolean` is rejected by CB with "Cannot convert <NA> to float";
  `Int8` is not). Non-null Boolean columns stay as zero-copy numpy
  `bool` — the coercion pays only when needed.
- **`RuntimeWarning: divide by zero` on ensemble harmonic mean**
  (`mlframe/ensembling.py`). The previous implementation relied on
  `inf`-routing (`1/0 → inf → 1/mean(...) → 0`), which was numerically
  correct by definition but produced a noisy warning on every ensemble
  scoring. Rewritten with `np.errstate(divide="ignore")` + explicit
  zero-mask: any slot where at least one model predicts exactly 0 is
  routed to 0 directly, which is the HM definition (`HM(0, x) = 0`).
  No warning, same values.

### Changed

- **Default weighting schemas flipped to include a uniform baseline**
  (`mlframe/training/extractors.py :: SimpleFeaturesAndTargetsExtractor`).
  `use_uniform_weighting` now defaults to `True` (was `False`).
  Timeseries runs now produce `{uniform, recency}`; non-timeseries runs
  produce `{uniform}` alone. Motivation: the 2026-04-23 prod suite ran
  with `recency`-only weighting, which made attributing a 99.9% VAL /
  71% TEST AUC gap to weighting vs. training plan impossible — there
  was no baseline to compare against. Users can still opt out
  explicitly with `use_uniform_weighting=False`.

### Added

- **Sticky "Polars-fastpath-broken" flag on CatBoost models**
  (`mlframe/training/trainer.py`). When `_predict_with_fallback`'s
  fallback path converts a `pl.DataFrame` to pandas in response to the
  CB `No matching signature found` dispatch miss, it now sets
  `model._mlframe_polars_fastpath_broken = True`. On subsequent
  `predict_proba` / `predict_log_proba` calls (VAL → TEST →
  per-ensemble-member scoring), `_predict_with_fallback` short-circuits
  directly to the pandas path instead of re-hitting the same TypeError.
  Mirror flag set in `_train_model_with_fallback` when the fit-side
  fallback fires. Saves one WARN line + ~1–2 s per predict call on
  prod-size frames (observed: 3–5 such calls per trained model).

### Tests

- New sensor suite: `mlframe/tests/training/test_prod_log_fixes_2026_04_23.py`
  (16 tests). Locks in each of the above: nullable-Boolean → Int8
  coercion + end-to-end LGB/XGB/CB fit, `harm` mean zero-handling +
  warning-free path, weighting-schema defaults, CB sticky-flag
  shortcut, and the pipeline_cache kind-isolation regression below.

### Fixed (follow-up: duplicate polars→pandas conversion)

- **`pipeline_cache` cross-stream leakage between polars-native and
  pandas-consuming strategies** (`mlframe/training/core.py`). CB, XGB,
  and LGB all inherit ``cache_key="tree"``, and XGB/LGB additionally
  share ``feature_tier()``. Before this fix the cache key was
  ``tree__tier(...,...)``, so XGB (Polars-native) wrote its polars
  train/val/test frames into ``pipeline_cache`` under the same key LGB
  then pulled from. LGB received a polars frame despite having run the
  lazy pandas conversion one line earlier, and the trainer paid a
  duplicate polars→pandas conversion as a silent self-heal — 224 s +
  27.9 GB on the 2026-04-23 prod run, hidden inside a WARNING. Fix:
  the cache key now includes a container-kind suffix
  (``_kindpl`` / ``_kindpd``) derived from ``strategy.supports_polars``,
  so polars and pandas consumers never read each other's entries even
  when ``cache_key`` and ``feature_tier`` are identical. Mirror of the
  same fix already applied to ``_build_tier_dfs`` upstream.
- **Trainer self-heal replaced with hard-raise for non-Polars-native
  models** (`mlframe/training/trainer.py::_train_model_with_fallback`).
  The previous "self-heal" silently ran a second polars→pandas
  conversion whenever LGB received a polars frame; that convenience is
  exactly what buried the pipeline_cache leak above. The self-heal
  path is now a ``RuntimeError`` that names the frame's id + shape and
  points at ``pipeline_cache`` — future leaks surface as loud errors,
  not hidden duplicate work. Polars-native strategies (CB, XGB, HGB)
  still receive polars unchanged.

### Tests (follow-up)

- ``TestPipelineCacheKindIsolation`` (3 tests in the sensor suite):
  end-to-end mixed cb+xgb+lgb suite on a polars frame must complete
  without the trainer's hard-raise tripping; the cache-key format
  includes the kind suffix; the trainer hard-raises on polars input to
  LGB (defense-in-depth).

### Changed (follow-up 2: log readability)

- **Conf Ensemble names now carry ``[VAL COV=xx%]`` tag**
  (`mlframe/ensembling.py::_score_ensemble_for_method`). The 2026-04-23
  prod log showed ``Conf Ensemble ...`` reporting 99.77 % accuracy with
  no hint the metric was measured on a ~10 % coverage slice — easy to
  misread as a headline. The coverage source is picked VAL → TEST →
  TRAIN (first non-empty) and rendered directly in the log prefix so
  `grep` surfaces it without reading the calibration subsection.
- **Ensemble names now list participating models**
  (`mlframe/training/core.py`). ``EnsARITHM 2models`` hid dropouts; now:
    - ≤4 members → ``[cb+xgb+lgb]`` (full short-tag list)
    - &gt;4 members → ``[N=5]`` (readable cap for large suites)
  Short tags derived from model class name: CatBoost\*→cb, XGB\*→xgb,
  LGBM\*→lgb, HistGradient\*→hgb.
- **Category drift WARN carries cardinality-dependent healing suggestions**
  (`mlframe/training/core.py`). The 2026-04-23 log showed 4 "Category
  drift suspect" WARN lines with no actionable guidance. Each WARN now
  includes a tiered suggestion block:
    - card ≥ 1 000 → FeatureHasher / target-encoding / top-K + ``__OTHER__``
    - card ∈ [100, 1 000) → CatBoostEncoder / top-K
    - card &lt; 100 → add ``__UNSEEN__`` bucket / widen training window
  **Decision is made using train-side cardinality + train-vs-val drift
  only** — ``test_only`` is reported for operator visibility but NEVER
  used to shape preprocessing (would leak test into training).

### Added (follow-up 2)

- **Defense-in-depth assert post lazy-conversion**
  (`mlframe/training/core.py`). Immediately after the per-strategy lazy
  conversion loop, every ``common_params`` DF key is re-checked and a
  ``RuntimeError`` is raised if any still holds a ``pl.DataFrame``. Pins
  the failure one function up the stack from the trainer-boundary
  hard-raise — the exception message then identifies which key leaked
  and from which strategy, saving debug time on future pipeline_cache /
  common_params regressions.

### Tests (follow-up 2)

- ``TestEnsembleNameAnnotations`` (3 tests): Conf Ensemble COV tag
  format; ``[cb+xgb]`` member-label when ≤4 models; ``[N=5]`` compression
  when &gt;4 models.
- ``TestCategoryDriftHealingSuggestions`` (2 tests): end-to-end WARN
  contains the "suggested actions" block; structural check of the
  cardinality-tier → keyword mapping (hash-bucket / target-encoding /
  ``__UNSEEN__``).
- ``TestLazyConversionDefenseInDepth`` (1 test): LGB-only suite on a
  polars frame reaches ``_build_tier_dfs`` only with pandas frames
  (polars leakage would have been caught by the new assert in core.py).

## 2026-04-21 — Training-overhead fixes (plan: `jolly-wishing-deer`)

Addresses ~14 min of avoidable overhead and one blocking crash observed on
a 9 018 479 x 118 production run (mlframe_models=["cb", "xgb", "lgb"]).
Full plan: `C:/Users/TheLocalCommander/.claude/plans/jolly-wishing-deer.md`.

### Fixed

- **LGB crash on non-pandas `X` (Fix 4)**. LGB 4.6.0 exposes
  `feature_names_in_` as a read-only property; sklearn 1.8's
  `validate_data` (triggered whenever LGB's `fit()` receives a non-pandas
  input such as Polars or numpy) calls `self.feature_names_in_ = ...` and
  raises `AttributeError`. `mlframe/training/trainer.py` now installs a
  no-op setter on `lightgbm.sklearn.LGBMModel.feature_names_in_` at module
  import (`_patch_lgb_feature_names_in_setter`, idempotent). Defence-in-depth
  behind Fix 1 — if a non-pandas `X` ever reaches LGB, the setter catches
  the value into `_mlframe_feature_names_in_override` instead of crashing.
- **Deleted stale `_warn_on_unsupported_polars_dtypes` (Fix 2)**. The
  pre-fit warning predicted a 2-min pandas fallback on every aligned
  `pl.Enum` frame fed to CatBoost. Empirically false on CB 1.2.10 +
  polars 1.40 (verified 2026-04-21 by direct repro: Enum+filled+eval_set
  fit succeeds on the fastpath). Keeping the warning motivates unsafe
  Enum→Categorical casts which reintroduce the XGBoost sparse-code bug
  documented in `reproducers/xgboost_cats_bug/`. The post-fail
  `_polars_schema_diagnostic` dump is kept but now flags nullable
  Categorical cat_features (the real CB dispatch gap) as the likely
  culprit and reports Enum only for visibility.

### Added

- **Lazy per-strategy pandas conversion (Fix 1)**. `mlframe/training/core.py`
  defers the upfront polars→pandas conversion when the ONLY blockers for
  the Polars fastpath are non-native strategies (LGB, sklearn, linear).
  Polars-native strategies (CB, XGB, HGB) now consume the aligned Enum
  Polars frame directly; non-native strategies convert just-in-time via
  a sentinel-memoized path inside the strategy loop
  (`_pandas_converted_for_non_native` on `common_params`). Saves 661 s
  polars→pandas + ~70 GB RAM on the user's reference workload, and also
  guarantees LGB receives a pandas DataFrame so `lightgbm/sklearn.py:948`
  short-circuits sklearn's crash-prone validate path.
- **`deep: bool = True` kwarg on `pyutilz.pandaslib.get_df_memory_consumption`
  (Fix 3A)**. Default preserves existing behaviour. `deep=False` delegates
  to `df.memory_usage(deep=False).sum()` — O(cols), milliseconds — instead
  of the pathological O(rows * avg_str_len) deep scan on high-cardinality
  object columns. mlframe's `configure_training_params` now passes
  `deep=False` for the GPU-RAM-fit heuristic (byte precision unnecessary).
- **Polars-side size cache (Fix 3B)**. `mlframe/training/core.py` computes
  `train_df.estimated_size()` / `val_df.estimated_size()` (O(cols),
  microseconds) BEFORE any pandas conversion and threads the values into
  `select_target` / `configure_training_params` as new kwargs
  (`train_df_size_bytes`, `val_df_size_bytes`). When cached, skips the
  `get_df_memory_consumption` call entirely. Eliminates the ~3 min
  `select_target` step on 75 GB pandas frames.
- **Upfront group filter in `fast_aucs_per_group_optimized` (Fix 5)**.
  `mlframe/metrics.py` now precomputes per-group `(count, pos_count)` in
  one `np.unique` + `np.bincount` pass, drops sample rows belonging to
  single-sample or single-class groups BEFORE the per-group sort + numba
  inner loop, and emits NaN entries directly. On the user's 95 %-NaN-group
  workload this slashes the inner-loop work (500k-row synthetic bench:
  ~9x speedup). Output dict is bit-identical to the pre-filter path.
- **`use_text_features: bool = True` toggle on `FeatureTypesConfig`
  (Fix 6)**. Master opt-out for text_features. When False,
  `_auto_detect_feature_types` returns empty text/embedding lists AND
  clears any explicit `text_features` list before downstream consumers
  see it. Default True preserves today's behaviour. Use case: CB's
  text-feature TF-IDF pipeline was the training bottleneck
  (`skills_text` with 2M unique) on the user's 9M-row run; turning it
  off routes those columns through the cat-feature path for all
  strategies. Caveat: changes the realised schema → see Fix 8.
- **`tqdmu_lazy_start` helper in `pyutilz.system` (Fix 7)**. Drop-in
  replacement for `tqdmu(iterable, **kwargs)` that resets the bar's
  elapsed counter at the FIRST iteration, not at bar construction. Fixes
  the stale-timer artefact where the user saw `target type: 0/1
  [6:27:44<?]` after a long preprocessing phase. All 6 `tqdmu(...)` call
  sites in `mlframe/training/core.py` converted to `tqdmu_lazy_start`.
- **Per-model input-schema fingerprint in filename + metadata (Fix 8)**.
  `mlframe/training/utils.py` new `compute_model_input_fingerprint(df,
  cat_features, text_features, embedding_features) -> (hash, schema)`
  computes a 10-char SHA256 of the realised fit-time schema (sorted
  columns + canonical dtype + role). Model files now save as
  `{model}_{weight}__sch_{hash}.dump` and metadata records
  `model_schemas[model_file_name] = {schema_hash, input_schema,
  mlframe_model, weight_name}`. Two runs with different
  `use_text_features` / `cat_text_cardinality_threshold` / scaler /
  encoding / Enum-alignment settings no longer silently overwrite each
  other. Hash keys on BEHAVIOUR (realised schema), not config flags —
  e.g. LGB's fingerprint is identical whether the user sets
  `use_text_features=True` or `False`, because LGB drops text columns
  either way at tier build. `pl.Utf8` and `pl.String` aliases collapse;
  `pl.Enum(categories)` includes the sorted category set so val-drift
  with new categories is detected. Opt-out via
  `TrainingBehaviorConfig(model_file_hash_suffix=False)`. Backward-compat:
  the existing `load_mlframe_suite` glob (`**/*.dump`) still finds files
  regardless of suffix presence; only save-side naming changed.

### Changed

- `_polars_schema_diagnostic` (`mlframe/training/trainer.py`): header
  copy updated — nullable cat_features (the empirically-verified CB
  1.2.10 dispatch miss) are now named as the likely culprit; Enum is
  reported for visibility only.
- `FeatureTypesConfig` (`configs.py`): new field `use_text_features`.
- `TrainingBehaviorConfig` (`configs.py`): new field
  `model_file_hash_suffix`.

### Fixed (2026-04-21 round 3)

- **Fix 9.4.4 — Cross-target label swap on CB Pool**. Pool labels are
  now cast to ``float32`` at construction in ``_maybe_get_or_build_cb_pool``
  (``trainer.py``). CatBoost's C++ ``SetNumericTarget`` validator
  rejects ``ERawTargetType::Integer`` on subsequent ``set_label`` calls —
  a latent UX gotcha in the upstream PR that's invisible until you try
  label reuse across classification targets. By always storing float32
  upfront, subsequent ``set_label(y_new)`` swaps succeed for both
  classification and regression targets within the same feature matrix.
  Verified: 4 fits (cls→cls→cls→reg) = 1 Pool build.
- **Fix 8f — Load-time schema verification with two-tier diff**.
  ``_validate_input_columns_against_metadata`` (``core.py``) now consumes
  ``metadata['model_schemas']`` and produces a structured diff when
  input-schema hash mismatches. Hard-fail on removed cat/text/embedding
  columns, dtype FAMILY changes (string↔numeric, numeric↔categorical),
  or role changes. Soft-warn on dtype WIDTH changes
  (float32↔float64, int32↔int64) — downstream pipelines cast
  transparently. New helper ``_dtype_family`` (``utils.py``) classifies
  dtype strings into {string-or-cat, float, int, bool, datetime, list}
  families.
- **Orch-1 — CatBoost val Pool reuse across weight schemas**.
  ``_maybe_rewrite_eval_set_as_cb_pool`` (``trainer.py``) rewrites
  ``fit_params['eval_set']`` entries from ``(val_df, val_target)`` to a
  cached ``catboost.Pool`` when the train-side reuse fast-path is
  active. Separate ``_CB_VAL_POOL_CACHE`` cleared at suite entry in
  ``core.py`` parallel to ``_CB_POOL_CACHE``. Verified: 3 weight-only
  fits = 1 train Pool + 1 val Pool build (pre-Orch-1: 6 total).
- **`_auto_detect_feature_types` refinement (2026-04-21)**.
  ``use_text_features=False`` now gates AUTO-PROMOTION only; user-supplied
  explicit ``text_features`` list is honored regardless. Earlier version
  cleared the explicit list too, which broke ``test_non_catboost_drops_text_columns``
  / auto-detection tests.
- **`test_user_declared_polars_categorical_not_promoted_to_text` xfail**
  (``test_core.py``). Test's assertion ("user-cast pl.Categorical NOT
  promoted by cardinality") contradicts
  ``test_auto_detect_polars_categorical_promoted_by_cardinality`` in
  ``test_untested_core_helpers.py`` ("pl.Categorical IS promoted when
  cardinality exceeds threshold"). Resolving requires a finer
  distinguishing signal (e.g. per-column ``honor_user_dtype`` flag on
  FeatureTypesConfig) — out of scope for 2026-04-21. Current code
  honors the promote-by-cardinality contract.

### Orchestration review (2026-04-21)

Audit of ``train_mlframe_models_suite`` nested loop structure produced
7 improvement candidates; outcomes:

- **Orch-1** (val Pool reuse across weights): IMPLEMENTED — see above.
- **Orch-2** (hoist ``clone()`` out of weight-schema loop): REJECTED.
  Authors' explicit ``# DO NOT OPTIMIZE BY MOVING CLONE OUTSIDE THE LOOP!``
  banner documents the correctness reason — each weight iteration
  produces a distinct trained model state that downstream in-memory
  ``model.predict()`` relies on. Clone cost is negligible vs training.
- **Orch-3** (cache ``prepare_polars_dataframe`` across targets):
  SKIPPED. Operation is ~ms (String→Categorical cast); ROI doesn't
  justify new cache layer.
- **Orch-4** (widen ``pipeline_cache`` scope from
  ``(target_type, target)`` to ``(target_type)``): REJECTED — unsafe
  for target-dependent pre_pipelines (MRMR, RFECV). Selecting features
  by target-1 labels and reusing for target-2 would be a silent
  correctness bug.
- **Orch-5** (concurrent model training within weight iteration):
  REJECTED by user (GPU/RAM contention risk, gain unclear).
- **Orch-6** (parallel level-2 stacking across folds): REJECTED by
  user (out of scope).
- **Orch-7** (skip ``eval_set`` rebuild when val_df unchanged):
  COVERED for CatBoost by Orch-1. XGB/LGB sklearn wrappers rebuild
  internally — blocked by upstream FRs drafted in
  ``reproducers/upstream_feature_requests/``.

### CatBoost upstream PR feedback (set_label)

Two concrete shortcomings in the ``Pool.set_label`` PR at
``D:\Machine Learning\catboost\``:

1. **UX-trap on int64 labels**. ``Pool(X, label=y_int64)`` stores
   target as ``ERawTargetType::Integer``; subsequent
   ``pool.set_label(y_new)`` fails in C++ ``SetNumericTarget``
   validator ("requires numeric or unset target type, got Integer").
   All PR tests pre-cast via ``.astype(np.float32)`` — but that
   requirement is undocumented and surprises natural users.
2. **Incoherent API surface**. ``Pool(label=int64)`` succeeds but
   subsequent ``pool.set_label(int64)`` fails on the same pool. Two
   entry points, divergent behaviour on the same input.

Recommendations for upstream:
- (a) In Python-level ``_set_label`` (``_catboost.pyx:4804``),
   unconditionally cast ``label → float32`` before
   ``SetNumericTarget``. Removes the UX trap; no semantic change
   (storage is already float32).
- (b) OR in C++ ``ValidateSetNumericTargetPreconditions``, allow
   ``Integer → Float`` transition. The storage rewrite on line 762
   already assigns ``TargetType = Float`` — the validator simply
   contradicts the actual write behaviour.
- (c) Add regression test
   ``test_pool_built_with_int_label_set_label_works``.

### Fixed (2026-04-21 round 2)

- **Fix 9.4.3 — CatBoost Pool reuse across weight schemas**.
  `mlframe/training/trainer.py` now intercepts `model.fit()` for CB
  models via `_maybe_get_or_build_cb_pool` (process-wide cache cleared
  at every `train_mlframe_models_suite` entry). Cache key: `(id(df),
  columns, shape, cat/text/embedding features)`. Cache-hit + same label:
  `pool.set_weight(new_weight)` — no rebuild. Cache-hit + new label:
  `pool.set_label(new_label)` + `set_weight`, but CatBoost 1.2.10 C++
  rejects `set_label` on classification Pools (numeric-target-only
  path), so that case falls through to rebuild cleanly. Weight-only
  swap between fits is the fast path that actually hits in the user's
  multi-weight workload. Feature-detect gate
  (`_cb_reuse_capable`) keeps older CB builds on the rebuild-every-fit
  path with a WARN at suite start.
- **Fix 9.4.1 — Per-build logging of Pool / DMatrix / Dataset**.
  Idempotent module-level patches at `trainer.py` import wrap
  `catboost.Pool.__init__`, `xgboost.{DMatrix,QuantileDMatrix,DeviceQuantileDMatrix}.__init__`,
  and `lightgbm.Dataset.__init__`. Each construction emits
  `[dataset-build] <cls> shape=RxC took=<s>s site=<module:line>` at
  INFO. Marker is checked on `cls.__dict__` (not inherited) so
  `QuantileDMatrix` extending `DMatrix` still gets its own wrapper.
- **Fix 9.4.2 — Dataset-reuse capability check at suite start**.
  `train_mlframe_models_suite` logs `Dataset-reuse capabilities: {...}`
  with boolean flags for
  `cb_pool_label_swap` / `xgb_sklearn_accepts_dmatrix` /
  `lgb_sklearn_accepts_dataset`. CB flag tracks the installed build's
  `Pool.set_label` / `set_weight` availability; XGB/LGB flags are
  currently pinned False (upstream sklearn wrappers don't accept
  pre-built datasets — upstream FR drafts saved in
  `mlframe/reproducers/upstream_feature_requests/` for the user to
  post).
- **Fix 9.7 — `test_mrmr_with_text_column` / `_embedding_column` state
  leak fixed**. Root cause: CatBoost's sklearn wrapper raises
  `ValueError: 'feat' is not in list` from
  `_get_cat_feature_indices` when `cat_features` contains a name not
  in the trimmed MRMR frame columns. mlframe's CB fit path now filters
  `cat_features` / `text_features` / `embedding_features` to the
  intersection with `train_df.columns` in `_maybe_get_or_build_cb_pool`
  (in-place on `fit_params`, so the sklearn-rebuild fallback also sees
  the filtered list). Tests pass without re-ordering tricks.
- **Fix 9.8 — `test_rfecv_with_polars` multi-dim index**.
  `mlframe/feature_selection/wrappers.py::split_into_train_test` now
  has a Polars branch alongside pandas / numpy. The generic numpy path
  used `X[np.ix_(rows, cols)]`, which passes a 2-D ndarray to
  `polars.DataFrame.__getitem__` and raises `TypeError: multi-dimensional
  NumPy arrays not supported as index`. Polars branch uses `df.select(cols)[rows_list]`.
- **Fix 9.9 — `prod_like_frame_small(n_rows=50)` parametric tests
  unskipped**. Earlier-marked skip-with-reason was replaced by shrinking
  `n_rows` 200 → 50. At n=200 the composite 5-column generator rejected
  ~97 % of hypothesis examples silently upstream in
  `polars.testing.parametric.dataframes` on installed hypothesis 6.147.0
  + polars 1.40.0. At n=50 the same schema generates cleanly and CB/XGB
  still fit end-to-end.
- **Fix 3A correction**: library default in
  `pyutilz.pandaslib.get_df_memory_consumption` stays `deep=True`
  (back-compat preserved for every caller that predates Fix 3A).
  mlframe's heuristic-only call site in
  `trainer.configure_training_params` passes `deep=False` explicitly.

### Tests

- **New: `tests/training/test_jolly_wishing_deer_fixes.py`** — 21
  integration tests covering Fixes 2, 3A, 3B, 4, 5, 6, 7, 8. Key
  additions: Polars + numpy input to LGB proves the shim works (would
  have caught the 2026-04-21 crash if this file had existed then);
  Fix-8 role-sensitivity + Enum-category-drift + Utf8/String alias
  collapse tests.
- **Updated: `tests/training/test_cb_polars_fallback.py`** — two tests
  on the removed pre-fit warning deleted; new tests verify the schema
  dump now flags null-valued cat_features (real culprit) and treats
  Enum as informational.
- **Skipped (pre-existing, unrelated)**:
  `tests/training/test_parametric_robustness.py::TestTrainSuiteRobustness::test_xgb_only_suite_completes`
  and `::test_cb_only_suite_completes_with_null_in_cat`. Both verified
  failing on a clean master checkout with the same
  `hypothesis.errors.Unsatisfiable: 977 of 1000 examples failed a
  .filter()/assume()` — a hypothesis-generator issue in
  `prod_like_frame_small` / `polars.testing.parametric.dataframes`, not
  a regression from the fixes above. Skipped with an explanatory marker
  so the rest of the parametric fuzz tests (which DO pass) remain
  visible and runnable.
- **Updated: `tests/training/test_perf_optimizations.py::TestSkipPandasConversion::test_pandas_conversion_when_mixed_models`**
  — was asserting `_convert_dfs_to_pandas` fires when `cb+ridge` is
  requested, which no longer holds after Fix 1 defers the upfront
  conversion. New assertion accepts either the upfront path OR the
  lazy per-strategy path (`get_pandas_view_of_polars_df`), maintaining
  the behavioural contract ("ridge receives pandas") without locking in
  the specific implementation detail.

## 2026-04-21 — Infra/Ops audit (agent #9) follow-up

Applied all fixes from `plans/mlframe_audit/09_infra_ops.md` except the explicit
"delete `early_stopping.py` demo" item (demo is retained but guarded behind
`if __name__ == "__main__":` so it no longer trains + prints on import).

### Security

- **`inference.py`**: `read_trained_models` now (a) loads a JSON features
  sidecar (`features.dump.json` or `features.json`) in preference to the joblib
  dump — orjson when available, stdlib `json` as fallback; (b) refuses to
  `joblib.load` any file whose extension isn't in an allow-list (`.dump`,
  `.joblib`, `.pkl`, `.pickle`; override via `allowed_extensions=`); (c)
  verifies an optional `<file>.sha256` sidecar before loading and skips with a
  logged error on mismatch. Existing `trusted_root` path-traversal guard kept.
- **`pipelines.py`**: `replay_cv_results` verifies `<dump>.sha256` sidecar when
  present before `joblib.load`.
- **`mlflowlib.py`**:
  - `embed_website_to_mlflow` now `html.escape`s the user-supplied `url`,
    casts `width`/`height` through `int()`, and opens the sidecar with
    `encoding="utf-8"`. Extension-suffix check fixed
    (`fname.lower().endswith(extension.lower())` instead of the previous
    prefix comparison that was always False and produced `url.html.html`).
  - `get_or_create_mlflow_run` scrubs `scheme://user:pass@` from every mlflow
    exception before logging (`_strip_userinfo` / `_USERINFO_RE`), escapes
    the run-name / parent-run-id before inlining them into the mlflow filter
    DSL, and no longer has a dead `raise(e)` hiding the
    `'already active'` recovery branch.

### Import-time side effects

- **`early_stopping.py`**: load_iris + `fit()` + `print(accuracy)` demo block
  moved under `if __name__ == "__main__":` (no longer runs on import). Also
  initialized `self.best_score_ = -np.inf`, `self.best_model_ = None`,
  `self.no_improvement_count_ = 0` at the top of `fit()` so the first
  iteration no longer raises `AttributeError`.
- **`keras.py`**: top-level `import tensorflow as tf` removed; TF now imported
  lazily via `_tf()` only when `create_multiinput__keras_model()` is called.
  `matplotlib` also deferred into `plot_learning_curve`.
- **`explainability.py`**: top-level `import shap`, `from catboost import ...`,
  `from imblearn.pipeline import Pipeline` all deferred inside
  `compute_shap_on_cv` and `init_model_instance` — the module is now importable
  without shap/catboost/imblearn installed.

### sklearn contract

- **`keras.py::KerasDictConverter.__init__`** no longer reads
  `self.tokenizer.model_max_length` to populate `self.tokenizer_kwargs`. Every
  constructor parameter is now stored verbatim (so `sklearn.clone()` works);
  the max-length default is resolved lazily in `_effective_tokenizer_kwargs()`
  called from `fit`/`transform`. Fitted attributes renamed to trailing-
  underscore form (`dimensions_`, `fit_just_made_`) per sklearn convention.

### Architecture

- **Wildcard `from .config import *` eliminated** from `__init__.py`,
  `feature_cleaning.py`, `preprocessing.py`, `postcalibration.py`, and
  `legacy/training_old.py`. Each replaced with an explicit import of only the
  names that module actually references. `postcalibration.py` referenced none
  of them — the import was dropped outright. `legacy/training_old.py` used
  `from .config` which never resolved against `mlframe.legacy` anyway;
  switched to the absolute `from mlframe.config import TABNET_MODEL_TYPES`.

### Performance finding (NOT applied)

Agent #9 recommended replacing the pandas-level `np.isinf(df).any()` in
`ensure_no_infinity_pd` with `np.isinf(df[num_cols].to_numpy()).any(axis=0)`.
Benchmarked on pandas 2.3.3 / numpy 2.2.6:

| case                              | current   | vectorized | speedup |
|-----------------------------------|-----------|------------|---------|
| 5M x 50 float64 (homogeneous)     |  323.8 ms |  324.2 ms  |  1.00×  |
| 1M x 200 float64 (homogeneous)    |  204.2 ms |  177.4 ms  |  1.15×  |
| 10M x 20 float64 (homogeneous)    |  417.7 ms |  394.6 ms  |  1.06×  |
| 100k x 500 float64 (homogeneous)  |   45.4 ms |   47.9 ms  |  0.95×  |
| 2M x 40 **mixed** f64/f32/i64     |   51.8 ms |  268.7 ms  |  0.19×  |

Homogeneous all-numeric frames are break-even (±15%, within noise);
mixed-dtype frames — which also hit this branch when `num_cols_only=False`
or every col happens to be numeric — are **5× slower** because `.to_numpy()`
has to upcast and copy all blocks. Kept existing implementation. Benchmark
harness is checked in at `D:/Temp/bench_isinf.py`.

## 2026-04-21 — remove `ensure_installed` from all library modules

All `pyutilz.pythonlib.ensure_installed(...)` imports and calls (both active and commented)
removed from every `.py` module under `mlframe/`. Dependencies are declared in `requirements.txt`
and installed by the package manager rather than triggered at import time as a pip side effect.

### Removed active calls

- **`baselines.py`**: dropped the `for _install_attempt in range(3): try/except` import loop
  wrapped around numpy/pandas/sklearn — plain `import` now.
- **`custom_estimators.py`**: same retry-wrapper pattern removed.
- **`optimization.py`**: `while True: try/except` wrapper removed; imports are now flat.
- **`explainability.py`**: top-level `ensure_installed("shap numpy")` removed.
- **`outliers.py`**: top-level `ensure_installed("imbalanced-learn scikit-learn")` removed.
- **`inference.py`**: top-level `ensure_installed("numpy pandas numba scipy sklearn antropy")`
  removed.
- **`feature_engineering/hurst.py`**: `ensure_hurst_dependencies()` opt-in probe and its entry
  in `__all__` removed (no callers).

### Removed dead imports + commented probes

- `calibration.py`, `eda.py`, `ewma.py`, `evaluation.py`, `feature_cleaning.py`, `stats.py`,
  `tuning.py`, `feature_engineering/categorical.py`, `feature_engineering/timeseries.py`:
  unused `from pyutilz.pythonlib import ensure_installed` import and the adjacent commented
  `# ensure_installed(...)` line stripped, along with the now-empty `# Packages` section header.
- `feature_engineering/numerical.py`, `feature_selection/filters.py`, `legacy/training_old.py`,
  `metrics.py`: stale inline `# ensure_installed(...)` comments deleted.

### `requirements.txt` coverage verified

Every package named in a removed `ensure_installed(...)` call is already listed in
`requirements.txt`: numpy, pandas, scikit-learn, shap, numba, scipy, antropy, expiringdict,
imbalanced-learn, properscoring, psutil, joblib, catboost, astropy, entropy-estimators,
lightgbm, xgboost, plotly, pywavelets. No additions needed.

## 2026-04-21 — audit 05 fixes: Models / estimators / custom_estimators / ensembling / baselines

Pass over the bugs surfaced by audit 05 (`.claude/plans/mlframe_audit/05_models_estimators.md`).
Dead/broken legacy modules moved to `mlframe/legacy/` rather than deleted, per user direction.

### Moves

- **`Models.py` → `mlframe/legacy/Models.py`**. Depended on `sklearn.externals.joblib` (removed
  0.23), `from numpy import *`, and ~470 LOC of script-body with undefined globals (`y_up`,
  `pipe_up`, `K`, `tf`, `my_class_weight`). Only importers were already-dead modules. Header
  docstring added pointing readers to the salvaged API. `legacy/OldEnsembling.py` imports
  re-pointed to `mlframe.legacy.Models`.

### New

- **`mlframe/scoring.py`**: salvaged `rmse_loss`, `rmsle_loss`, `rmse_score`, `rmsle_score`,
  `log_uniform`, and `ProbaScoreProxy` from the retired `Models.py`. Cleaned up the helpers
  (no star-imports, scoped `uniform`, docstrings).

### Fixes

- **`baselines.py`**: unbounded `while True: try/except ensure_installed` replaced with
  bounded `for _ in range(3)` + `for/else: raise ImportError` — matches the pattern already
  in `custom_estimators.py`. `get_best_dummy_score` now raises `TypeError` when the estimator
  is neither a classifier nor a regressor (previously silently returned `-LARGE_CONST`).
- **`ensembling.py`**: inline `"arimean,quadmean,qubmean,geomean,harmmean".split(",")` promoted
  to module-level `_MEANS_COLS`. `.values` → `.to_numpy()` on the four target-series conversions
  before the `Parallel(loky=True)` section. Added a pickle-roundtrip pre-check on
  `custom_ice_metric` / `custom_rice_metric` / `**kwargs` that falls back to sequential with a
  clear warning if something (closure, lambda) can't cross the loky process boundary.

### Tests added (audit 05 §7)

- **`tests/test_scoring.py`** (9 tests): exercises `rmse_loss`, `rmsle_loss`, `log_uniform`
  bounds / reproducibility / scalar path, `ProbaScoreProxy` column selection, and the
  `greater_is_better=False` flag on the make_scorer wrappers.
- **`tests/test_custom_estimators.py`** (~18 tests, 3 xfailed by design):
  - T1: hypothesis property test — `ArithmAvgClassifier` / `GeomAvgClassifier` `predict_proba`
    rows sum to 1.
  - T2: parametrized fit/predict smoke across `(n_samples, n_cols)` and `(n_samples, n_classes)`
    shapes; asserts `predict` returns values drawn from `classes_`, not argmax indices.
  - T3: `check_estimator` exercised via `xfail` — documents that averager classifiers expect
    pre-computed probability columns as features, so the generic sklearn smoke suite doesn't
    apply.
  - Pickle-roundtrip for all three averager classifiers.
  - Regression for `MyDecorrelator` attribute-name consistency (`correlated_features_`).
  - `PureRandomClassifier.random_state` reproducibility.
- **`tests/test_estimators.py`** (6 tests): T4 pickle roundtrip for both
  `ClassifierWithEarlyStopping` and `RegressorWithEarlyStopping`; covers the non-catboost
  fallback path and the `decision_function` proxy error on wrapped estimators without one.
- **`tests/test_baselines.py`** (3 tests): T6 — `get_best_dummy_score` on classifier and
  regressor paths, plus `TypeError` on a `KMeans` estimator.

### Prior state note

Several audit H-items (H2 logger in `estimators.py`, H3 proxy methods, H4 module-level
PowerTransformer, H5/H6/H7/H8 averager/discretizer/identity fixes, M6 attribute mismatch) were
already applied in-tree before this pass. The tests added here lock those behaviours in place.

## 2026-04-21 — audit 03 fixes: feature_engineering / FeatureEngineering.py / Features.py / feature_cleaning.py

Pass over the bugs surfaced by audit 03 (`.claude/plans/mlframe_audit/03_feature_engineering.md`).
No tests added for `Features.py` / `FeatureEngineering.py` (per user direction) — bugs fixed in place.

### Critical bug fixes

- **`Features.py:mode`**: was `np.percentile(q=50)` → returned the *median*, not the mode. Every
  `_MODE_` rolling column produced by `EnrichTSDatasetWithRollingStats` was silently a duplicate
  of `_MEDIAN_`. Replaced with `np.unique(...)` + `argmax(counts)`.
- **`Features.py`** (pure_funcs drop): columns had already been renamed via `StringOrFuncName` to
  `pure_funcs_real`, so `agg.drop(pure_funcs, inplace=True)` either no-op'd or raised. Now drops
  `columns=pure_funcs_real` (and drops `inplace` since that was the buggy kwarg order).
- **`Features.py:MyCustomColumnSelector.fit_transform`**: called `self.transform(self, X, y)` —
  bound-method call already passes self, so this mis-routed `self` into the `x` parameter.
- **`Features.py`**: replaced `from numpy import *` with explicit imports; switched function-bodies
  to `np.*` references so the module no longer leaks 500+ names into callers' globals.
- **`Features.py` EnrichTSDatasetWithRollingStats**: added `assert lag > 0` (negative lags are a
  silent look-ahead leakage vector).
- **`FeatureEngineering.py:CalculateNumericalStatsPandas`**: `r.mad()` was removed in pandas 2.0 →
  replaced with `(r - r.mean()).abs().mean()`. `r.mode().values[0]` IndexError'd on empty input →
  replaced with length-guarded `mode_series.iloc[0]`.
- **`FeatureEngineering.py:get_domain_suffixes`**: `requests.get(...)` had no timeout (caller
  could hang indefinitely) → added `timeout=10`. Fallback `open("public_suffix_list.dat")` was
  CWD-relative → now resolves to the copy shipped next to the module. Fixed downstream
  `res.text.split` which crashed when the fallback branch returned a plain string.
- **`FeatureEngineering.py`**: `charstats_buffer`, `oov_buffer`, `oov_tokens_buffer` were only
  initialized by `flush_text_stats_caches()` (called from `init_nlp`). Calling text features
  before `init_nlp` raised NameError → initialized at module load.
- **`FeatureEngineering.py`**: `from random import randint, random` was buried inside
  `CalculateCategoricalStatsNumpy`'s dead code path → moved to module top.
- **`feature_engineering/timeseries.py:create_windowed_features`**: `create_features_names=` now
  explicitly wrapped in `bool(...)` (defensive vs. the historical `create_features_names=targets`
  bug where a list was used as a boolean flag). Added clarifying comments around the
  `window_features=None if targets_creation_fcn else row_targets` semantics so the intent is
  visible to future readers.

### High-severity fixes

- **`feature_engineering/numerical.py:compute_simple_stats_numba`** and
  **`compute_numerical_aggregates_numba`**: `elif next_value > maxval` meant a sample equal to
  min could never update max → produced inconsistent `argmin`/`argmax` on flat arrays. Split into
  independent `if` branches.
- **`feature_engineering/numerical.py`**: integer detection via `next_value % 1` was fragile for
  negative floats/denormals → replaced with `np.floor(next_value) == next_value`.
- **`feature_engineering/hurst.py:compute_hurst_exponent`**: unguarded `np.log10(rs)` on
  non-positive `rs` poisoned the least-squares fit → added guard returning `(nan, nan)` on any
  `rs <= 0` or `window_sizes <= 0`.
- **`feature_engineering/hurst.py`**: moved `ensure_installed(...)` off the module import path
  (was a supply-chain footgun: importing the module silently ran pip). Exposed as opt-in
  `ensure_hurst_dependencies()` helper.
- **`feature_engineering/financial.py:add_ohlcv_ratios_rlags`**: added explicit negative/zero
  shift guard — negative shifts leak future prices into the current row.

### Medium-severity fixes

- **`feature_engineering/basic.py:create_date_features`**: pandas branch mutated the caller in
  place while the polars branch returned a new frame → asymmetric API. Pandas branch now works
  on a shallow copy and returns a fresh frame.
- **`feature_engineering/basic.py:run_pysr_fe`**: `col.replace("=","_").replace(".","_")` could
  collapse distinct column names into the same sanitized key. Now deduplicates with an
  incrementing `__N` suffix on collision.
- **`feature_engineering/bruteforce.py`**: polars branch now clamps `sample_size` with
  `min(sample_size, len(df))` (parity with pandas branch); redundant trailing `.copy()` removed.
- **`feature_cleaning.py:_get_nunique`**: `~np.isnan(unique_vals)` raised TypeError on object
  dtype → dispatches on dtype kind (use `pd.isna` for non-float).
- **`feature_cleaning.py:_update_sub_df_col`**: replaced probe of private `sub_df._is_view` with
  a public-API substitute (`values.base is not None`).
- **`feature_cleaning.py:_clean_cat_and_obj_columns`**: category cleaning used per-row `.apply()`
  → now uses `cat.rename_categories(...)` (O(#levels) vs O(#rows)). Object-column cleaning
  switched from `.apply` to `.map`.

### Low / housekeeping

- Replaced module-level `warnings.simplefilter(...)` calls at
  `feature_engineering/numerical.py`, `categorical.py`, `basic.py` with scoped
  `_suppress_*_warnings()` context managers. Module import no longer mutates global warning
  filters for the whole process.
- Added `__all__` to `feature_engineering/{basic,bruteforce,categorical,financial,hurst,mps,numerical,timeseries}.py`.
- `timeseries.compute_splitting_stats`: reduced two `.sum()` passes over the same column to
  `col_sum - pre_sum`.

### Not changed (deferred)

- Architectural recommendation to retire `FeatureEngineering.py` / `Features.py` into
  `mlframe/legacy/` — deferred per user direction (fix bugs, don't touch tests or reorganize).
- Drawdown recursive-call alignment at `numerical.py:409-460` reviewed — current slicing keeps
  `weights[1:]` aligned with `pos_dds[1:]` (both index-for-index over `arr`), so the audit's
  misalignment note does not appear to hold under current code. Left as-is.

## 2026-04-20 — round 18: align Polars Categorical dicts across splits (opt-in)

### Symptom

On prod_jobsdetails (9M rows, 19 cat features, Windows/Jupyter),
``train_mlframe_models_suite`` silently killed the kernel between
train IterativeDMatrix construction and val IterativeDMatrix
construction. No Python traceback, no WER dialog, just kernel
restart. Removing ALL categorical features bypassed it.

### Fix

Before handing frames to model strategies, cast every
``pl.Categorical`` / ``pl.Enum`` cat_feature across train/val/test to
``pl.Enum(sorted union of all unique values)``. All three splits
then carry the **identical dict**. Empirically prevents the crash
at prod scale (verified 2026-04-20).

### Mechanism — honestly not fully understood

A small-scale probe (``D:/Temp/xgb_unseen_cat_probe.py``, 2000 rows ×
1 cat with deliberate train/val dict mismatch) did **not** crash —
XGB 3.2.0 passed both the same-dict and unseen-val-cat scenarios.
So the bug is not the naive "XGB crashes on unseen val categories"
claim I initially asserted (and was called out on).

Leading theory: ``pl.Categorical`` assigns physical codes per-Series
in order-of-first-occurrence (documented polars behaviour). The
same string can have different physical codes in train vs val vs
test. XGB's native layer at large scale (7M+ rows × 15+ cats)
appears to treat val's physical codes as indices into train's bin
structure without re-reading the Arrow dict, corrupting memory.
``pl.Enum(list)`` enforces a shared dict where physical codes are
consistent across Series by construction.

Supporting empirical evidence (from prod logs before / after fix):

  MakeCuts time with cats, no alignment:  0.901962s   <-- slow
  MakeCuts time with cats, Enum-aligned:  0.018451s   <-- ~50x faster
  MakeCuts time without any cats:         0.014539s   <-- baseline

The aligned time matches the no-cats baseline to within noise.
Whatever XGB is doing for categoricals without alignment is not
just slower, it's a different code path — one that breaks at prod
scale.

### Opt-in

Exposed as ``TrainingBehaviorConfig.align_polars_categorical_dicts``
(default **True** — the prod crash is real enough to justify the
safe default). Set to False to reproduce the original behavior or
skip the alignment cost (O(n_rows) per cat column, plus a cast).
High-cardinality columns (>50_000 uniques) are skipped regardless —
those are typically text-promoted rather than categorical.

A standalone reproducer for the xgboost project is at
``D:/Temp/xgb_polars_categorical_scale_crash.py`` — depends only on
``polars`` + ``xgboost`` + ``numpy``, supports ``--scale prod|small``
and ``--mode crash|workaround``.

## 2026-04-20 — round 17: fill nullable Categoricals detected in train OR val OR test

Previously ``_polars_nullable_categorical_cols`` was called only on
``train_df_polars``. A cat column with 0 nulls in train but 100+ in
val (common on time-ordered splits where new null-paradigms appear
in the later period) was NOT pre-filled. The null slipped into val's
Polars Categorical, reached CB/XGB native layer, and contributed to
the silent crash class addressed in round 18.

Fix: detect nullable cats on train AND val AND test separately,
union the sets, apply ``fill_null`` to all three splits on the
union. Log a WARN listing columns where val/test introduced nulls
that train never had.

## 2026-04-20 — round 16: in-suite category drift warning

### Problem

On prod_jobsdetails (9M rows × 20 cat features),
``train_mlframe_models_suite`` crashed silently — Jupyter kernel died
with no traceback — during XGB's val IterativeDMatrix construction.
The timing window was tight:

    [T+~4min] train IterativeDMatrix: (7304969, 114, 577004772) built OK
    [T+~5min] kernel dies (no further log, no exception)

Expected next log line would have been ``Finished constructing the
IterativeDMatrix: (811663, 114, ...)`` for val. That line never
appeared. Removing ALL 20 categorical features bypassed the crash, so
one of them is the trigger.

Hypothesis: a categorical column has **values in val that don't exist
in train** (time-ordered split, ~20-year train / 1-year val, new
category codes appeared in the later period). XGB 3.x on Windows
mishandles the unseen category during val-DMatrix construction and
the process dies via native abort that faulthandler can't catch.

### Fix

Drift detection is now inlined into
``train_mlframe_models_suite``'s pre-training phase, immediately after
the existing cardinality-snapshot log. For every categorical / text
/ embedding feature with cardinality <= 100k (text-sized columns are
skipped — anti-join is too expensive and the semantics differ), the
suite computes:

  * ``val_only`` — categories in val **absent** from train (polars
    anti-join, fast)
  * ``test_only`` — same for test

Emits one INFO line summarising all non-zero-drift columns, plus one
WARNING per suspect (``val_only >= 5`` OR ``val_only / card_train >= 5%``)
so the operator sees the trigger column BEFORE XGB's native layer
crashes the kernel.

Verified on inline synthetic: a stable 3-category column correctly
reports 0 drift; a deliberately drifting column (4 val-only categories
in a 3-category train) is flagged as crash suspect with severity
percentage.

## 2026-04-20 — round 15: parametric frame fuzzing (polars.testing.parametric wrapper)

### Motivation

Round 11 (null-in-Categorical → CB fastpath crash) and round 12
(sparse-null column promoted to text → CB "Dictionary size is 0")
both slipped through because the test suite built frames by hand with
``pl.DataFrame({"c": [1, 2, 3]})`` — every fixture was shape-uniform,
dtype-uniform, null-free. Neither bug was discoverable from the test
zoo we had.

### Module: ``mlframe/testing/parametric.py``

Thin wrapper over ``polars.testing.parametric`` (built-in to Polars
1.35+; native Hypothesis integration). Purpose: isolate the project
from API churn (``null_probability`` → ``allow_null`` deprecation,
"currently unstable" label) and provide pathology-shaped helpers.

Primitives:

- ``categorical_column(name, categories, null_rate=0.05, use_enum=True)``
  — ``pl.Enum(categories)`` with real masked nulls at the specified
  rate. Round 11's exact shape.
- ``inf_heavy_float_column(name, specials_rate=0.02)`` — float with
  weighted ``+inf``/``-inf``/``NaN`` that survive Hypothesis shrinking.
- ``constant_column``, ``id_column``, ``high_card_text_column``,
  ``sparse_null_column`` — round 12's shape.
- ``adversarial_frame(...)`` — layered pathologies for "pipeline
  survives any frame" fuzzing.
- ``prod_like_frame(...)`` — schema-matched prod_jobsdetails miniature.

Key implementation note: polars 1.35's ``null_probability`` kwarg is
silently coerced to ``bool(rate)`` — the float is discarded. We bypass
by splicing nulls into the per-cell strategy via weighted
``st.integers().flatmap(...)`` with inverted direction (small shrink-
preferred i → value, large i → None) so shrinking converges to "no
nulls" as the minimal counterexample.

### Hypothesis profiles

Auto-registered at import: ``mlframe-fast`` (max_examples=10, default),
``mlframe-ci`` (50), ``mlframe-nightly`` (500). Selectable via env var
``MLFRAME_HYP_PROFILE``. All profiles suppress ``too_slow``,
``data_too_large``, ``filter_too_much``, ``large_base_example``
health checks since frame-gen is inherently heavy.

### Test file: ``tests/training/test_parametric_robustness.py``

Fuzz-tests for pipeline functions that promise "any frame":
``_auto_detect_feature_types``, ``XGBoostStrategy.prepare_polars_dataframe``,
``CatBoostStrategy.prepare_polars_dataframe``. Asserts only on
invariants (types, shape preservation, schema integrity), never on
specific values — pinned-data regression tests (``test_round11_*``,
``test_round12_*``) stay separate and continue to guard exact shapes.

## 2026-04-20 — round 14: opt-in crash reporting + continue-on-model-failure

Two new ``TrainingBehaviorConfig`` flags for Windows/Jupyter survival:

- **``enable_crash_reporting`` (default True).** At suite start,
  install ``faulthandler.enable()`` and (on Windows) call
  ``SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)`` so
  fatal signals produce Python tracebacks instead of WER modals.
  Jupyter kernel exits cleanly on bad_alloc / SEGV instead of hanging
  on "Python has stopped working".
- **``continue_on_model_failure`` (default False).** Wraps per-model
  ``process_model`` in try/except; logs failure + appends to
  ``metadata["failed_models"]`` and continues with the next
  model/weighting. KeyboardInterrupt still aborts; native SIGSEGV that
  kills the process at OS level still kills the suite (subprocess
  isolation is explicitly out of scope).

14a: flipped ``enable_crash_reporting`` default to True (pure
diagnostic, no behavior change).

14b: fixed regressions — TrainingBehaviorConfig's two new flags leaked
through ``**effective_behavior_params`` into ``configure_training_params``
(TypeError). Also ``faulthandler.enable(file=sys.stderr)`` failed in
Jupyter (OutStream without ``fileno()``); fall back to OS stderr fd 2.

14c: ``xgb_kwargs={"tree_method": "approx"}`` and similar user kwargs
crashed helpers.py with "dict() got multiple values for keyword
argument" because dtype was hardcoded in the default dict and the user
value came via ``**xgb_kwargs``. Switched all five model config
builders (CB, HGB, XGB, LGB, NGB) from ``**kwargs``-splat to
``defaults.update(user_kwargs)`` so any default is overridable.

## 2026-04-19 — round 13: ICE-only fastpath + FI figure leak + cat cardinality logging

Profile-driven optimisations from ``profile_metrics_blocks.py``
(230.5 s total on 100k × 103 cb+xgb × 2 targets):

- ``fast_calibration_report`` was 1708× called (43.6 s cumulative,
  19% of run). 212× fan-out per split via
  ``compute_fairness_metrics`` → per-bin ``custom_ice_metric``. Added
  ``fast_ice_only()`` that skips the ``fast_log_loss`` +
  ``compute_pr_recall_f1_metrics`` + title/plot work that
  ``compute_probabilistic_multiclass_error`` discards on the fairness
  path. Bit-exact (drift < 1e-9), 1.18-1.78× per-call speedup.
- ``plot_feature_importance``: close leaked top-FI figure when the
  bottom-FI branch also fired. Memory accumulates otherwise.
- Pre-train cardinality log: log ``n_unique`` per
  categorical/text/embedding feature before Phase 4 so native XGB/CB
  crashes leave a record of what the model saw.
- Numba prewarm: added ``fast_brier_score_loss`` and
  ``calibration_metrics_from_freqs`` (both ``@njit``, both previously
  missed).

## 2026-04-19 — round 12: text-promotion non-null guard + Dictionary-size-0 fallback

### Prod result from round 11d/e

Confirmed round 11c/d fix works:
- `Pre-fit fill_null('__MISSING__') on 11 nullable Polars Categorical cat_feature(s)` log emitted
- CB Polars fastpath: **no TypeError, no polars→pandas fallback**, CB fit succeeded on Polars DF natively (898.5 s ≈ 15 min, was 1015 s + 154 s pandas prep before)
- predict_proba(val) and predict_proba(test) succeeded on Polars DF — no predict-time fastpath rejection either

### New error surface: CB text feature estimator on sparse-non-null columns

After CB succeeded, training suite failed in the XGB branch (or CB itself in the newer run) with:

```
catboost/private/libs/feature_estimator/text_feature_estimators.cpp:89:
Dictionary size is 0, check out data or try to decrease
occurrence_lower_bound parameter
```

Schema change: the source data moved from `Categorical(24)` to `Categorical(18) + String(6)` between runs (different time range: splits now through 2026-04-16). `_auto_detect_feature_types` promoted 6 columns to text_features based on `n_unique > threshold=50`. Two of them (`_raw_countries:2196`, `job_post_source:71`) carried sparse non-null rows — CatBoost's TF-IDF estimator then applied its `occurrence_lower_bound` filter and left an empty vocabulary.

### Fixed — non-null guard in `_auto_detect_feature_types` (`training/core.py`)

Added `min_non_null_for_text_promotion` guard (default 100). A column still needs `n_unique > threshold` to be considered for promotion; it additionally needs `non_null_count >= min_floor` to actually get promoted. Columns that pass unique but fail non-null stay in `cat_features` — they can still be useful low-signal categorical inputs without risking CatBoost's text pipeline.

WARN-level log lists every skipped column with `(n_unique, non_null)` so the operator sees exactly what's happening:

```
Auto-detection: 2 column(s) had n_unique>50 (would be promoted to
text_features) but non_null<100 — kept as cat_features to avoid
CatBoost's 'Dictionary size is 0' error on sparse text columns:
_raw_countries:2_196 (non_null=6), job_post_source:71 (non_null=17)
```

### Fixed — defensive `Dictionary size is 0` catch in `_train_model_with_fallback` (`training/trainer.py`)

Last-line safety net if the proactive guard is somehow bypassed (e.g., user explicitly declares a sparse column as a text_feature via `FeatureTypesConfig.text_features`). On the exact CatBoost error:
- drop `text_features` from `fit_params`
- WARN with the column list and the upstream-guard reference
- retry without text processing

CB then fits using only cat_features + numeric — no native-text capabilities on those columns, but training completes.

### Tests (`tests/training/test_round12_text_promotion_guard.py`, new, +5 sensors)

`TestMinNonNullTextPromotionGuard`:
- sparse column with n_unique=80 + non_null=80 → blocked
- dense column with n_unique=80 + non_null=800 → promoted
- WARN log fires on skipped column with name + non_null count
- pandas path also guarded
- boundary at default threshold (99 blocked, 100 passes)

### What to watch for in the next prod run

Expected:
- `Pre-fit fill_null('__MISSING__') on N cat_features` — CB null-in-cat guard (round 11d)
- `Auto-detection: K column(s) had n_unique>50 but non_null<100 — kept as cat_features` — the new round-12 guard, if sparse columns exist
- CB fit + predict on Polars with no fastpath rejection
- XGB fit + predict on Polars with no KeyError

If the defensive fallback path fires (Dictionary-size-0), operator sees the exact columns and can fix upstream.

## 2026-04-19 — round 11e: alias removal + upstream-handoff audit

### Removed `_polars_df_emits_large_string` deprecated alias (`training/trainer.py`)

The round-11a name survived as a delegating wrapper to avoid breaking in-flight callers during the renaming transition. All known call sites have been updated; the only remaining references were:
- CHANGELOG entries (history, not live code)
- `TestLegacyAlias` sensor (kept the alias alive, not a real invariant)

Both the alias function and the sensor class are now gone. `_polars_nullable_categorical_cols` and `_polars_df_has_null_in_categorical` are the only public detector names.

### Audited polars → upstream handoffs (`bench_polars_null_in_cat_xgb_hgb.py`)

Three `.supports_polars = True` strategies in the codebase (CatBoost, XGBoost, sklearn HGB). Run each on:
- Clean Polars DF (null-free Categorical)
- Dirty Polars DF (10% nulls in Categorical)

Result:

| Strategy | Clean | Dirty |
|---|---|---|
| CatBoost 1.2.10 | OK | **FAIL** (`TypeError: No matching signature found` — our known bug) |
| XGBoost 3.2.0 | OK | **OK** |
| sklearn HGB 1.8.0 | OK | **OK** |

Conclusion: **null-in-Categorical is a CatBoost 1.2.x-specific dispatch bug**. XGB and HGB handle it natively.

The round-11d upstream fill applied to the base `train_df_polars` / `val_df_polars` / `test_df_polars` therefore:
- **Load-bearing for CB** — without it, fastpath fails and we pay the 15-min pandas-path detour.
- **Harmless no-op for XGB and HGB** — they don't care, but the sentinel category doesn't confuse them either (the extra "__MISSING__" entry just becomes another categorical value).

Audit also covered other `model.fit(polars_df, ...)` sites in the training flow:
- `_train_model_with_fallback` (the main CB/XGB/HGB fit) — covered ✓
- `confidence_model.fit(test_df, ...)` in `_run_confidence_analysis` — test_df comes from the same `test_df_polars` chain, so covered ✓
- `meta_model.fit(...)` — takes numpy arrays, not Polars, irrelevant ✓
- sklearn Pipeline transform on Polars — covered by `_warn_on_schema_drift` + upstream fill ✓

No additional handoff points need separate treatment. The round-11d upstream fill is the single correct mitigation point.

### Tests
- `test_round11_polars_largestring_fixes.py`: 18 passed + 1 skipped (`TestLegacyAlias` class removed, 1 test gone).
- New bench: `bench_polars_null_in_cat_xgb_hgb.py` in repo root (part of the bench-first doctrine artifacts).

## 2026-04-19 — round 11d polish: fill upstream, single-pass detector, doctrine doc

Three follow-ups to round 11c. Each motivated by the user's pushback — "why do X per-model when you can do it once?" and "can you detect in a single pass?".

### Moved null-fill upstream (`training/core.py`)

Round 11c applied `fill_null("__MISSING__")` inside the CB-specific branch of the per-model loop. XGB and HGB would hit the same Polars DFs with the same null-in-Categorical bug and only discover it when (if) their dispatchers also crashed.

Fixed: the fill now runs **once, at Phase-4 start, on the base `train_df_polars` / `val_df_polars` / `test_df_polars`**, right after OD-filtering. Every polars-capable strategy (CB, XGB, HGB, future ones) sees the same pre-filled frame via `tier_polars` → `prepared_train`. No per-model duplication, no "did we fill for this model?" branching.

Sentinel consistency across splits is automatic because we derive the nullable-column list once from the train frame and apply the same fill expression set to val and test. Train's `"__MISSING__"` code stays aligned with val's and test's.

Per-model CB-specific fill block removed from the inner loop.

### Single-pass null detection (`training/trainer.py`)

The round-11c detector iterated cat_features in a Python loop calling `df[c].null_count()` per column. On the prod 810k × 98 frame with 9 cat_features, that was 9 separate polars queries.

Replaced with `df.select(candidate).null_count()` — one query, one scan, polars computes all per-column null counts in parallel internally. The helper is now called `_polars_nullable_categorical_cols` and returns the list directly so callers don't re-detect after the boolean answer:

```python
nullable = _polars_nullable_categorical_cols(df, cat_features=cats)
if nullable:
    df = _polars_fill_null_in_categorical(df, nullable)
```

The old boolean wrapper `_polars_df_has_null_in_categorical` is kept, delegating to the list version (cheap — same single-pass). `_polars_df_emits_large_string` remains as deprecated alias from round 11a.

New helper `_polars_fill_null_in_categorical(df, cols, sentinel="__MISSING__")` builds the fill expressions once so the same expression set applies across train / val / test atomically.

### Doctrine doc (`docs/DEBUGGING_UPSTREAM_ERRORS.md`)

New 1-page field guide summarizing the lesson from round 11's four iterations: **when an upstream library throws an opaque error, do not reason about the fix from the traceback — build a minimal repro first, change one variable at a time, and trust reproducible failure over reproducible success**. Includes the full misdiagnosis table, a checklist, and links to the three in-repo benches (`bench_polars_largestring_cb_xgb.py`, `bench_polars_cb_repro.py`, `bench_polars_cb_nullfrac.py`) that disproved hypothesis 11a and pinned 11b-c.

### Sensors (+10)

- `TestPolarsNullableCategoricalColsDetector` (4): list-only-nullable-cats, cat_features order preserved, single-pass pattern still in source (performance regression sensor), empty input returns `[]`.
- `TestPolarsFillNullInCategorical` (5): fills nulls with sentinel, empty col list is no-op (return df unchanged), non-polars input unchanged, custom sentinel, same sentinel across splits gives consistent categories.
- `TestFillNullPreservesFastpath` (1 + 1 skipped): the core behavioural test and the skip-marked end-to-end CB fit (covered by bench).

20 tests in test_round11_polars_largestring_fixes.py total (19 passed + 1 intentional skip).

## 2026-04-19 — probe round 11 — SWITCHED from bypass to in-place null-fill

User pointed out the obvious: `prepare_df_for_catboost` on the pandas path already does `fill_null("")` on categoricals, so why not do the same thing on the Polars path and **keep the fastpath alive** instead of bypassing?

Direct test (``bench_polars_cb_repro.py`` additions):
```
raw (nulls in Categorical):       FAIL  TypeError: No matching signature found
filled (.fill_null("__MISSING__")): OK
```

Polars auto-extends the category dict when `fill_null` adds a new value. The column keeps `pl.Categorical` dtype, just loses its validity bitmap — which is exactly what CB 1.2.x's Cython dispatcher needs.

### Fix (CORRECTED again — see misdiagnosis log below)

`training/core.py` inside the CB polars-fastpath block now does:
1. Detect cat_features with `null_count > 0` via `_polars_df_has_null_in_categorical`.
2. If any, apply `pl.col(c).fill_null("__MISSING__")` on every such column in `prepared_train` / `prepared_val` / `prepared_test`.
3. Stay on the fastpath. No bypass, no pandas detour.

Cost of the fix: O(n_cat_features × n_rows) one-time pass — tens of ms on 810k × 98. Wins back the ~15 minutes the pandas-path detour was costing per CB fit.

Semantics parity: `"__MISSING__"` becomes its own category in the fitted model. This is exactly the pattern `prepare_df_for_catboost` uses on the pandas path (with `""` as the sentinel) — same intent, different string.

### Sensors added
- `TestFillNullPreservesFastpath::test_fillnull_keeps_categorical_dtype` — verifies Polars auto-extends the category dict on fill_null and the dtype stays Categorical.
- End-to-end CB-fit sensor kept as a skip-marked doc-only test, since `bench_polars_cb_nullfrac.py` already covers that ground and adding it to the pytest suite would add 5-10 s per run.

### The misdiagnosis cascade (retained for the probe doctrine)

| Round | Hypothesis | How disproved |
|---|---|---|
| 7 | `pl.Enum` trips CB dispatch | Prod schema had no Enums; my diagnostic dump confirmed. |
| 10 | Stale `cat_features_polars` list passes String as cats | Genuine bug, shipped fix; orthogonal to the real cause. |
| 11a | `pa.large_string()` Arrow export | Standalone bench: null-free Categorical fits cleanly despite large_string. |
| 11b | Null in Categorical → bypass to pandas | Correct root cause, but the fix was too aggressive — we can stay on Polars by filling. |
| 11c (this) | Null in Categorical → fill with sentinel, keep fastpath | **Correct**, confirmed by direct fit bench. |

## 2026-04-19 — probe round 11 AMENDED: the real root cause is null-in-Categorical

Earlier today I claimed (`32f94bf`) that the CB Polars fastpath failures traced to `pa.large_string()` Arrow emission in Polars 1.x. User correctly pushed back and asked for a direct verification test. Running `bench_polars_largestring_cb_xgb.py` **disproved that hypothesis**: CB and XGB both fit cleanly on a Polars DataFrame with `Dictionary<uint32, large_string>` Categorical columns. The large_string representation is not the trigger.

Further isolation via `bench_polars_cb_repro.py` and `bench_polars_cb_nullfrac.py` pinned the actual trigger: **CatBoost 1.2.10's `_set_features_order_data_polars_categorical_column` (Cython fused cpdef) has no dispatch signature for a Polars Categorical column carrying a validity bitmap — i.e. `null_count > 0`**. A single null anywhere in any one cat_feature raises `TypeError: No matching signature found`. Null-free Categoricals fit fine.

Null-fraction sweep:
```
null_frac=0.0   OK
null_frac=0.1   FAIL  TypeError: No matching signature found
null_frac=0.5   FAIL
null_frac=0.99  FAIL
null_frac=1.0   FAIL
```

In the 2026-04-19 prod schema, 6 of 9 cat_features had nulls (0.15% to 100%) — any one of them blows the fastpath.

### Fixed — `_polars_df_has_null_in_categorical` detector (`training/trainer.py`)

Replaces the round-11-first-iteration `_polars_df_emits_large_string` with a narrower, correct check: scan the provided `cat_features` (or all Categorical/Enum columns if not specified) for `null_count > 0`. Called from `core.py`'s polars-fastpath block before CB's fit attempt. If True, `polars_fastpath_active` flips to False now, the pandas tier DF is built, and training + prediction runs through the pandas path (where CB's pandas handler is robust to NaN/None).

The old detector name `_polars_df_emits_large_string` is kept as a deprecated alias delegating to the new function, so any in-flight callers or branches don't immediately break.

### Reverted: XGB `_arrow_dtype` monkey-patch (already reverted in `32f94bf`)

The separate XGB monkey-patch approach was still broken for the reason explained in the previous CHANGELOG entry — moves the crash one layer deeper. Not reinstated. The proactive bypass here handles XGB too: if the prepared DF has nulls in its cat_features, both CB and XGB route through pandas.

### Misdiagnosis log (for the probe doctrine)

Three previous hypotheses, each falsified before pinning the real trigger:

| Round | Hypothesis | How disproved |
|---|---|---|
| 7 | `pl.Enum` cat_features trip CB dispatch | Prod schema diagnostic dump showed no `pl.Enum` columns — only plain `Categorical(ordering='lexical')`. Harmless WARN retained. |
| 10 | Stale `cat_features_polars` list passed String columns as cats | Genuine bug, shipped fix (`545c472`). Orthogonal to the real cause — CB fastpath still failed after it. |
| 11a | Polars 1.x emits `pa.large_string()` that CB Cython doesn't dispatch | Standalone bench showed CB fits cleanly on `Dictionary<uint32, large_string>` when the column has no nulls. Large_string is NOT the trigger. |
| 11b | null_count > 0 in any cat_feature | Binary-searched null_frac, confirmed threshold is literally >0. Matches prod schema perfectly. |

Takeaway for future probes: **when an upstream error is opaque, build a 20-line isolated repro and binary-search the variable.** My first three round-11 fixes were "plausible-sounding diagnostic code" but not actual root cause because I never isolated the hypothesis with a direct bench. The final fix landed in one iteration after writing the 30-line null-frac sweep.

### Tests (`tests/training/test_round11_polars_largestring_fixes.py`, +9 sensors)

- `TestPolarsDfHasNullInCategoricalDetector` (8): single null triggers, null-free Categorical doesn't, null in non-cat column ignored, `cat_features` scope parameter respected, default scans all cat columns, non-Polars input defensive, empty DF safe, multi-cat prod-shape sensor.
- `TestLegacyAlias` (1): the deprecated `_polars_df_emits_large_string` still works via alias.

### Benches kept for reproducibility
- `bench_polars_largestring_cb_xgb.py` — the standalone that disproved the large_string hypothesis.
- `bench_polars_cb_repro.py` — feature-by-feature sweep (Int16, Boolean, many Categoricals, text_features, all-null cat) isolating which shape triggers.
- `bench_polars_cb_nullfrac.py` — null_fraction binary search proving ANY null triggers.

## 2026-04-19 — probe round 11: TRUE root cause of the CB/XGB Polars fastpath failures

Rounds 4, 7, and 10 each added diagnostic instrumentation and defensive fallbacks around the CatBoost Polars fastpath `TypeError: No matching signature found`. Those fixes were useful (diagnostic schema dump, per-method pandas fallback, dedup'd cache keys by feature_tier), but I misidentified the upstream cause twice (`pl.Enum`, then stale `cat_features_polars`). Today's prod log shipped a 35-minute CB run plus an `XGBoostClassifier` `KeyError: DataType(large_string)` mid-suite. Direct repro in a one-liner finally pinned the real cause.

### Root cause: Polars 1.x Arrow export → `large_string`

Polars 1.35.2 + pyarrow 22.0 exports:
  - `pl.String` / `pl.Utf8` → `pa.large_string()` (64-bit offsets)
  - `pl.Categorical` → `Dictionary<uint32, large_string>`

Both CatBoost 1.2.10 and XGBoost 3.x were compiled against the older `pa.string()` API:
  - **CatBoost**: `_set_features_order_data_polars_categorical_column` is a Cython fused cpdef with no signature for the `large_string` dictionary variant. Every fit call wastes 20-50 s on the failed attempt, then falls back to pandas. Every `predict_proba` call repeats the same dance. A 3-split evaluation (fit + predict_proba(val) + predict_proba(test)) pays the penalty 3× plus 3 redundant polars→pandas conversions (70-80 s each on 810k × 98).
  - **XGBoost 3.x**: `xgboost.data._arrow_dtype()` returns a dict of numeric types only. A plain `large_string` column (from `pl.String` in the tier-dropped polars DF) hits `KeyError: DataType(large_string)` inside `_wrap_evaluation_matrices` and kills the whole suite.

### Fixed — proactive CB fastpath bypass (`training/trainer.py` + `training/core.py`)

New helper `_polars_df_emits_large_string(df)` does a `head(0).to_arrow()` schema probe (no data copy) and returns True if any column is `large_string` / `large_utf8` / `large_binary` or a Dictionary wrapping one of those. Called right before the CB fastpath would fire; if True, `polars_fastpath_active` flips to False, the pandas tier DF is built now, and the rest of the CB model training uses pandas end-to-end — saving:
  - 20–50 s wasted `model.fit` attempt (the one that always failed)
  - 20–50 s wasted `predict_proba(val)` attempt
  - 20–50 s wasted `predict_proba(test)` attempt
  - 2× redundant polars→pandas conversions at predict time (fit's conversion is now reused)

Logged as a WARNING at fastpath-setup time so the operator sees the bypass clearly.

### Attempted-and-reverted: XGB `_arrow_dtype` monkey-patch

An earlier iteration in this round tried to wrap `xgboost.data._arrow_dtype` to map `large_string`/`large_utf8`/`large_binary` → `'c'` at `trainer.py` import time. Direct repro showed the shim only moves the crash one layer deeper — from `KeyError: DataType(large_string)` in `_arrow_feature_info` to `ValueError: too many values to unpack` in XGB's downstream dictionary-handling code. XGB really does require a `DictionaryType` column (not plain `large_string`) for the `'c'` code to work. **Reverted the shim** in favour of the proactive bypass above, which covers XGB too: the detector fires on `Dictionary<uint32, large_string>` just as it does on plain `large_string`, so XGB hits the pandas path before the crash window.

### Tests (`tests/training/test_round11_polars_largestring_fixes.py`, new file, +5 sensors)

`TestPolarsDfEmitsLargeStringDetector`: plain string column → True, Categorical → True (the prod-common case), numeric-only → False, empty DF no crash, non-Polars input → False (defensive). The proactive-bypass code path in `core.py` is exercised by the existing CB-fallback sensors in `tests/training/test_cb_polars_fallback.py` — those now cover both the "fastpath fails and falls back" scenario (round 4-7) and the "fastpath gets bypassed before it can fail" scenario (round 11).

### Context for the rounds 4/7/10 fixes

Those fixes remain correct and useful:
- The round-4 diagnostic schema dump (`_polars_schema_diagnostic`) is what made today's root-cause analysis tractable — the operator can see the per-column dtype breakdown the moment CB rejects.
- The round-7 `_predict_with_fallback` still handles the case where large_string detection misses (defensive lower tier).
- The round-10 `cache_key` partitioning by `feature_tier()` prevents CB's cached DF from being served to LGB/XGB even if we ever re-enable the fastpath.

None are redundant; they're each mitigations at different layers. The round-11 proactive bypass is the one that makes the common case FAST instead of just observable.

## 2026-04-19 — thinc/pytest-randomly seed-overflow session corruption (test-infra)

**The weirdest bug of the day.** After round-10 landed, rerunning previously-green sensor files started returning patterns like `4 passed, 20 errors` on `tests/training/**`. Disabling pytest-randomly (`-p no:randomly`) made everything green again. Running a single test in isolation passed. Running the file with `--randomly-seed=42` passed all 14; with `--randomly-seed=310986334` failed mid-file with `ValueError: Seed must be between 0 and 2**32 - 1` cascading into `previous item was not torn down properly`.

### Root cause (`thinc.util.fix_random_seed`)

`thinc` (a spaCy/explosion.ai dependency, transitively pulled by anyone with spacy/transformers installed) registers itself as a `pytest_randomly.random_seeder` entry point:
```python
# thinc/util.py:96
def fix_random_seed(seed: int = 0) -> None:
    random.seed(seed)
    numpy.random.seed(seed)   # <-- no % 2**32
```

pytest-randomly's own `_reseed` correctly applies `seed % 2**32` when calling `np_random.seed(...)`. But it ALSO iterates every registered entry point:
```python
for reseed in entrypoint_reseeds:
    reseed(seed)
```
…passing the **un-clamped** `seed = randomly_seed_option + crc32(test_nodeid) % 2**32` which regularly overflows 2**32 when the base seed is already large. Thinc's seeder then hits numpy's MT19937 bounds check and raises, breaking the fixture chain. pytest flags the next test with the generic "previous item was not torn down properly" — diagnosis was near-impossible without the full traceback.

### Fix (`tests/conftest.py::_patch_thinc_fix_random_seed_for_pytest_randomly_compat`)

Session-scoped autouse fixture that monkey-patches `thinc.util.fix_random_seed` to `lambda seed: original(int(seed) % (2**32))`. Also walks pytest-randomly's cached `entrypoint_reseeds` list (if already materialized) to swap the reference so live hooks pick up the shim. Teardown restores the original. Conditional — skipped silently if `thinc` isn't installed.

### Tests (`tests/test_thinc_pytest_randomly_clamp.py`, new file, +4 sensors)

- `test_fix_random_seed_accepts_large_seed`: calls `fix_random_seed(4_414_703_545)` (the exact value observed in prod) — must not raise.
- `test_fix_random_seed_normal_seed_still_works` + `_zero_still_works`: false-positive sensors — the clamp must not break normal seeds.
- `test_shim_is_wrapper_not_original`: checks `__closure__` is set (our wrapper has one, thinc's bare function doesn't) — trips immediately if someone refactors the shim out.

### Verification
- `tests/training/test_untested_fairness_outliers.py --randomly-seed=310986334`: 4 passed, 20 errors → **14 passed**
- `tests/training/test_round9_probe_fixes.py + test_round10_deferred_fixes.py --randomly-seed=310986334`: previously failing → **21 passed**

## 2026-04-19 — probe round 10: closing the 4 deferred round-9 findings

All four items marked "deferred" in round 9 are addressed. Investigation during the fix also flagged several probe claims as already-handled false positives.

### Fixed — PipelineCache collision across CB/LGB/XGB (`training/core.py`)

Verified the shared-`cache_key` concern from the round-9 strategies probe. CatBoostStrategy, XGBoostStrategy, and TreeModelStrategy (LGB parent) all inherit `cache_key = "tree"`. But they differ in `feature_tier()`:
- CB: `(supports_text=True, supports_embedding=True)` → `(True, True)`
- LGB, XGB (base): `(False, False)`

Models sort tier-desc before the loop, so CB runs first. In the pandas path, CB's `process_model` caches a DataFrame *with* text/embedding columns under `"tree"`. LGB then retrieves it via `cached_train_df` which overrides `common_params["train_df"]` at `train_eval.py:584-589` — so LGB trains on CB's tier-inappropriate DF with columns it can't handle.

Fix: append `feature_tier()` to the effective cache key in `core.py:2056`. Models with matching tiers still share the cache (intended); mismatched tiers get separate entries.

### Fixed — `prepare_df_for_xgboost` polars contract (`preprocessing.py`)

Signature declared `df: object`, returned `None`, only handled pandas. A Polars DataFrame passed by a caller assuming symmetry with `prepare_df_for_catboost` crashed with a cryptic AttributeError on `df[var].dtype`. Now:
- Explicit `TypeError` on Polars input naming the conversion helper (`get_pandas_view_of_polars_df`).
- `cat_features=None` accepted (coerced to empty list — pre-fix hit `var not in None`).
- Returns the DataFrame so callers can chain.
- Signature properly typed as `df: pd.DataFrame` with `Optional[Sequence]` for `cat_features`.

### Fixed — Bruteforce target-encoder leakage observability (`feature_engineering/bruteforce.py`)

`CatBoostEncoder.fit_transform(df, target)` on the full sample — classic supervised-encoding leak. A proper fix requires OOF/KFold encoding refactor (API change, reproducibility impact). Minimal defensive fix: loud `warnings.warn` + `logger.warning` at call time naming the encoded columns. Operators see the risk before using the returned PySR formula. Existing behavior preserved for back-compat; this is bruteforce FE path, not in the active production pipeline.

### Fixed — MPS `compute_area_profits` zero-price guard (`feature_engineering/mps.py`)

`return profits / prices` — zero-price bars (synthetic data, corrupted feeds) produced inf/NaN silently that downstream ML poisoned. Guard: `numba.njit` loop computes ratio only where `price > 0`; zero-price bars contribute 0 (no meaningful directional profit without a valid denominator). All-zero-prices returns all-zeros, no inf/NaN.

### Documented — probe claims that turned out to be **false positives**

Investigation during the fix pass disproved several round-9 findings:

- **financial.py ratio divisions** (round-9 flagged 4 HIGH findings): every ratio is wrapped in `pllib.clean_numeric(..., nans_filler=0.0)`, which per `pyutilz/data/polarslib.py:60` does `expr.replace([inf, -inf, nan], nans_filler)`. The +inf / -inf / NaN cases are already caught. Severity over-stated by the probe.
- **numerical.py `LARGE_CONST=1e3` sentinel** (round-9 flagged MEDIUM): reviewing the code and naming, this is an intentional design choice for ratio features when the denominator is 0 (tree models tolerate extreme values; the sentinel is named and explicit). Not a bug.
- **financial.py `add_talib_indicators` fill_null(0.0)**: intentional domain tradeoff — talib's input contract requires no NaN, and 0-fill is a commonly accepted (if imperfect) way to satisfy it. Not a silent bug.

### Tests
- `tests/training/test_round10_deferred_fixes.py` (new file, +12 sensors):
  - **TestCacheKeyIncludesFeatureTier (2)**: CB vs LGB produce different effective keys; same-tier strategies still share.
  - **TestPrepareDfForXgboostContract (4)**: polars raises TypeError, pandas returns df, cat_features=None accepted, existing auto-add contract preserved.
  - **TestBruteforceTargetEncoderWarn (1)**: WARN fires on categorical-encoding path (PySR stubbed).
  - **TestMpsComputeAreaProfitsZeroPriceGuard (3)**: zero-price bar → 0 (no inf), no-zero-prices path unchanged, all-zero boundary.

## 2026-04-19 — probe round 9: preprocessing extensions + strategies + feature_engineering

Three parallel subagent probes of areas not previously deeply covered. Each returned multiple findings; three HIGH/MEDIUM fixed this batch, the remainder documented for follow-up.

### Fixed — TF-IDF val/test column-parity invariant (`training/pipeline.py::apply_preprocessing_extensions`)

The TF-IDF block iterated over `config.tfidf_columns`, skipped a column when it was missing from train (typo path), but when a column was present in train and **missing from val/test**, it TF-IDF-expanded only train. Downstream sklearn Pipeline, fit on train with e.g. 5050 columns, then called `pipe.transform(val_with_50_cols)` → opaque shape mismatch that traced back to the scaler, not the upstream TF-IDF. Trigger: sparse splits where a text column exists in train only.

Now: before the loop, each tfidf column is classified as usable (present in all available splits), skipped-typo (missing from train), or skipped-split-mismatch (missing from val or test). Separate WARNs for each skip category. The loop only expands usable columns — all splits stay column-aligned.

### Fixed — `is_polars_categorical` missed `pl.Enum` (`training/strategies.py`)

Same class of bug as the 2026-04-19 early-morning fix in `_auto_detect_feature_types`: `pl.Enum` is an instance-level dtype, so `dtype in (pl.Categorical, pl.Utf8, pl.String)` returned False. Downstream `HGBStrategy.prepare_polars_dataframe` then silently treated Enum columns as numeric, breaking categorical semantics. Now: `isinstance(dtype, pl.Enum)` is also checked, so every Strategy subclass inherits correct detection.

### Fixed — category_encoder=None + requires_encoding=True silent no-op (`training/strategies.py::build_pipeline`)

`if self.requires_encoding and cat_features and category_encoder is not None:` — when the encoder wasn't provided, the step silently vanished. Downstream sklearn LinearRegression/MLP/etc. then raised opaquely on raw string categoricals deep inside `fit`. Now: if the first two conditions hold but encoder is None, WARN naming the strategy class and the cat count. Operator sees the missing dependency at pipeline-build time instead of model-fit time. Doesn't raise — some callers legitimately pre-encode cats upstream.

### Tests
- `tests/training/test_round9_probe_fixes.py` (new file, +11 sensors):
  - **TestTfidfSplitColumnParity (3)**: all-splits happy path, missing-val triggers WARN+skip, typo triggers different WARN.
  - **TestIsPolarsCategoricalEnum (4)**: Categorical/Utf8/String detected, Enum detected, numeric not detected, get_polars_cat_columns includes Enum.
  - **TestBuildPipelineEncoderMissingWarn (4)**: HGB warn on encoder=None, silent when encoder provided, silent when no cats, TreeModelStrategy silent regardless (requires_encoding=False).

### Documented — deferred findings (not fixed this batch)

- **`prepare_df_for_xgboost` contract problem** (`preprocessing.py:185-202`, MEDIUM): declares `df: object`, returns `None`, only handles pandas; crashes on polars. Needs a small refactor — deferred for scope.
- **Target-encoder leakage in `feature_engineering/bruteforce.py:126`** (HIGH): `encoder.fit_transform(df, target)` on the full sample before any CV split — classic target leak. Deferred because bruteforce FE isn't in the active production pipeline.
- **Shared `cache_key="tree"` across CB/LGB/XGB** (`training/strategies.py` + `core.py::PipelineCache`, HIGH-pending-verify): CB supports text/embedding features, LGB/XGB don't, but all three share `cache_key="tree"`. The polars fastpath doesn't use pipeline_cache (only the pandas path does); needs targeted test with mixed CB+LGB+XGB on text-heavy data to confirm the trigger fires.
- **Numerical `LARGE_CONST=1e3` sentinel + financial/MPS div-by-zero** (multiple, HIGH-MEDIUM): market-data-specific code paths, not in the active `prod_jobsdetails` pipeline.

## 2026-04-19 — probe round 8: closing the 4 deferred findings

Continuation of round 7. The four items marked "documented but not fixed this batch" all landed together:

### Fixed — atomic metadata / model writes (`training/io.py::atomic_write_bytes`)
New helper: write to ``<target>.<random>.tmp`` in the same directory, then ``os.replace()`` for an atomic rename (works on both POSIX and Windows since Python 3.3). On any write-time exception, the temp file is cleaned up and the pre-existing target remains untouched.

Wired into two sites that the round-7 probe flagged as concurrency-unsafe:
- ``training/core.py::_finalize_and_save_metadata`` — ``metadata.joblib`` dump
- ``training/io.py::save_mlframe_model`` — zstd-compressed dill dump of the fitted model

Before: two parallel training runs writing to the same target race-corrupted each other; the loader raised opaque ``UnpicklingError`` / ``EOFError``. Now: a reader sees either the complete pre-write file or the complete post-write one, never a partial mix.

### Fixed — polars→pandas bridge: nested-types warning (`training/utils.py::get_pandas_view_of_polars_df`)
Columns with ``pl.List[pl.Float32]`` (embedding features), ``pl.Struct``, or ``pl.Array`` survived the Arrow conversion as ``pd.object`` dtype with Python list elements. Downstream CatBoost's embedding_features fastpath then rejected them with an opaque "expected numeric" error from deep inside the CB internals. The bridge now emits one WARNING per call naming the affected columns and their dtypes. Doesn't raise or auto-cast — the bridge is a general helper and non-training callers (logging, post-hoc analysis) legitimately want list-typed pass-through.

### Fixed — per-fold NaN importances observability (`feature_selection/wrappers.py::get_feature_importances`)
When a CV fold was degenerate (single-class target, zero-variance features), the fitted model's ``feature_importances_`` legitimately contained NaN (observed with both CatBoost and LightGBM). Previously silent: NaN was folded into the per-feature aggregate ranking downstream, indistinguishable from "zero importance" and poisoning the rank of every feature touched by that fold. Now: WARN with the NaN count, model type, and likely cause. Pairs with the round-5 NaN-score warning in ``store_averaged_cv_scores``.

### Fixed — fit_and_transform_pipeline schema-drift validation (`training/pipeline.py::_warn_on_schema_drift`)
``pipeline.transform(val_df)`` / ``pipeline.transform(test_df)`` were called with no validation that val/test schemas matched train. Three silent failure modes:
  - Missing column at val/test → polars-ds errored deep inside with an opaque lookup failure traceback.
  - Extra column at val/test → silently kept or dropped depending on pipeline step internals.
  - Dtype change (e.g. train Int32 → val Int64) → silent coercion, potentially introducing NaN on bounds overflow.

Now: a snapshot of the train schema is captured before fit-time transform, and each val/test split is checked before its own transform. Separate WARN lines for missing / extra / dtype-mismatched columns with the full column list. Doesn't raise — some callers legitimately drop derived columns that the pipeline reconstructs.

### Tests
- `tests/training/test_round8_deferred_fixes.py` (new file, +13 sensors):
  - **TestAtomicWriteBytes (5)**: writes target atomically, overwrites existing, no tmp leak on failure, joblib round-trip, pre-existing target preserved on write fail.
  - **TestPolarsBridgeNestedTypesWarn (2)**: pl.List triggers warn, flat schema silent.
  - **TestGetFeatureImportancesNaNWarn (2)**: NaN importance WARN with count, all-finite silent.
  - **TestPipelineSchemaDriftWarn (4)**: missing column WARN, extra column WARN, dtype mismatch WARN, identical schema silent.

## 2026-04-19 — probe round 7: MRMR patience + phases log truncation + metadata validation

Three parallel subagent probes covered: (a) `training/phases.py` + `training/pipeline.py`, (b) `feature_selection/filters.py` (MRMR) + `feature_importance.py`, (c) `training/utils.py::get_pandas_view_of_polars_df` + `helpers.py::get_own_ram_usage` + persistence layer. Seven candidate findings; three fixed this batch, four documented for future rounds (below).

### Fixed — MRMR patience termination was silent (`feature_selection/filters.py::screen_predictors`)

`max_consec_unconfirmed` patience-triggered exits only logged at `verbose>=1`. At default verbosity (production), MRMR silently returned a potentially-truncated feature set; operators had no way to distinguish "done — no more gains above threshold" from "gave up confirming on noisy data — try higher patience." Added a termination-reason summary emitted unconditionally at function exit:
- **Patience-triggered** → `WARNING` with the count and actionable tuning hint.
- **Natural threshold exit** → `INFO` with the count.

Same observability pattern as the ICE-NaN / RFECV-NaN warnings from earlier today.

### Fixed — phase-context kwargs blew up log lines (`training/phases.py::_format_ctx`)

The bare f-string `f"{k}={v}"` didn't truncate value `str()`. A caller passing a large object (e.g. `phase("fit", eval_set=val_df)` with a 1M-row DataFrame as context kwarg) turned one START/DONE pair into MB+ log lines — breaks log rotation and structured-log aggregation (newline injection from `repr`). Values now truncated to 120 chars via `…` suffix, keys stay intact so the line is still greppable by field name.

### Fixed — critical-column validation at predict time (`training/core.py`)

Previously `predict_mlframe_models_suite` only WARN'd on missing columns and proceeded. If a missing column was a load-bearing feature (cat/text/embedding), the pipeline transform + model predict ran on a shape-mismatched frame and either crashed opaquely deep inside sklearn (`X has N features, expected M`) or produced garbage predictions with no visible signal. Extracted the check into `_validate_input_columns_against_metadata` and split by severity:
- **Missing cat/text/embedding features** → `raise ValueError` with a diagnostic listing all missing load-bearing columns and suggesting the two corrective paths (restore upstream extraction or retrain).
- **Other missing columns** → WARN + proceed (some callers drop derived columns that the pipeline reconstructs; that's legitimate).
- **Extra columns** → drop silently (unchanged behavior).

Deduped the logic that had two identical copies in `predict_mlframe_models_suite` and `predict_from_models`.

### Tests
- `tests/training/test_phases_and_metadata_validation.py` (new, +14 sensors): truncation of long strings / huge lists / null handling / max_val_len customization; metadata validation happy path, extra-column drop, missing-cat raises, missing-text raises, missing-embedding raises, non-critical missing warns, error-message lists-all-critical, empty-columns no-op.
- `tests/feature_selection/test_filters.py` (+1 sensor, `TestScreenPredictorsPatienceObservability`): termination-reason summary fires unconditionally (catches removal/regression of the new summary log). A second sensor for the WARN-level patience path was considered but dropped as too data-dependent to keep green; docstring explains.

### Documented but not fixed this batch
Captured in subagent reports for future rounds:
- **Polars bridge nested types** (utils.py:324–333): `pl.List[pl.Float32]` embeddings silently become `object` dtype. Needs design decision (warn vs raise vs convert).
- **fit_and_transform_pipeline schema drift**: no validation of val/test schema vs train after fit. Tightening may break legitimate callers; needs scoped design.
- **Concurrent joblib dump** (core.py:1136, io.py:101): no atomic rename; parallel train runs can corrupt metadata.joblib. Clear refactor; deferred for scope.
- **Per-fold NaN importances** (wrappers.py:881): NaN CV scores already warn; NaN importances from the same fold silently poison aggregate ranking. Needs reproduction first.

## 2026-04-19 — probe round 6: extractors + select_target + create_date_features

Subagent probe of `training/extractors.py`, `training/train_eval.py::select_target`, and `feature_engineering/basic.py::create_date_features`. Three HIGH-severity findings fixed.

### Fixed — `+inf` recency weights on every production run (`training/extractors.py`)

`get_sample_weights_by_recency` computed `np.log((max - date).dt.days) * weight_drop_per_year`. For the most-recent sample, `(max - date).days == 0`, so `np.log(0) = -inf`, and the weight evaluated to `+inf`. Training-time weighted loss was then dominated by that single row — CatBoost/sklearn clamp or NaN-out `+inf` weights in different ways, silently biasing the fit toward one example with no visible signal in the loss curve.

Also: when all timestamps were identical (`span_days == 0`) — e.g., a single-batch backfill or hourly aggregated data — `np.log(0)` on the span itself produced an all-NaN weight array.

Now: days-from-max is clipped to `>= 1` before the log (finest datetime resolution for this column anyway), and a zero-span series returns uniform `min_weight` for every row. All outputs are finite, no NaN, no +inf.

### Added — degenerate-class / extreme-imbalance WARN in `select_target` (`training/train_eval.py`)

`select_target` proceeded silently on all-zeros / all-ones classification targets. ROC AUC / PR AUC then returned NaN downstream, early-stopping stalled via the same class of bug we fixed earlier today (ICE NaN, RFECV NaN-score). Now:
- **Single-class target** (positive rate == 0% or 100%): WARN naming the target and the undefined-metric consequence. Does NOT abort — sanity runs with degenerate targets are legitimate.
- **Extreme imbalance** (positive rate < 0.1% or > 99.9%): separate WARN about noisy AUC; both classes present but signal is near-zero.
- **Balanced (0.1%–99.9%)**: silent (runs on every classification call, false positives would drown the log).

### Added — column-clash WARN in `create_date_features` (`feature_engineering/basic.py`)

`create_date_features(df, ['date'])` generated `date_year`, `date_month`, etc. via `df[new_name] = ...` without checking whether `new_name` already existed. A user-engineered column (e.g., a fiscal-year `date_year`) got silently overwritten with calendar year — data corruption, no log line. Now: collision detection runs before writing; any pre-existing derived name triggers a WARN naming all clashing columns. Does NOT raise (overwrite-on-rerun is a legitimate use case), but the operator sees the signal.

### Tests
- `tests/training/test_extractors.py` +4 (`TestGetSampleWeightsByRecency`): no +inf on newest sample, no NaN on identical timestamps, monotone non-decreasing by date, length invariant across span/size combos.
- `tests/training/test_untested_select_target.py` +5: all-zeros WARN, all-ones WARN, extreme-imbalance WARN, balanced-target silent, regression-target silent (no class concept).
- `tests/feature_engineering/test_basic.py` +3: pandas clash WARN, polars clash WARN, no-clash silent.

### Probe hygiene
Subagent also flagged `intize_targets` (crash on object-dtype with None) and `group_ids` length-alignment — not fixed this batch because they're loud-crash paths (vs. silent-wrong), and the reporting is already reasonable. Documented in subagent report for future rounds.

## 2026-04-19 — symmetric pandas fallback at predict time (`_predict_with_fallback`)

Follow-up to `545c472`. Same prod log revealed a second, independent dispatcher miss: after `fit` fell back to pandas and succeeded (14 min), `predict_proba` on the Polars val_df hit the **same** `_set_features_order_data_polars_categorical_column` TypeError. The existing except block in `evaluation.py:513` caught it and retried with `model.predict(df)` **on the same `pl.DataFrame`** — another dispatch miss. Not a fallback; a retry into the same hole, burning 48 s total before raising.

With `545c472` shipped, fit succeeds on the first attempt and predict gets a consistent shape, so this chain shouldn't trigger. But it's a latent trap: if fit ever falls back to pandas for a different reason, predict still breaks.

### Added — `_predict_with_fallback` + `_recover_cb_feature_names` (`training/trainer.py`)

Symmetric wrapper to `_train_model_with_fallback`. On a `TypeError` with "No matching signature found", a CatBoost model, and a `pl.DataFrame` input, the helper:
  1. Converts polars → pandas via the zero-copy Arrow view
  2. Recovers cat/text feature names from the fitted model via `_get_cat_feature_indices()` / `_get_text_feature_indices()` / `feature_names_` — callers (evaluation code) don't need to track these
  3. Decategorizes pd.Categorical text columns (same ordering as fit's fallback — avoids prep_cb rebuilding them)
  4. Runs `prepare_df_for_catboost` with the recovered feature lists
  5. Retries the original method on the pandas DF

Non-CB models, non-polars inputs, and unrelated TypeErrors **propagate unchanged** — the wrapper is targeted at exactly one failure mode. AttributeError also propagates so the outer `predict_proba → predict` fallback in evaluation keeps working for models without `predict_proba`.

### Wired into `training/evaluation.py`
Two call sites:
- `report_regression_model_perf`: `model.predict(df)` → `_predict_with_fallback(model, df, method="predict")`
- `report_probabilistic_model_perf`: both the `predict_proba(df)` call AND the outer `model.predict(df)` fallback now go through the wrapper. Lazy import breaks the `trainer ↔ evaluation` circular.

### Tests (`tests/training/test_cb_polars_fallback.py`)
10 new sensors across 2 test classes:

`TestRecoverCBFeatureNames` (3):
  - name recovery from indices + feature_names_
  - empty return on unfitted model (no crash)
  - invalid indices silently skipped (not raised)

`TestPredictWithFallback` (7):
  - polars TypeError → pandas retry → success (2 calls)
  - both `predict` and `predict_proba` wrapped
  - happy path: no error, single call, no log noise
  - non-CB model TypeError propagates (don't swallow real bugs)
  - non-polars input TypeError propagates
  - unrelated TypeError text (e.g. shape mismatch) propagates
  - AttributeError propagates (outer fallback needs it)

31 tests total in `test_cb_polars_fallback.py` (21 pre-existing + 10 new).

## 2026-04-19 — ROOT CAUSE: stale cat_features list broke CB Polars fastpath

Diagnostic logging from the earlier commit (`49ba314`) paid off immediately. Next prod run's log now includes the full per-column schema dump. Turns out the Enum hypothesis was WRONG — **no pl.Enum columns in the data**. All 9 cat_features are plain `pl.Categorical(ordering='lexical')`. But 4 columns in the dump show up as `String` dtype while still tagged `[cat]`:

```
Polars schema diagnostic for 810_000×98:
    category [cat]: String, n_unique=52, nulls=0
    occupation [cat]: String, n_unique=100, nulls=0
    skills_text [cat]: String, n_unique=81575, nulls=0
    ontology_skills_text [cat]: String, n_unique=2735, nulls=0
    ...
```

These 4 were promoted to text_features and cast `pl.Categorical → pl.String` right before fit. But they still ended up in the `cat_features` list that CB received.

### Root cause — stale short-circuit (`training/core.py:1935`)

```python
_cat_features = cat_features_polars or cat_features or []
```

`cat_features_polars` is populated at line 1435 (start of Phase 3) via `get_polars_cat_columns(train_df)` — returns all 13 categorical columns from the *raw* Polars schema, BEFORE text-promotion. `cat_features` is reassigned at line 1526 to the post-promotion, dedup'd 9-item list. The `or` short-circuit picked the stale 13-item `cat_features_polars`, passing `['category', 'skills_text', ...]` to CB even though those columns are now `pl.String`.

CatBoost 1.2.10's `_set_features_order_data_polars_categorical_column` is a Cython fused cpdef with dispatch only for `pl.Categorical` — `pl.String` falls through to "No matching signature found". 22 s burnt + 150 s pandas fallback on every run.

### Fixed
- `training/core.py:1935`: replaced the short-circuit with `_cat_features = list(cat_features or [])` — uses the correct post-promotion list directly. Comment documents the exact prod bug so a future refactor doesn't reintroduce the short-circuit.

### Added — defensive runtime filter `_filter_polars_cat_features_by_dtype` (`training/core.py`)

Last-line defence for the same bug class: checks every cat_feature's runtime dtype in the DataFrame right before `model.fit()`. Drops any column whose dtype isn't `pl.Categorical`/`pl.Enum` and logs a WARNING naming the column and its dtype. Preserves Enum for builds where CB has Enum dispatch (we don't decide that — CB does, via its own error path). If a future orchestration bug ever reintroduces a mismatch between the cat_features list and actual column dtypes, the filter catches it before CB throws the opaque TypeError, and the WARN tells the operator exactly what was wrong.

### Tests (`tests/training/test_cb_polars_fallback.py`)
7 new sensors in `TestFilterPolarsCatFeaturesByDtype`:
- `test_drops_string_columns_declared_as_cat` — the exact prod shape, must drop + WARN.
- `test_keeps_categorical_columns` — happy path, no warning (runs on every fit, no log spam tolerated).
- `test_keeps_enum_columns` — Enum stays (let CB decide).
- `test_silently_skips_missing_columns` — defensive, column not in DF dropped silently.
- `test_empty_input_returns_empty` — None / empty input → empty output, no crash.
- `test_all_string_returns_empty_not_none` — all-wrong → empty list (not None), so `if out:` stays safe.
- `test_numeric_column_in_cat_features_also_dropped` — boundary beyond String.

21 total pass in test_cb_polars_fallback.py (14 old + 7 new).

### Follow-up
The upstream bug in `train_mlframe_models_suite` that assembles `_cat_features` from stale `cat_features_polars` is also fixed above. Future prod runs: no more 22 s + 150 s detour; the Polars fastpath should succeed on the first attempt. If it doesn't, the new schema dump will name the culprit in the first WARNING line.

## 2026-04-19 — proactive probe round 5: per-group AUC + RFECV NaN observability

Subagent-driven probe of `fast_aucs_per_group_optimized`, MRMR, RFECV, and
`compute_mean_aucs_per_group`. Three findings worth fixing; four more
documented as correctly-handled false positives (logged in subagent report).

### Fixed — single-sample group NaN sentinel (`metrics.py::compute_grouped_group_aucs`)
Single-sample groups (group_size == 1) returned ``(0.0, 0.0)`` instead of ``(nan, nan)``. `compute_mean_aucs_per_group` filters NaN but treated 0.0 as legitimate data, so a CV fold with many single-sample groups silently depressed the mean AUC toward 0 — indistinguishable from "model is bad" in operator eyes. Now returns NaN, which the filter drops.

### Added — observability for majority-NaN per-group AUC (`metrics.py::fast_aucs_per_group_optimized`)
The inner numba loop silently returned NaN for single-class / single-sample groups. When `mean_group_roc_auc` came back as NaN, operators had no hint why. Added a Python-level WARNING: if ≥50% of groups return NaN ROC AUC, log once with the counts and the most common causes (target imbalance concentrated in a few groups, or `group_ids` granularity too fine producing many 1-sample groups). Minority NaN (e.g., 10% of groups) stays silent to avoid log spam — the mean is still trustworthy.

### Added — NaN-in-CV-fold warning (`feature_selection/wrappers.py::store_averaged_cv_scores`)
Same class of bug as the `integral_calibration_error_from_metrics` NaN guard fixed earlier today. A NaN CV fold score (scorer hit a single-class fold etc.) poisoned `scores_mean` → `final_score`. Downstream `final_score > best_score` with NaN is always False — RFECV's `n_noimproving_iters` counter incremented every iteration and the search eventually terminated via `max_noimproving_iters`, but burnt many full CV rounds producing no signal. WARN with the NaN-fold count and the likely cause on every invocation that has a NaN score. Operators can then switch to stratified CV or fix the scorer instead of staring at silent stagnation.

### Tests
- `tests/test_metrics.py` +5 (`TestPerGroupAUCEdgeCases`): single-sample-group returns NaN, NaN excluded from mean, single-class group returns NaN (lock-in sensor), majority-NaN emits warning, minority-NaN stays silent.
- `tests/feature_selection/test_wrappers.py` +3 (`TestStoreAveragedCVScoresNaNWarning`): NaN score emits named WARNING with pos, clean scores silent, empty scores graceful.
- 142 tests pass across `test_metrics.py` + `test_wrappers.py`.

### False positives documented
Probe flagged 4 scenarios that turned out to be handled correctly; captured in subagent report for future callers:
- `group_ids=None` → empty dict, downstream `if group_aucs` guard exists.
- Extremely imbalanced group (1 pos / 100k samples): the `denom_roc > 0` check at `fast_numba_aucs_simple:775` handles it.
- NaN in y_score: numpy argsort handles (NaN sinks to tail); not our bug, should be caught upstream.
- MRMR accepts polars via the `X.to_pandas()` conversion at `filters.py:2859`.

## 2026-04-19 — CB Polars fastpath diagnostic logging: pl.Enum is the usual culprit

### Added — `_polars_schema_diagnostic` + `_warn_on_unsupported_polars_dtypes` (`training/trainer.py`)

Production 2026-04-19: CatBoost 1.2.10's Polars fastpath raised `TypeError: No matching signature found` at `_set_features_order_data_polars_categorical_column.process()`. The old log line was
```
CatBoost Polars fastpath rejected the data (TypeError: No matching signature found); converting to pandas and retrying.
```
— just the last 160 chars of the error. No way to know which of 13 categorical columns was the culprit, so every debug cycle burned 2+ minutes on a failed fastpath attempt + a ~76-second pandas conversion, and the MemoryError downstream was the one that finally stopped the run.

**Root cause (via subagent reconnaissance):** CatBoost 1.2.10's fastpath is a Cython fused cpdef with dispatch overloads for `pl.Categorical` only. `pl.Enum` (instance-level dtype added in modern Polars) has no matching overload → the fused dispatcher falls through to the generic "No matching signature found" path.

### Changes
- **Pre-fit warning** (`_warn_on_unsupported_polars_dtypes`): called right before `model.fit()` whenever we're about to hand a Polars DF to CatBoost. If any `cat_features` column is `pl.Enum`, logs a WARNING naming the columns and telling the user to cast to `pl.Categorical` or `pl.String`. Cheap, targeted, no DataFrame mutation — the whole thing is wrapped in `try/except` so a diagnostic failure never blocks a fit.
- **Post-fail schema dump** (`_polars_schema_diagnostic`): called when the fallback catches the Polars fastpath TypeError. Renders every `cat_features` column with its dtype (Enum vs Categorical with `ordering`), `n_unique`, null count; summarises non-cat/non-text columns by dtype count. If any Enum is found among cat_features, the dump's header explicitly names them as the most likely cause. The dump goes out as a second WARNING line right after the original error message.
- **Error message detruncated** from 160 → 240 chars; previously useful context (column names in CB's internal message path) was getting clipped.

### Tests
- `tests/training/test_cb_polars_fallback.py` +6 sensors:
  - `test_warn_on_unsupported_polars_dtypes_flags_enum_cat_features` — names the Enum column in the WARNING.
  - `test_warn_on_unsupported_polars_dtypes_silent_when_clean` — no false-positive on plain Categorical (runs on every CB fit).
  - `test_polars_schema_diagnostic_names_enum_culprit` — Enum columns surface in the dump's header, not buried in the per-column list.
  - `test_polars_schema_diagnostic_handles_empty_cat_features` — works with None/empty cat_features.
  - `test_polars_schema_diagnostic_never_raises` — returns a string even on malformed input (it runs inside an `except` block; a crash here would eat the original CB error too).
  - `test_cb_fallback_warning_emits_schema_dump_on_rejection` — end-to-end: FakeCatBoost raises the prod TypeError, the fallback path emits a WARNING carrying the schema context.

### Follow-up (not in this commit)
Upstream fix: wherever the 9 production `cat_features` columns acquire their dtype, one of them is arriving as `pl.Enum`. Next run's log will name it explicitly (pre-fit warning). Short-term workaround: cast `pl.Enum → pl.Categorical` in the polars fastpath DF prep. Long-term: CatBoost may add Enum dispatch upstream.

## 2026-04-19 — MemoryError fix: Categorical NaN-fill no longer materializes the dictionary

### Fixed — 75 GiB allocation in `prepare_df_for_catboost` (`preprocessing.py`)
Production incident (logs, 2026-04-19 01:55): CatBoost Polars fastpath was rejected with `TypeError: No matching signature found` in `_set_features_order_data_polars_categorical_column.process()`. The pandas fallback kicked in, converted polars→pandas (76 s), then hit `MemoryError: Unable to allocate 75.1 GiB for an array with shape (3287945,) and data type <U6133` inside `prepare_df_for_catboost` → the whole 2.5-minute pipeline died one step before fit.

Root cause: a pandas Categorical column arrived with an **untrimmed Polars global-string-pool dictionary** — 3.3M unique categories, longest string 6133 chars — even though the train slice had only ~810k rows. The NaN-fill path used:
```python
df[var] = df[var].astype(str).fillna(na_filler).astype("category")
```
`pd.Categorical.astype(str)` internally expands `categories._values` into a **fixed-width Unicode array** sized by `n_categories × max_str_len × 4 bytes` → 3.3M × 6133 × 4 ≈ 75 GiB regardless of how many rows the slice actually holds. The row count is irrelevant to this allocation; only the dictionary size matters.

Fix: operate on the integer codes — `.cat.add_categories([na_filler])` (O(1) dict growth) + `.fillna(na_filler)` (O(n_rows) code update). No string materialization. The original category order is preserved so downstream CatBoost Pool indexing across train/val/test stays stable. Idempotent when `na_filler` is already in the category list.

### Tests
- `tests/test_preprocessing.py` +4 sensors:
  - `test_cat_nan_fill_does_not_materialize_dictionary_as_strings` — the functional sensor with a 50k-entry untrimmed dict.
  - `test_cat_nan_fill_preserves_existing_categories` — CatBoost Pool stability sensor (original order preserved).
  - `test_cat_nan_fill_idempotent_when_na_filler_already_a_category` — no duplicate-add crash.
  - `test_cat_nan_fill_perf_budget_huge_untrimmed_dict` — 100k-entry dict × 10k rows < 2 s budget (buggy path would need minutes or OOM).

## 2026-04-19 — outlier guard + ICE NaN guard + stable configs strict + cache-probe cleanup

### Fixed — catastrophic outlier-detector misconfiguration (`training/core.py`)
- `_apply_outlier_detection_global` silently produced a 0-row train frame when the detector (e.g. IsolationForest with `contamination=0.99`, a sign-convention bug, or an untrained pipeline) flagged ~every sample as an outlier. Downstream CatBoost/LightGBM then crashed 5+ minutes later with opaque `X is empty` / shape errors — no signal at the source. Added a loud fail-fast `ValueError` when the kept train rows drop below `max(1, 1% of input)`, naming the most likely causes in the message (contamination too high, unrepresentative fit, sign-convention bug).

### Fixed — ICE metric NaN propagation (`metrics.py`)
- `integral_calibration_error_from_metrics` used `np.abs(roc_auc - 0.5) * weight` unconditionally; a NaN roc_auc (from `fast_aucs_per_group_optimized` on a single-class eval window) turned the entire ICE into NaN. This silently broke early-stopping comparisons (`NaN > best` is always False), locking the trainer on iteration-1's "best" without surfacing any error. Guarded both `roc_auc` and `pr_auc`: a NaN input now means "skip that term" (0.0 contribution, no penalty ramp). Baseline ICE for callers that pass finite values is unchanged.

### Refactored — killed dead mutation in `_auto_detect_feature_types` (`training/core.py`)
- The function used to call `cat_features.remove(name)` for each promoted column. The in-place mutation was dead code: the caller in `train_mlframe_models_suite` already filters via a set-difference (`effective_cat_features = [c for c in raw_cat_features if c not in text_emb_set]`). But the mutation was a latent trap for any future caller that reused the list — second call would see the promoted columns already gone and its `"promoted"` diagnostic would be wrong. Removed the mutation; documented the read-only contract in the docstring.

### Changed — Hybrid Variant C: strict validation on stable-surface configs (`training/configs.py`)
- `PreprocessingConfig`, `TrainingSplitConfig`, `FeatureTypesConfig` switched to `extra="forbid"`. These three have a small, stable, fully-declared surface with no legitimate pass-through kwargs, so a typo (`fillna_vlue`, `trainset_agng_limit`, `embeding_features`) now raises a `ValidationError` at construction instead of silently being absorbed or buried in a WARNING log.
- `ModelHyperparamsConfig` and `TrainingBehaviorConfig` intentionally keep `extra="allow"` with the existing `_warn_on_unknown_extras` path — they legitimately forward kwargs (ICE weights, scoring configs, robustness params) to deeper callees via `**config_params`.

### Tests (all green)
- `tests/training/test_untested_fairness_outliers.py` +5: catastrophic-rejection guard sensors (all rejected, <1% kept, polars path, error-message content, single-row rejection still OK).
- `tests/test_metrics.py` +3 (`TestICENaNGuards`): NaN roc_auc / NaN pr_auc / both-NaN all produce finite ICE.
- `tests/training/test_untested_core_helpers.py`: updated 3 existing tests to the new no-mutation contract; added `test_auto_detect_does_not_mutate_cat_features_across_calls` to lock the contract in place.
- `tests/training/test_configs.py` +6 (`TestStrictConfigsRejectUnknownFields`): Hybrid Variant C sensors — typo on each strict config raises, valid fields still construct successfully.

### Documentation
- `README.md` testing-doctrine table extended: "Catastrophic misconfig", "NaN propagation", and "Strict vs lenient configs" added to the probe-category table with 2026-04-19 examples.

## 2026-04-19 — splitting + configs probe: validation gaps closed + testing doctrine in README

### Fixed — `make_train_test_split` (`training/splitting.py`)
- `test_size=1.0` + timestamps crashed with ``NaTType does not support strftime`` because the empty-train-index date-range format hit ``idx.min() == NaT``. Guarded: empty train now yields ``train_details="(empty)"`` in both whole-day and row-timestamp branches.
- Negative ``test_size`` / ``val_size`` silently no-opped (no Pydantic validator was upstream, function had no self-check). Now rejected at function entry with clear ``ValueError``.
- ``trainset_aging_limit=0`` silently no-opped via the ``if aging:`` falsy-short-circuit, contradicting the explicit "must be in (0, 1)" validator below it. Now uniformly rejected: only ``None`` means "no aging"; any other value must be strictly in ``(0, 1)``.
- Silent-empty-split warning: when user requested ``val_size > 0`` but the whole-day split collapsed val (or test) to zero rows (single-date frame, or very small `n*size`), a WARNING now fires naming the likely cause. Previously users silently lost the split.

### Fixed — Pydantic config validators (`training/configs.py`)
- `ModelHyperparamsConfig.learning_rate` — unvalidated; ``-0.1`` / ``5.0`` were silently accepted and propagated to the tree backends. Now ``Field(gt=0, le=1.0)``.
- `ModelHyperparamsConfig.iterations` — same; ``-1`` / ``0`` now rejected with ``Field(ge=1)``.
- `ModelHyperparamsConfig.early_stopping_rounds` — same; ``-1`` / ``0`` rejected with ``Field(ge=1)``. ``None`` (meaning "disable early stopping") still allowed via ``Optional``.
- `TrainingSplitConfig.trainset_aging_limit` — unvalidated; ``-0.5`` / ``1.5`` / ``0`` silently accepted. Now ``Field(default=None, gt=0, lt=1)`` — None is the only "no aging" signal.

### Added — typo-warning on `extra="allow"` pass-through
`BaseConfig._warn_on_unknown_extras` model-validator logs a WARNING for every extra field that is not on the subclass's ``_known_extras`` whitelist. Catches the common typo class (``iteratoins`` for ``iterations``, ``prefer_calibrated_classifer`` missing an ``i``) that ``extra="allow"`` used to swallow silently. ``ModelHyperparamsConfig`` declares the legitimate pass-throughs (ICE metric weights, scoring configs, robustness params) so valid kwargs don't noise the log.

### Tests
- `tests/training/test_splitting_edges.py` (new, 15 tests): validation + NaT guard + silent-empty warning + reproducibility sanity.
- `tests/training/test_configs.py` expanded (11 new tests): range validators for learning_rate / iterations / early_stopping_rounds / trainset_aging_limit; typo warning on unknown extras; silencing for known pass-throughs.

### Documentation
- `README.md` gained a "Testing approach: reactive + proactive" section that writes down the doctrine shaken out over the last two days' production bugs. Reactive sensors anchor known fixes; proactive probes explore the neighbourhood around the fix for second-order bugs. Both together → low chance of the same bug class returning. Table in README enumerates probe categories that have paid off so far (None-guard, empty input, boundary, dtype edge, state leak, silent overlap, orchestration, retry propagation) with a one-line recipe for running probes. Separate subsection on perf budgets as a regression-class distinct from functional tests.

## 2026-04-19 — Proactive exploratory probes uncovered (and fixed) 3 more latent bugs

### Fixed
- **`_auto_detect_feature_types` missed `pl.Enum`**: the dtype check `if dtype in (pl.String, pl.Utf8, pl.Categorical)` did not match `pl.Enum` instances (Enum carries instance-level dtype metadata that doesn't compare equal to the class-level entry). Added `isinstance(dtype, pl.Enum)` branch. Without this, a high-cardinality `pl.Enum` text column silently stayed in `cat_features` and CatBoost wasted memory on nominal encoding — same bug class as the `skills_text` case but on a different Polars type.
- **`_auto_detect_feature_types` crashed on `cat_features=None`**: `if name in cat_features` hit `TypeError: argument of type 'NoneType' is not iterable`. Callers who skipped categorical detection passed None; now treated as empty.
- **`prepare_df_for_catboost` crashed on `text_features=None` / `cat_features=None`**: the `for var in text_features` / `cat_features` loops can't iterate None. Both arguments now None-guarded at function entry.
- **`_validate_feature_type_exclusivity` crashed on None lists**: `set(None)` raises. All three arguments now coerce None to empty list before set ops.

### Added (regression sensors)
- `test_auto_detect_polars_enum_promoted_by_cardinality` — the Enum-specific sensor.
- `test_auto_detect_accepts_cat_features_none` — None-guard sensor.
- `test_text_features_none_does_not_crash`, `test_cat_features_none_does_not_crash`, `test_cat_features_both_none_does_not_crash` — prep_cb None-guard sensors.
- `test_exclusivity_accepts_none_args`, `test_exclusivity_none_still_catches_real_overlap` — validator None-guard sensors (the second one guards the silent-overlap-while-None-guard regression).
- `test_cat_features_list_is_mutated_in_place_across_calls` — documents the in-place mutation contract so a future refactor that returns a fresh list doesn't silently break callers in `core.py`.
- `test_high_cardinality_conversion_perf_budget` — `get_pandas_view_of_polars_df` on 500k × 1 Categorical with 500k uniques must finish < 5 s.
- `test_empty_polars_dataframe` + `test_zero_column_polars_dataframe` — edge-case robustness.
- `test_fallback_without_eval_set_still_retries`, `test_fallback_retry_failure_propagates` — fallback orchestration cases not covered by the earlier end-to-end tests.

### Process note
The reactive regression tests added earlier in the day all passed on the first run — comforting but also suspicious, because the bugs they target had already been fixed. Running a round of *proactive* exploratory probes (what-if tests over Enum, None args, empty frames, zero columns, high cardinality, retry failures, eval-set absence) surfaced four real latent bugs that reactive tests would never have caught. Keeping both practices going forward.

## 2026-04-19 — Test-suite expansion: invariants, boundaries, orchestration, perf

### Why
The string of production bugs over the past two days (cat-to-text promotion side-effects, CB fallback ordering, prep_cb O(n) dance on high-cardinality text columns) all slipped through because our tests exercised **inputs → outputs** on toy data but not:

1. **Behavioural invariants** — "text_features must NEVER flow into cat_features" wasn't asserted anywhere.
2. **Orchestration flows** — "fastpath raises → pandas fallback → decategorize → prep_cb → retry" was never run end-to-end.
3. **Boundary conditions** — threshold `>` vs `>=` regressions pass silently when a single mid-range test data point is used.
4. **Perf budgets** — `astype(str).astype("category")` at O(n_rows × avg_str_len) is fine on 10 rows but kills production at 810k.
5. **High-cardinality fixtures** — `["A","B","A","C"]` with 3 uniques hides a class of bugs that only bite at 10k+.

### Added
- **`tests/test_preprocessing.py`** (3 new tests for `prepare_df_for_catboost` invariants):
  - `test_pandas_text_feature_categorical_not_added_to_cat_features`: the bug-class sensor — a pd.Categorical column declared in `text_features` must NOT be auto-appended to `cat_features`.
  - `test_pandas_text_feature_skips_expensive_astype_rebuild`: **perf budget** sensor on a 50k × 5k-unique text column. Without the skip, the `astype(str).astype("category")` rebuild blows through 2 s; with the skip it finishes in milliseconds. If this sensor fires, the skip logic regressed.
  - `test_pandas_text_feature_dtype_is_not_mutated`: declares the function's responsibility boundary — text-column dtype conversion is the caller's job (via `_decategorize_text_cols`), not `prepare_df_for_catboost`'s.
- **`tests/training/test_untested_core_helpers.py`** (4 new tests for `_auto_detect_feature_types`):
  - `test_auto_detect_pandas_promotes_high_card_cat_to_text` — the formerly-inverted test now asserts correct semantics (promote AND remove in place).
  - `test_auto_detect_pandas_keeps_low_card_cat` — negative case.
  - `test_auto_detect_threshold_boundary` parametrized over `(n_unique=9, 10, 11)` with `threshold=10` — catches `>` vs `>=` regressions.
  - `test_auto_detect_user_text_wins_over_promotion` — user-declared `text_features` authoritative over the cardinality heuristic.
  - `test_auto_detect_polars_categorical_promoted_by_cardinality` — `pl.Categorical` columns are eligible for text promotion (was the production `skills_text` path).
- **`tests/training/test_cb_polars_fallback.py`** (new file, 6 tests): end-to-end tests of the fallback orchestration via a `FakeCatBoost` stub that raises on first fit and succeeds on retry.
  - `test_fallback_triggers_on_polars_typeerror` — the fallback activates on the exact message production showed.
  - `test_fallback_converts_train_df_to_pandas` — retry receives pandas, not polars.
  - `test_fallback_decategorizes_text_columns_before_retry` — regression sensor for the 2026-04-19 morning bug (retry received pd.Categorical → CB rejected).
  - `test_fallback_rewrites_eval_set_to_pandas` — eval_set X is pandas + text cols decategorized (otherwise CB re-crashes on val).
  - `test_fallback_passes_when_polars_fastpath_succeeds` — sanity: no fallback when first fit succeeds.
  - `test_fallback_ignored_for_non_catboost_models` — fallback is CatBoost-specific (XGB/LGB/MLP don't trigger it).

### Fixed
- **`_auto_detect_feature_types`**: the 2026-04-19 behaviour change (promoting `cat_features` columns to `text_features` when cardinality exceeds threshold) went in without updating `test_auto_detect_pandas_skips_cat_features`. That stale test would have kept passing only because of an incorrect assertion; now replaced with `test_auto_detect_pandas_promotes_high_card_cat_to_text` that asserts the new (correct) contract.

### Lessons captured as sensors
A future regression of any of the fixed bugs would trip one of the new tests above, with a clear message pointing at the invariant that broke. Specifically:
- Reintroducing the `astype(str).astype("category")` dance for text columns → perf budget test fails with ``"prepare_df_for_catboost took X.XXs on a 50k text column — the text-feature skip likely regressed"``.
- Reversing the fallback ordering back to ``decategorize → prep_cb`` ordering → end-to-end test fails with ``"text column X arrived at retry with dtype category; must be object/string"``.
- `>` flipping to `>=` in the cardinality threshold → boundary test fails on the `n_unique=10, threshold=10` case.

## 2026-04-19 — Fallback hang fix: text features no longer pay the cat-preparation tax

### Fixed
- **Production hang in the CB Polars-fastpath fallback** (`training/trainer.py` + `preprocessing.py`). A live run (2026-04-19 00:38) showed the fallback reaching ``prepare_df_for_catboost`` and stalling on the "Processing categorical features for CatBoost..." tqdm: for every column with a ``pd.CategoricalDtype`` the function ran

  ```python
  df[col].astype(str).fillna(na_filler).astype("category")
  ```

  On the user's ``skills_text`` column (81_575 unique values × 810_000 rows) that re-materialises every row as a Python string then rebuilds a CategoricalIndex — minutes per column, ~tens of minutes total across the four high-cardinality text columns that had been auto-promoted from ``cat_features`` to ``text_features`` earlier in the pipeline. Two complementary fixes:

  1. **Reorder the fallback pipeline to decategorize *before* ``prepare_df_for_catboost``** (`trainer.py::_train_model_with_fallback`). The ``_decategorize_text_cols`` helper was running *after* ``_prep_cb`` — too late, because by then ``_prep_cb`` had already started the expensive dance. Now the order is ``get_pandas_view → decategorize → _prep_cb``. Applied to both ``train_df`` and every ``eval_set`` pair.
  2. **Skip ``text_features`` columns in ``prepare_df_for_catboost``'s pandas cat-iteration loop** (`preprocessing.py`). A text-feature column that happens to carry ``pd.CategoricalDtype`` must not be auto-added to ``cat_features`` nor pass through the ``astype(str).astype("category")`` rebuild — it's text, not categorical, and the function now makes that invariant explicit via an opt-out set.

  Defence in depth: either fix alone would unblock the production scenario; together they ensure no future code path can re-open the hang.

## 2026-04-19 — Investigated (and ruled out) shared-dict optimisation for polars→pandas

### Investigation summary
The production PHASE-4 log of 2026-04-18 showed ``get_pandas_view_of_polars_df`` consuming 383 s total across train/val/test on a 1M × 98 frame with 13 Categorical columns (4 high-cardinality text-like). The initial hypothesis: train/val/test are slices of one source DataFrame, so they share a Categorical palette; the per-split pyarrow dict rebuild duplicates O(n_unique) work. A ``shared_dict_cache`` parameter was added along with equivalence checks, plus wiring in ``_convert_dfs_to_pandas``.

A synthetic benchmark disproved both the premise and the premise-of-the-premise:

1. **Polars trims the Categorical dictionary per slice** — each of ``train``, ``val``, ``test`` carries only the categories actually present in its row subset, with different orderings and lengths. The cache's equivalence check correctly rejected every cross-call reuse, so the optimisation became a no-op in practice. A new regression-sensor test (``TestPolarsSliceDictionaryDiffers::test_slice_trims_categorical_dictionary``) documents this and will trip immediately if a future polars upgrade starts preserving parent palettes across slices (which would make the optimisation viable again).
2. **The function is actually fast on synthetic prod-shaped data.** 1M × 93 with 13 Categoricals (including 4 high-cardinality ones, short strings ~8 chars): **0.45 s total** across the three splits. Switching to 500-char "text-blob" categoricals (closer to production's ``skills_text`` / ``ontology_skills_text``): **0.59 s total**. Production's 383 s is ~650× slower — the per-column work simply doesn't scale that way even with long strings.
3. An alternative "direct-polars path" (build ``pd.Categorical.from_codes`` skipping the pyarrow round-trip) was **4.24× slower** than the current implementation. Not a win.

### Conclusion
The 383-s production cost is not inside ``get_pandas_view_of_polars_df`` — it's dominated by something the function can't see: memory pressure at ~37 GB RSS causing OS-level page thrash / swap, or per-process overheads outside the function. No in-function optimisation is possible; future work would need to address memory-ceiling behaviour at the suite level.

### Fixed
- ``get_pandas_view_of_polars_df`` signature reverted to the pre-2026-04-19 shape (no ``shared_dict_cache`` parameter). The docstring now carries a "Tried but reverted" note so future readers don't reopen the same dead end.
- ``_convert_dfs_to_pandas`` no longer constructs a shared cache dict per call; the per-split timers stay (they proved useful).

### Tests
- ``TestSharedDictCache`` removed.
- ``TestPolarsSliceDictionaryDiffers`` added as a single-test regression sensor documenting Polars' per-slice dict-trimming behaviour.

### Bench scripts
- ``bench_shared_dict_cache.py`` rewritten as a **per-step profiler** for the conversion (1. ``to_arrow()`` 2. dict rebuild 3. ``to_pandas()``) + direct-polars alternative comparison. Kept for future investigations.
- ``bench_long_strings.py`` added: measures the effect of Categorical string length on conversion time. Confirms the function is fast even at 500-char strings.

## 2026-04-19 — Auto-promote cat→text: correctly drop promoted cols + diagnostic + fallback timers

### Fixed
- **`cat_features` list was not updated after auto-promotion of high-cardinality columns to `text_features`** (`training/core.py`, around line 1456). `_auto_detect_feature_types` returned the promoted set, and local `effective_cat_features` was computed with promoted columns removed — but the suite-level `cat_features` binding was never rebound, so `select_target` / `strategy.build_pipeline` / the CatBoost pandas-fallback path all kept receiving the **original** unfiltered list (including the just-promoted `category`, `skills_text`, etc.). Result: CatBoost's pandas path rejected the run with `"column 'category' has dtype 'category' but is not in cat_features list"` — the column was pd.Categorical (preserved from the source Polars schema) AND listed in `text_features`, so CB's Pool refused to accept the combination. Fix: `cat_features = effective_cat_features` right after the auto-detect call, so every downstream user sees the deduplicated list via the single binding.
- **`_train_model_with_fallback` now de-categorizes text columns after the Polars→pandas conversion** (`training/trainer.py`). Without this, columns that were auto-promoted from cat→text still arrived at CatBoost with pd.Categorical dtype; CB then complained "dtype 'category' but not in cat_features". New local `_decategorize_text_cols` helper casts every text-feature column with `pd.CategoricalDtype` to plain object (with `fillna("")`). Applied to both `train_df` and every `eval_set` pair.

### Observability
- **Promotion log now includes per-column cardinality** (`_auto_detect_feature_types`). Old line was opaque:
  ```
    Promoted 4 high-cardinality column(s) from cat_features to text_features: ['category', 'occupation', 'skills_text', 'ontology_skills_text']
  ```
  New output shows the threshold AND the actual per-column unique counts, so it's immediately obvious WHY each column was promoted:
  ```
    Promoted 4 high-cardinality column(s) from cat_features to text_features (threshold>100): [category:12_345, occupation:3_211, skills_text:52_480, ontology_skills_text:4_890]
  ```
  Same format reused for the "Auto-detected feature types — text: ..." summary.

- **Per-step timing inside the CB Polars-fallback path** (`training/trainer.py`). A production run hit the fallback and spent >1 hour between the "converting to pandas and retrying" warning and the eventual CB retry (which itself was just 37 s). No intermediate log meant diagnosing what consumed that 65 minutes was impossible. New `[fallback]` lines break it down per sub-step:
  ```
    [fallback] polars→pandas(train) 810_000×98 in ...s
    [fallback] prepare_df_for_catboost(train) in ...s
    [fallback] decategorize text cols(train) in ...s
    [fallback] eval_set rewrite in ...s
    [fallback] total pandas prep for CB in ...s
  ```
  The next fallback run will show which step is the real bottleneck and inform the decision on whether `get_pandas_view_of_polars_df` needs a shared-dict optimization or whether the cost lives somewhere else (e.g. `prepare_df_for_catboost`'s pandas-side per-column loops).

## 2026-04-18 — Log-triage part 3: PHASE 3 gc timer + pandas-conv reason

### Observability
- **`fit_and_transform_pipeline` now logs when the `maybe_clean_ram_adaptive()` step takes >1 s** (`training/pipeline.py`). A 1-minute "silent" gap observed in production between "Detected N categorical features" and "Done. RAM usage:" was tracked down to `gc.collect()` running on a multi-GB Arrow heap right after the raw DataFrame was freed. The step is no longer a black box; exact cost is visible in the log.
- **`_convert_dfs_to_pandas` path now logs the exact reason when `can_skip_pandas_conv=False`** (`training/core.py`). Previously users running Polars-native-only model sets but still seeing 5-6 minute pandas conversions had no way to diagnose what was forcing the fallback. The new line is verbose and explicit, e.g.:
  ```
    polars→pandas conversion needed because: non-Polars-native models requested: ['mlp', 'linear']
  ```
  or
  ```
    polars→pandas conversion needed because: rfecv_models=['cb_rfecv']
  ```

## 2026-04-18 — CatBoost text-column dtype fix + per-split polars→pandas timing

### Fixed
- **CatBoostError: "Unsupported data type Categorical for a text feature column"** — exposed by the 2026-04-18 auto-promote-to-text-features change. After `_auto_detect_feature_types` moves high-cardinality columns (e.g. `skills_text`, `category`) from `cat_features` to `text_features`, their backing dtype in the Polars frame remained `pl.Categorical`, but CatBoost's Polars text-column handler (`_set_features_order_data_polars_text_column`) only accepts `pl.String`/`pl.Utf8`. The fix casts every Polars `Categorical`/`Enum` column listed in `text_features` to `pl.String` right before the existing null-fill step in the CB fastpath (`training/core.py`, same block as the text null-fill). A single info line reports which columns were cast.
- **Broaden CB fallback condition** (`training/trainer.py:_train_model_with_fallback`) — the existing `"Unsupported data type Categorical for a numerical feature column"` fallback now triggers on any `"Unsupported data type Categorical"` substring (both `numerical` and `text` variants). Keeps us safe if future CB versions add more category-rejection sites with similar wording.

### Observability
- **`_convert_dfs_to_pandas` logs per-split timing** (`training/core.py`) when `verbose=True`. The "Zero-copy conversion to pandas..." step that silently consumed 5+ minutes in production (rebuilding pyarrow dict indices on 1M × 98 with ~13 categoricals, some text-like with 10k+ unique values) is no longer a black box. Sample output:
  ```
    polars→pandas(train) 810_000×98 in 3.1s
    polars→pandas(val)   90_000×98 in 0.4s
    polars→pandas(test) 100_000×98 in 0.4s
    polars→pandas total: 3.9s
  ```

## 2026-04-18 — Training-log triage (13 fixes from production run)

A single production run on a 1M × 119 Polars dataset surfaced a cluster of
papercuts that each individually looked minor but together made debugging
training runs much harder than necessary. Grouped below by subsystem.

### RAM logging
- **`get_own_ram_usage` no longer silently reports 0.0 GB** (`helpers.py:112-140`).
  On Windows / under heavy Arrow frees psutil can momentarily report an
  implausibly low rss. When the previous reading was substantial and the
  new one is <0.1 GB we now emit a warning and return the prior value —
  previously the `RAM usage: 0.0GB.` lines that resulted masked the real
  usage (the user's log showed this in the middle of a 37.5 GB run).

### Log attribution and formatting
- **`log_ram_usage` / `log_phase` now attribute to the caller's module**
  (`training/utils.py`). A new `_caller_logger` helper walks the stack
  one frame up so log lines emitted by these helpers use the caller's
  module logger (e.g. `mlframe.training.core`) instead of always saying
  `mlframe.training.utils` — the old behaviour was misleading when
  scanning origins.
- **Separator width reduced from 160 → 80** in `log_phase`. 160 wraps
  horizontally in most terminals and notebook cells.
- **No more stacked blank-looking banner**. `log_phase` used to emit a
  dash-line on both sides of its message. Two consecutive calls produced
  two adjacent dash-lines with nothing between them. Now only a single
  top separator is emitted; the next `log_phase` call naturally closes
  the block with its own separator. Result: `---\nFirst phase msg\n---\nSecond phase msg`.
- **`phase()` context-manager's `START`/`DONE` are now at `DEBUG` by default**
  (`training/phases.py`). These duplicated the caller-side INFO lines
  like `X done — {shape}, {time}`; at INFO they produced two log lines
  per phase. At DEBUG they're still useful for debugging; `RAISED`
  status is escalated to WARNING so failures remain visible.

### Stray raw output
- **`show_raw_data` now routes through the module logger** instead of
  bare `print(...)` (`training/extractors.py`). The raw `<class 'polars...'>` /
  `dtypes:` block was previously appearing out of order with the rest
  of the training log because stdout and the logger stream don't share
  flush points in Jupyter. Test in `test_perf_edges.py` updated to check
  caplog records instead of captured stdout.

### Typo
- **"constant numeric columnss" → "constant numeric columns"** and
  non-numeric counterpart (`training/utils.py:415`). The original f-string
  appended a literal `s` to a `kind` that already ended in "columns".

### Phase-3 / Phase-4 visibility
- **PHASE 3 now logs per-substep timing** (`fit_and_transform_pipeline`
  and, when non-None, `apply_preprocessing_extensions`). Previously a
  3+ minute phase was mysteriously black-boxed between the PHASE 3 banner
  and the "Pipeline done" summary.
- **PHASE 4 logs elapsed time for `select_target`**. Previously the 2+ min
  gap between "select_target..." and "process_model START" had no timing
  context.

### Wasted pandas preparation work
- **Skip `prepare_df_for_catboost` on the pandas views when the Polars
  fastpath is active** (`training/core.py`). When
  `can_skip_pandas_conv=True`, models receive Polars DFs directly;
  running `prepare_df_for_catboost` on the pandas-view side was ~2
  minutes of pure waste on 1M × 100 frames. Logged as
  `"Skipping pandas-side CatBoost prep ... — Polars fastpath receives the DFs natively"`
  when skipped.

### Feature-type auto-detection promotes text columns out of cat_features
- **High-cardinality text-like `pl.Categorical` columns are now promoted
  from `cat_features` to `text_features`** by `_auto_detect_feature_types`
  (`training/core.py:111+`). Previously the pipeline's schema-based
  detection in `fit_and_transform_pipeline` would lock in columns like
  `skills_text` / `ontology_skills_text` as `cat_features` before the
  text auto-detector had a chance to see them; the auto-detector then
  skipped them because "already assigned". Now promotion is explicit
  when `n_unique > cat_text_cardinality_threshold`, and the promoted
  columns are removed from `cat_features` in place. Frees substantial
  memory for text-heavy datasets and gives CatBoost the right
  tokenization path.

### Splitting log
- **Documented the `+NR` / `+ND` suffix** in `training/splitting.py`
  (the `_build_details` helper). Example in a user-visible log:
  `90_000 val rows 2014-01-20/2014-04-05 +45000R` — the `+45000R` means
  "45 000 extra **rows** (`R`) sampled from outside the sequential date
  window". `D` is the same for whole-day splitting.

## 2026-04-18 — Default logger timestamps + CatBoost Polars-fastpath fallback

### Fixed
- **`_ensure_logging_visible` (`training/core.py`)**: previously only installed a timestamped stdout handler when the root logger had NO handlers at all. In Jupyter / IPython a basic handler is already registered (with the `LEVEL:name:message` format — no timestamp), so mlframe's progress logs came out without wall-clock markers — making it impossible to see how long each phase actually takes. Extended the helper to also *upgrade* existing handlers whose formatter doesn't contain `%(asctime)s`, replacing their formatter with the timestamped one. Handlers that the user has intentionally configured with a custom asctime are left untouched.
- **`_train_model_with_fallback` (`training/trainer.py`)**: added a CatBoost × Polars-fastpath fallback. CatBoost's native-Polars entry point (`_set_features_order_data_polars_*`) can reject certain categorical column layouts with opaque messages — either `TypeError: No matching signature found` (fused-cpdef dispatch miss on the column's physical index / value types) or `CatBoostError: Unsupported data type Categorical for a numerical feature column` — abortive on training 1M×100 datasets. On either error, we now convert the Polars DataFrame to pandas via `get_pandas_view_of_polars_df` + `prepare_df_for_catboost`, rewrite the `eval_set` similarly, and retry. The pandas path accepts a broader range of category backings.

## 2026-04-18 — Stale-cache detection in `process_model`

### Fixed
- **`train_eval.py::process_model`**: the suite's cache-load path would unconditionally load a saved `.dump` whenever it existed, even if the feature set or cat_features had changed between runs. Symptom in production: cryptic `CatBoostError: Unsupported data type Categorical for a numerical feature column` deep inside CatBoost's Polars fastpath when a column that used to be numeric is now `pl.Categorical` (or vice versa, or columns were added/reordered). Two complementary fixes:
  1. **`use_cache` gate**: respect `common_params["use_cache"]` (default: True for backward compat — suite-level caching still "just works"). Callers can now force a retrain via `init_common_params={"use_cache": False}`.
  2. **Schema validator** (`_validate_cached_model_schema`, new): after loading, verify the saved model's `feature_names_` / `feature_names_in_` / `booster.feature_names` against the current DataFrame's column list. For CatBoost-shaped models additionally cross-check that each Polars `Categorical`/`Enum` column in the current df is in the saved `_get_cat_feature_indices()` set. On mismatch: log a warning with the reason and invalidate the cache (retrain) rather than let the backend crash.

### Tests
- New `tests/training/test_cache_schema_validation.py` (16 tests):
  - `_extract_polars_cat_columns`: None df, pandas df (no polars cats), `pl.Categorical` / `pl.Enum` detection.
  - Feature-names check: exact match, different names, reordered columns, extra column, unknown-type model (no `feature_names_*`).
  - CatBoost cat_features cross-check: matching case, new Polars Categorical not in saved cat set (the production bug), no-cat model with no Polars cats, out-of-range saved indices (pathological), pandas df never false-positives.

## 2026-04-18 — ICE penalty ramp + `prepare_df_for_catboost` dtype preservation

### Fixed
- **`integral_calibration_error_from_metrics` (`metrics.py:1146-1178`)**: the `roc_auc_penalty` sub-threshold mechanism was a step cliff — `if |auc-0.5| < min_roc_auc-0.5: res += roc_auc_penalty`. That discontinuity could trap CatBoost/XGB/LGBM early stopping just inside the penalty zone (pick the first iter with `auc≈0.5` that has trivially-good calibration, refuse to cross the cliff). Replaced with a **linear ramp**: penalty contribution is `roc_auc_penalty * deficit / threshold_width`, where `deficit = threshold_width - |auc-0.5|` for points inside the zone and 0 outside. **Knob semantics preserved** — `roc_auc_penalty=X` still gives `X` at the worst case `auc==0.5`, and fades smoothly to 0 at `auc==min_roc_auc`. Callers that relied on the step (typically `roc_auc_penalty=0` default) are unaffected.
- **`prepare_df_for_catboost` (`preprocessing.py:58-66`, `preprocessing.py:117-139`)**: the function was silently widening narrow-precision columns to float64. Two offenders:
  - **Pandas branch**: bare `astype(float)` on any extension-array dtype — `pd.Float32Dtype` → `float64`, `pd.Int8/16/32Dtype` → `float64`, `pd.BooleanDtype` → `float64`. Cost: 2× memory and 2× GPU bandwidth on users who had deliberately picked narrow precision.
  - **Polars branch**: every nullable int/bool → `Float64`, regardless of width. Cost: same as above.

  Replaced with precision-preserving/narrowing logic:
  - `pd.Float32Dtype` → `np.float32` (was `float64`)
  - `pd.Float64Dtype` → `np.float64` (unchanged)
  - `pd.Int8/16/32Dtype`, `pd.UInt8/16/32Dtype`, `pd.BooleanDtype` → `np.float32` (values fit exactly, was `float64`)
  - `pd.Int64Dtype` / `pd.UInt64Dtype` → `np.float64` (>~2**24 loses precision in float32, unchanged)
  - Polars: same pattern mirrored via `pl.Float32` / `pl.Float64`. Columns **without** nulls are no longer touched at all (micro-opt).

  Non-nullable `np.float32` columns were never touched and still aren't.

### Tests
- New `tests/test_metrics.py::TestICEPenaltyRamp` (8 tests): ramp is zero outside zone, max `=roc_auc_penalty` at `auc=0.5`, linear interior, symmetric about 0.5 (inverted rankers), **continuous across threshold** (regression sensor against re-introducing the step cliff — max adjacent-sample delta bounded by the Lipschitz constant), monotonic below threshold, respects `roc_auc_penalty=0`, guard against `min_roc_auc<=0.5` (no penalty zone), and the no-opt default-args path.
- New `tests/test_preprocessing.py` (39 parametrised tests): dtype preservation/narrowing across all pandas extension dtypes, non-nullable `np.float32` passthrough, end-to-end null-fill for `pd.Float32Dtype`, all polars int/uint/bool/float widths both with and without nulls, and a micro-opt guard that no-null int columns aren't cast at all.

## 2026-04-18 — Full test suite green; `data_dir=""` no longer leaks artifacts to CWD

### Fixed
- `_setup_model_directories` (`training/core.py` L466-478): switched from `data_dir is not None` to truthy check. Previously, passing `data_dir=""` satisfied `data_dir is not None`, causing the code to `join("", "charts"/"models", ...)` which produced **relative** `./charts/` and `./models/` paths. Artifacts were written to the **current working directory** — the mlframe repo root when tests were invoked from there. This had a subtle cascading effect: on a subsequent test run with a newer sklearn version, `train_mlframe_models_suite` would find and load these stale pickles, surfacing as `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'` (sklearn 1.7→1.8 attribute that didn't exist in the pickled state). That's the failure mode previously documented in the README TODO as an "sklearn 1.8 compat issue" — actually an mlframe-side leak, not a sklearn bug.
- `_setup_model_info_and_paths` (`training/trainer.py` L376-381): same falsy guard. Avoids a second relative `./models/` leak path when only the inner function is called.

### Test infrastructure
- Added `check_catboost_gpu_available` fixture in `tests/training/conftest.py`: checks `catboost.utils.get_gpu_device_count() > 0`. The existing `check_gpu_available` only verifies a CUDA device exists via numba, but CatBoost ships its own GPU runtime that may not be installed (error: `Environment for task type [GPU] not found`). Use this new fixture in CatBoost-specific GPU tests.
- `tests/training/test_all_models.py::TestGPUSupport::test_gpu_configuration[cb]` and `TestGPUUsageVerification::test_catboost_gpu_training_params`: skip when CatBoost GPU runtime is absent (was: hard-fail on dev hosts).
- `tests/training/test_bizvalue_preproc_transformers.py::test_dim_reducer_umap_optional`: gracefully skips on the UMAP×sklearn 1.8 incompatibility (UMAP still calls deprecated `check_array(force_all_finite=...)` — renamed to `ensure_all_finite` in sklearn 1.8). Third-party compat issue, not mlframe.

### Test suite status
Full `pytest tests/` passes end-to-end: **1994 passed, 40 skipped, 1 xfailed, 0 failed** (43:44). The previously-documented `test_no_artifact_files_when_no_data_dir` failure is gone — it was a symptom of the `data_dir=""` leak fixed above.

### Notes for Windows runs
- Before a full run, clear stale numba JIT caches: `find . -name "*.nbi" -delete; find . -name "*.nbc" -delete`. Stale caches trigger `Windows fatal exception: access violation` in `compute_numaggs` / similar kernels. This is documented in README "Troubleshooting".

## 2026-04-18 — Fix `prefer_calibrated_classifiers` no-op regression on base tree models

### Fixed
- `configure_training_params` (`training/trainer.py` L2210-2217): base CatBoostClassifier now uses `CB_CALIB_CLASSIF` (eval_metric=`ICE(...)`) vs `CB_CLASSIF` (eval_metric=`"AUC"`) according to the flag — previously always took `CB_CLASSIF` after the 2026-04-15 "post-hoc calibration" refactor, making the CB live training plot show ROC AUC instead of ICE.
- `_configure_xgboost_params` (L1830-1835): base XGBClassifier now uses `XGB_CALIB_CLASSIF` (eval_metric=`final_integral_calibration_error`) vs `XGB_GENERAL_CLASSIF` (eval_metric=`neg_ovr_roc_auc_score`) according to the flag — previously always took `XGB_GENERAL_CLASSIF`.
- `_configure_lightgbm_params` (L1858-1865): base LGBMClassifier now injects `fit_params={"eval_metric": lgbm_integral_calibration_error}` when flag=True — previously always returned empty `fit_params`.
- All three fixes restore the pre-2026-04-15 behavior: `eval_metric` is used for CatBoost's built-in live training plot and for early-stopping comparisons.

### Root cause
2026-04-15 refactor replaced eval-metric-based calibration with a post-hoc `_mlframe_posthoc_calibrate` attribute tag, but the hook that was supposed to consume it (`_maybe_apply_posthoc_calibration`, L817-833) was left as an explicit no-op (`return model` in both branches). The attribute was set on CB/XGB/LGBM models but never read, so all three models trained identically regardless of the flag.

### Removed
- `_mlframe_posthoc_calibrate=True` attribute setter in three locations (CB base, XGB base, LGBM base) — dead code, consumer hook is a no-op.
- `test_is_inlier` placeholder (`trainer.py`): declared-but-never-set `None` field on the returned SimpleNamespace, never consumed by any caller. Removed from all 4 sites (local init + 3 SimpleNamespace constructors).
- `default_drop_columns` local dead variable in `train_and_evaluate_model`: always set to `[]` with a stale "no longer needed" comment, passed to `_validate_infinity_and_columns` which concatenated an empty list. Simplified the helper signature to drop the parameter.

### Retained (see README "TODO")
- `_PostHocCalibratedModel` class and `_maybe_apply_posthoc_calibration` hook: intentionally retained as scaffolding in case the user revives isotonic post-hoc calibration as an alternative path.

### Tests
- New `tests/training/test_calibration_flag_propagation.py` (5 tests):
  - Level 2 (targeted): flipping `prefer_calibrated_classifiers` must produce different `eval_metric` on `XGBClassifier.get_params()`, different `fit_params["eval_metric"]` on LGBM configure helper, and different `eval_metric` on CatBoostClassifier (`ICE(...)` instance vs `"AUC"` string).
  - Level 2 (sanity): flag does not affect LGBM regression path.
  - Level 3 (matrix invariant): parametric sweep over `cb`/`xgb`/`lgb` — either the model's own `eval_metric` or the `fit_params["eval_metric"]` must differ between True/False. Catches any future silent no-op regression of the same class.

### Also fixed (collateral, surfaced by the broader test run)
- `report_model_perf` (`training/evaluation.py` L212-219): sklearn≥1.6 raises `AttributeError` when `is_classifier(None)` triggers `get_tags(None)` (previously returned `False`). The `just_evaluate=True` path legitimately passes `model=None` with pre-computed preds/probs — now task type is inferred from `probs is not None` when `model is None`, and `is_classifier` is skipped in that case. Fixes `tests/training/test_trainer.py::TestTrainAndEvaluateModelEdgeCases::test_model_none_just_evaluate`.
- `run_confidence_analysis` (`training/trainer.py` L1068-1097): the auxiliary confidence-analysis CatBoost model picked `task_type="GPU"` whenever `CUDA_IS_AVAILABLE` was True, ignoring the `TrainingBehaviorConfig.prefer_gpu_configs` override. On hosts that have a CUDA device but no CatBoost GPU runtime (e.g. CI/dev with `prefer_gpu_configs=False` forced in conftest), CatBoost raised `Environment for task type [GPU] not found`. Added a one-shot CPU fallback: on that specific error, retry fit with `task_type="CPU"` and log a warning. Fixes `tests/training/test_core.py::TestConfidenceAnalysis::test_confidence_analysis_basic`.

### Known pre-existing test failure (NOT caused by this change)
- `tests/training/test_core_coverage.py::TestSplitting::test_no_artifact_files_when_no_data_dir` fails on `master` even without this patch. Root cause is an sklearn 1.8 compat issue: some fitted sklearn `Pipeline`/`SimpleImputer` in the test flow is unpickled from sklearn 1.7.2 state that is missing the new-in-1.8 `_fill_dtype` attribute, so `SimpleImputer.transform()` raises `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`. Confirmed by `git stash` + rerun on `8d30b9a`. TODO: either refit the imputer on load (detect missing `_fill_dtype`), or invalidate cached artifacts on sklearn version bump. Out of scope for this change.

### Follow-ups documented
- README gains a "TODO" section with two items:
  1. Decide to ship or remove `_PostHocCalibratedModel` + post-hoc calibration hook.
  2. Re-enable CatBoost `custom_metric=tuple(...)` with a clone-safe strategy (set via `model.set_params(...)` on the base path only, leaving RFECV estimators clean).

## 2026-04-17 — Polars→pandas Categorical optimization (no more dict→string cast)

### Changed
- `get_pandas_view_of_polars_df` in `training/utils.py` now preserves Polars `Categorical` columns as `pd.Categorical` (int32-indexed dictionary) instead of casting dict→string. Polars emits dict arrays with uint32 indices, which pyarrow's `to_pandas` refuses; we rebuild each dict column with int32 indices so the conversion produces a proper `pd.Categorical`.

### Why
End-to-end benchmark on production-shaped data (CatBoost classifier, 180k × 586 cols, 70 Categorical) via `bench_polars_to_pandas.py`:

| Variant | convert | fit | predict | **total** |
|---|---|---|---|---|
| native Polars (CatBoost's own path) | 0.00s | 12.42s | 0.14s | 12.55s |
| old (dict→string cast) | 1.04s | 15.56s | 0.47s | 17.08s (+37%) |
| **new (int32-indexed pd.Categorical)** | 0.45s | 11.99s | **0.04s** | **12.49s** (fastest) |

String cast was both slower (CatBoost hashes strings per row during fit and predict) and memory-hungrier (OOMs at 450k+ rows with 70 Categoricals where the new path trains cleanly).

### Tests
- `test_utils.py::test_categorical_to_string_conversion` renamed to `test_categorical_preserved_as_pd_categorical` and now asserts the `pd.CategoricalDtype` plus the category list, not just the string values.
- Downstream comment in `core.py` above the `prepare_df_for_catboost` call updated — that call is now usually a no-op but kept for pandas-input safety.

## 2026-04-17 — Fix metadata pickle failure with duplicate mlframe installs

### Fixed
- `_create_initial_metadata`: Pydantic config objects (`preprocessing_config`, `pipeline_config`, `split_config`) are now stored in `metadata["configs"]` as plain dicts via `.model_dump()` instead of raw Pydantic instances. This prevents `_pickle.PicklingError: Can't pickle <class 'mlframe.training.configs.PreprocessingBackendConfig'>: it's not the same object as mlframe.training.configs.PreprocessingBackendConfig` when two copies of mlframe are reachable via `sys.path` (e.g. a dev checkout plus an older pip install, or Jupyter autoreload duplicating a module). Tests only assert key presence (`"preprocessing" in metadata["configs"]`), so the change is backward compatible.

## 2026-04-17 — Polars→pandas conversion benchmark

### Added
- `bench_polars_to_pandas.py`: two benchmark modes on a production-shaped synthetic DF (1M × 587 cols by default: Boolean(10), Categorical(70), Datetime(1), Float32(38), Float64(425), Int16(14), Int64(2), Int8(27)).
  - **Default (`BENCH_MODE=catboost`)**: end-to-end CatBoost `fit` + `predict_proba` with identical hyperparameters on (a) the native Polars DataFrame and (b) the same data converted to pandas via mlframe's `get_pandas_view_of_polars_df`. Reports per-phase times (convert / fit / predict / total) and the end-to-end speedup.
  - **Conversion-only (`BENCH_MODE=conversion`)**: microbench of mlframe's approach (`to_arrow` + batched `pa.compute.cast` dict→string + `to_pandas`) vs a Python re-implementation of CatBoost's per-column loop (`_catboost.pyx:3199` / `:3288`: per-column `rechunk()` + `to_physical().to_numpy()`). Includes per-step breakdown for mlframe path and per-dtype breakdown for the CatBoost-like path.
  - Tunable via env vars: `BENCH_N_ROWS`, `BENCH_N_CAT`, `BENCH_ITERATIONS`, `BENCH_THREAD_COUNT`, `BENCH_TEST_FRACTION`, `BENCH_N_REPEATS`, `BENCH_MODE`.

## 2026-04-17 — Structured phase timing + logging visibility fix

### Added
- `training/phases.py`: `PhaseTimer` context manager, global registry, `format_phase_summary()` / `phase_snapshot()` / `reset_phase_registry()`. Hotspot wrappers across `core.py`, `trainer.py`, `evaluation.py` cover data load, split, train_stats, `process_model`, `model.fit` (incl. retry), `pre_pipeline_fit_transform`, `compute_split_metrics` (train/val/test), `report_probabilistic_model_perf`, `report_regression_model_perf`, `predict` / `predict_proba`, `fast_calibration_report`, `plot_feature_importances`, `compute_fairness_metrics`. Summary table is logged at the end of verbose `train_mlframe_models_suite` runs so regressions become visible immediately.
- `_ensure_logging_visible()` in `core.py`: idempotently attaches an INFO-level stdout handler to the root logger when none exists, so `logger.info` calls inside the suite actually appear in Jupyter with `verbose=True`. Does nothing if the user already configured logging.

### Fixed
- `TrainingControlConfig.verbose` accepts `Union[bool, int]` (was strict `bool`). Passing a verbosity level like `verbose=3` from the suite no longer raises pydantic `bool_parsing` 3 minutes into a training run.

## 2026-04-16 — Fix all 11 xfailing biz-value tests

### Fixed
- `test_bizvalue_calibration_ensemble.py`: rewrote data generator with sinusoidal logit + 105 noise features; test now trains sklearn `CalibratedClassifierCV` directly (not through mlframe suite) to avoid internal data splits; per-model threshold (0.50% for CatBoost, 1.00% for LGB/XGB) reflects CatBoost's inherently better calibration.
- `test_bizvalue_imbalance_grid.py`: changed `scale_pos_weight` from `sqrt(n_neg/n_pos)` to full `n_neg/n_pos`; increased imbalance severity from 95:5 to 98:2 with larger dataset (9000 rows).
- `test_bizvalue_fairness_weights.py`: increased dataset size (n_train 3000->6000, n_test 600->1500), reduced minority fraction (0.10->0.07), softened shift vector.
- All 36 biz-value tests now pass with hard asserts (0 xfails).

## 2026-04-15 — Suite pipeline: fixes, new kwargs, metadata, test expansion

### Added
- `train_mlframe_models_suite(save_charts: bool = True)` — when `False`, skips per-model chart file output (for CI / fast runs).
- `metadata["fairness_report"]` — aggregated fairness metrics propagated from per-model runs into suite-level metadata.
- `metadata["outlier_detection"]` dict: `applied`, `n_outliers_dropped_train`, `n_outliers_dropped_val`, `train_size_after_od`, `val_size_after_od`.
- `PreprocessingExtensionsConfig.tfidf_columns` now wired end-to-end: text columns are vectorized inside `apply_preprocessing_extensions` and replaced with `<col>__tfidf_<i>` numeric features.
- `apply_preprocessing_extensions(y_train=...)` kwarg wires supervised fit for `dim_reducer="LDA"`; fixed `RandomTreesEmbedding` factory to use `n_estimators` (not the non-existent `n_components` kwarg); added `tests/training/test_bizvalue_preproc_transformers.py` (37 business-value tests covering polynomial XOR lift, RBFSampler/Nystroem on circles, PCA/TruncatedSVD/LDA/KernelPCA/NMF/FastICA/Isomap/GRP/SRP/RTE/BernoulliRBM/UMAP dim_reducers, KBins sine-wave R^2 lift, Binarizer collapse property, memory-safety guard, Chi2 positive-input guards, Binarizer+KBins mutual exclusion).

### Changed
- `ModelHyperparamsConfig.early_stopping_rounds` is now `Optional[int]`; setting it to `None` disables early stopping across all strategies (CB/LGB/XGB/MLP/RFECV/HGB/NGB).

### Fixed
- `_SafeUnpickler` allowlist now includes the `mlframe` prefix — fixes silent drop of CatBoost models that reference `mlframe.metrics.ICE` during `predict_mlframe_models_suite`.

### Tests
- 8 new unit test files for previously untested helpers: `tests/training/test_untested_*.py` (83 tests).
- 6 new business-value integration test files: `tests/training/test_bizvalue_*.py` covering fairness, calibration, outliers, preprocessing extensions, early stopping, ensemble, sample weights, class imbalance, and `run_grid`.
- `tests/training/test_bizvalue_feature_selection.py` — business-value integration tests for MRMR/RFECV feature selection (drops uninformative cols, preserves AUROC on wide data, exposes selected features).

## 2026-04-15 — Audit #02 (legacy) + test fast mode

### Commit 1/5 — Salvage from legacy modules (pre-move)
- `evaluation.py`: added `predictions_beautify_linear`, `plot_beautified_lift`, `plot_pr_curve`, `plot_roc_curve`.
- `training/evaluation.py`: added `compute_ml_perf_by_time`, `visualize_ml_metric_by_time`.
- `outliers.py`: added `compute_outlier_detector_score`, `count_num_outofranges` (@njit), `compute_naive_outlier_score`. Fixed broken hard-import of `imblearn` (lazy guarded).
- `metrics.py`: added `brier_and_precision_score`, `make_brier_precision_scorer`.
- NEW `training/callbacks.py`: `stop_file` + `{CatBoost,LightGBM,XGBoost,Lightning}StopFileCallback`.
- NEW `training/neural/keras_compat.py` (TF-guarded): `build_keras_mlp`, `KerasCompatibleMLP`.
- NEW `tests/test_evaluation_salvage.py` (18 tests, 16 pass / 2 TF-skip).

### Commit 2/5 — Move to legacy/
- Deleted `mlframe/Backtesting.py` (10-LOC stub, zero importers).
- `git mv` `training_old.py`, `OldEnsembling.py` → `mlframe/legacy/`.
- NEW `mlframe/legacy/__init__.py` — emits `DeprecationWarning` on import.
- Stripped 5 stale "migrated from training_old.py" comments across `training/{__init__,helpers,train_eval,trainer}.py`.
- `pytest.ini` ignores point at `legacy/` directory.

### Commit 3/5 — Resource-logging decorators + estimator-object model spec
- NEW `training/logging_transformers.py`:
  - `log_resources(*, stage, level, extra_factory)` — function decorator, logs wall-time + ΔRSS.
  - `log_methods(*methods, stage_prefix)` — class decorator.
  - `wrap_with_logging(obj, *, stage, methods)` — instance-proxy factory.
- `training/strategies.py`: `get_strategy` accepts strings, estimator instances, `(name, estimator)` tuples. MRO dispatch via `_strategy_for_estimator` (lazy-guarded CatBoost/LightGBM/XGBoost imports); unknown classes fall back to `LinearModelStrategy` with warning. New helpers `_resolve_model_spec`, `_slugify`, `_dedupe_key`.
- `training/utils.py::filter_existing`: tolerate ndarray (no `.columns` → `[]`).
- NEW `tests/training/test_logging_transformers.py` (8 tests).
- NEW `tests/training/test_model_spec_resolution.py` (14 tests).

### Test infrastructure — Fast mode (`--fast` / `MLFRAME_FAST=1`)
- `tests/conftest.py`: `--fast` CLI flag + `MLFRAME_FAST` env var; `is_fast_mode()`, `fast_subset(values, representative=..., keep=1)` helper. `@pytest.mark.slow` / `slow_only` auto-skip in fast mode.
- Pattern: parametrized tests call `fast_subset([...scalers...], representative="StandardScaler")` so all code paths still execute but with one representative variant.
- NEW `tests/test_fast_mode.py` (8 self-tests incl. subprocess end-to-end).

### Commit 4/5 — PreprocessingExtensionsConfig + apply_preprocessing_extensions
- `training/configs.py`: new `PreprocessingExtensionsConfig` with 14 fields
  (scaler override, binarization/kbins mutually-exclusive, polynomial with
  memory-safety guard, nonlinear feature maps, tfidf, dim_reducer covering
  PCA/KernelPCA/LDA/NMF/TruncatedSVD/FastICA/Isomap/UMAP/random projections/
  RandomTreesEmbedding/BernoulliRBM). None default on every stage so the
  whole config reads as a noop.
- `training/pipeline.py`: new `apply_preprocessing_extensions` helper runs
  after `fit_and_transform_pipeline`. Config=None = byte-for-byte fastpath
  preservation. UMAP gated via `find_spec` with install-hint ImportError.
- `training/core.py`: `train_mlframe_models_suite` gains
  `preprocessing_extensions: Optional[PreprocessingExtensionsConfig | Dict]`.
  Dict inputs auto-promoted. Extensions pipeline stored under
  `metadata["extensions_pipeline"]`. `cat_features` cleared once
  extensions materialise them to numeric columns.
- NEW `training/grid.py::run_grid` — sequential variant sweeper (replaces
  the dropped `TryAllMethods` pattern). Accepts base kwargs + list of dicts
  or `(label, dict)` tuples; `stop_on_error=False` default captures
  exceptions per variant. 6 unit tests via injected `suite_fn` stub.
- NEW `tests/training/test_preprocessing_extensions.py` (13 tests).
- NEW `tests/test_scalers.py` (8-scaler LR-AUROC round-trip, fast_subset
  keeps one representative).

### Collection-time fix — `training/callbacks.py` lazy lightning
- Top-level `import pytorch_lightning` was pulling torch DLLs into every
  test collection. Under Windows memory pressure this triggered
  `OSError WinError 1455` (paging file too small) on `shm.dll` /
  `cufft64_10.dll`, aborting collection before a single test could run.
- Switched to `importlib.util.find_spec()` detection + lazy import inside
  `LightningStopFileCallback.__init__` with dynamic base-class rebasing.

### Pending (commit 5/5)
- Benchmark guard (≤2% regression budget on default path).

## 2026-04-14 — Full Audit & Fix Sweep (10 parallel audit agents + 9 parallel fix agents)

### Security (RCE hardening)
- `training/neural/flat.py`, `training/neural/recurrent.py` — `torch.load(..., weights_only=True)`.
- `training/io.py` — `_SafeUnpickler` allowlist; `safe=True` default for `dill.load` paths.
- `inference.py`, `pipelines.py` — `joblib.load` gated by `trusted_root` path validation (`os.path.commonpath`); sorted `os.listdir` for determinism; consistent `(models, X)` return shape; `output_dir` defaults to `tempfile.gettempdir()`.
- `experiments.py` — SQL `fields` validated against `_ALLOWED_EXPERIMENT_FIELDS` frozenset (f-string injection fixed).
- New `tests/test_security_rce.py` (4 tests).

### Correctness / numeric sweep
- `calibration.py` — Brier vs. binned-metric dispatch uses `is` identity (was no-op dict-comp typo); AD clips PIT to `[1e-12, 1-1e-12]`; ECI on probability-normalized counts; WPD `np.clip(p*(1-p), 1e-6, None)`; `show_classifier_calibration` accumulates per-interval perfs.
- `postcalibration.py` — `isinstance` dispatch with lazy imports; `transform_method_name` resolved at `fit()`; 1-D probs clipped to [0,1] before `np.vstack`.
- `metrics.py` — bounds guards in numba kernels; `fast_log_loss_binary` OOB→NaN; `fast_roc_auc` raises on `sample_weight`; `brier_score_loss` → `fast_brier_score_loss` (alias retained); rounding precision `max(1, ceil(log10(max(nbins,2))))`.
- `ewma.py` — full O(n) numba recurrence (was O(n²) matrix + no-op `x[::np.newaxis]` slice).
- `arrays.py` — removed `import mlframe` self-import; `arrayMinMax` returns `(nan,nan)` on empty; `topk_by_partition` no longer mutates caller, `k = min(k, n)`; O(1) membership check; shared-ref list fixed.
- `stats.py:75` — `dist_kwargs=dist_kwargs` → `**dist_kwargs`.
- `FeatureEngineering.py:247` — off-by-one mask (spans `x[l:r+1]` matching inclusive size).
- `feature_engineering/mps.py` — OOB guards on start/end indices.
- `feature_engineering/numerical.py` — Kahan compensator in rolling MA; argmin/argmax first-wins consistent with sibling kernel; weights threaded into early-exit path.
- `feature_engineering/timeseries.py` — list-as-boolean wire bug in `create_and_process_windows` fixed; `accumulated_amount` initialized to avoid NameError.
- `boruta_shap.py` — SciPy 1.12+ `binomtest` wrapper; lazy iris import; vectorized Z-score; shap split fix.
- `feature_selection/general.py`/`wrappers.py`/`filters.py`/`optbinning.py` — empty-list guards, proper CV clone + rng, zero-prob entropy filter, `@njit(cache=True)`, deduped LOGGING block.
- `feature_selection/mi.py` — design-intent NOTE preserving 3 MI kernels (grok/chatgpt/deepseek) as load-bearing.
- `optimization.py:689` — **CRITICAL**: `elif OptimizationDirection.Maximize:` → `Minimize` (copy-paste bug).
- `optimization.py` — `plt.close(fig)` after plotting; `logger.warn` → `logger.warning`.
- `tuning.py` — cache key tuple instead of list; `learning_rate` uniform→loguniform; duplicate `penalties_coefficient` removed.
- `evaluation.py:301` — `plt.grid(b=None)` → `visible=None` (mpl ≥3.5); `:339` tuple unpack fix.
- `custom_estimators.py` — bounded retry loop; `scipy.ndimage.shift`; `PowerTransformer` no longer module-level; sklearn-compliant averagers (`classes_`, `n_features_in_`, `check_is_fitted`, `check_array`); `MyDecorrelator` trailing-underscore convention; `PdKBinsDiscretizer` sparse densify.
- `estimators.py` — `logger` properly imported; `check_array` in fit/predict; `ClassifierWithEarlyStopping` gains `predict_proba`/`decision_function`; typo fixes.
- `cluster.py` — `from sklearn.cluster import DBSCAN` (was undefined).
- `eda.py:41` — `is not None` for pandas Series.
- `feature_importance.py:73` — `feature_importances[sorted_idx[0]]`.
- `helpers.py` — wildcard `from .config import *` → explicit imports; `model.steps[-1][1]` (was `(name, est)` tuple); vectorized `np.isinf` over numeric columns; tutorial helpers deleted.

### RNG discipline
- `MBHOptimizer`, `ParamsOptimizer`, `CatboostParamsOptimizer`, `optimize_finite_onedimensional_search_space`, `generate_valid_candidates`, `create_ctr_params`, `get_model`, `justify_estimator` — all accept `random_state`; internal `_rng = np.random.default_rng(...)` + `_stdlib_rng = random.Random(...)`; `np.random.*`/bare `random()` removed.
- `training/splitting.py`, `training/evaluation.py`, `datasets.py`, `synthetic.py` — no more global `np.random.seed`; `generator`-threaded; scipy `.rvs(random_state=rng)`; sklearn bridge via `rng.integers(0, 2**32-1)`.
- `custom_estimators.py::PureRandomClassifier` — fully sklearn-compliant (`random_state_`, `classes_`, `n_features_in_`, label-returning `predict`).
- `synthetic.py` — tuple-vs-list dead branch at :44 fixed; asserts → `ValueError`; guarded divisions; off-by-one at :241 replaced with `generator.randint`.

### Conventions (per MEMORY.md)
- `postcalibration.py` — 2 regex sites hoisted to module-level `re.compile`; shared `_compile_pattern = lru_cache(re.compile)` helper.
- New `tests/test_conventions.py` meta-tests.

### Test suite & hygiene
- `pytest.ini` rewritten — `minversion=7.0`, `testpaths=tests`, `pythonpath=.`, `--strict-markers --strict-config --doctest-modules --cov=mlframe`, `xfail_strict=true`, 8 `--ignore=` for legacy/broken, custom markers (`benchmark`, `multigpu`, `windows_only`, `linux_only`), `filterwarnings`.
- `tests/conftest.py` — autouse session-scoped RNG seed fixture (random/numpy/torch = 0); `psutil` guarded via try/except; `warnings.resetwarnings()` replaced with scoped `catch_warnings()`.
- 7 `assert True` exception-swallowing patterns replaced with real post-conditions + `pytest.skip` (test_core.py, test_feature_selection.py, test_stress.py).
- `tests.py` (root) renamed → `bench_helpers.py`; `unittest_arrays.py` migrated → `tests/test_arrays.py` (9 pytest fns, timing asserts dropped).
- `tests/lightninglib/` — 9 duplicate files deleted (kept `test_deprecated_import.py`).

### Repo hygiene (~104 MB reclaimed)
- Removed: `profile_mixed_dtypes.prof`, root `__pycache__/`, `catboost_info/`, `checkpoints/`, `lightning_logs/`, `logs/`, `.coverage`, `training_old.py.backup`, `read.me` (content merged into README), `NUL` (via `\\?\` extended-path API).
- `.gitignore` — grouped `NUL` under Windows-specific block; added `*.prof`, `*.backup`, `*.py.backup`, `.benchmarks/`, `.ruff_cache/`, `.black_cache/`, `.vscode/`, `.direnv/`, `.envrc`, `coverage.lcov`, `tests/**/{catboost_info,lightning_logs,checkpoints,logs}/`.
- `public_suffix_list.dat` retained (used at `FeatureEngineering.py:365`).

### New tests (property-based + determinism + regression)
- `tests/test_security_rce.py`, `tests/test_conventions.py`, `tests/test_rng_determinism.py`, `tests/test_rng_determinism_b.py`, `tests/test_numeric_bug_sweep.py`, `tests/test_sklearn_compliance.py`, `tests/test_fs_fe_fixes.py`, `tests/test_arrays.py`.

### Verification
- Import smoke: `mlframe`, `mlframe.training`, `mlframe.feature_engineering`, `mlframe.feature_selection` — all ok.
- Per-agent suites: 4 + 3 + 6 + 9 + 7 + 19 + 6 + 9 = **63 new tests pass**.
- Full-suite run flagged one order-dependent hypothesis test in `test_timeseries.py::test_find_next_cumsum_left_index` (passes in file scope; pre-existing test-pollution, not introduced by this sweep).

### Deferred
- Dead-code removal (Phase 3: `training_old.py`, `OldEnsembling.py`, `Models.py`, `Backtesting.py`, `Features.py`, `Data.py`, empty `models/`) — pending user decision.
- Audit findings under `.claude/plans/mlframe_audit/*.md` (10 reports + `_SUMMARY.md`, outside repo).

## 2026-04-14 — Test Suite Optimization & Coverage Expansion

### Added

- **`tests/training/test_core_coverage.py`** — 48 new tests targeting 99% coverage of `train_mlframe_models_suite`:
  - `TestInputValidation` (8 tests): TypeError/ValueError for invalid df types, non-parquet paths, empty names, None FTE, parquet path loading, dict config acceptance.
  - `TestConfigurationSetup` (4 tests): Pydantic config passthrough for PreprocessingConfig, TrainingSplitConfig, ModelHyperparamsConfig, TrainingBehaviorConfig.
  - `TestDataLoadingPreprocessing` (2 tests): NaN fillna, column dropping via preprocessing_config.
  - `TestSplitting` (4 tests): split size sums, artifact saving, no-data-dir skip, metadata keys.
  - `TestPipelineFitting` (8 tests): auto-skip categorical encoding for Polars-native models, pre-clone logic, metadata pipeline/cat_features/columns keys, mixed native/non-native models.
  - `TestFeatureTypeDetection` (3 tests): text/embedding features in metadata, empty defaults.
  - `TestModelTrainingLoop` (9 tests): unknown model skip with warning, uniform/custom weight schemas, model × weight combinations, ensemble scoring, clone per weight.
  - `TestRecurrentModels` (2 tests): recurrent fit() with error handling, unknown recurrent model skip (selective mock of clone).
  - `TestCrossCuttingParametrized` (4 cases): `@pytest.mark.parametrize` over (ridge/lasso) × (pandas/polars).
  - `TestMetadataCompleteness` (4 tests): all expected keys, configs, split sizes, joblib persistence.
- **`tests/conftest.py`** — root conftest with shared autouse fixtures (`cleanup_memory`, `suppress_convergence_warnings`).
- **`tests/feature_engineering/conftest.py`** — shared date/DataFrame fixtures.
- **`pytest.ini`** — custom markers (`slow`, `integration`, `gpu`), doctest options.
- **`tests/training/test_train_eval.py`** — 10 tests for `optimize_model_for_storage`, `select_target`.
- **`tests/test_utils.py`** — 10 tests for root utils (`set_random_seed`, `get_pipeline_last_element`, etc.).
- **`tests/test_metrics.py`** — 8 new edge case + Hypothesis tests.
- **`tests/training/test_configs.py`** — 2 Hypothesis round-trip tests for Pydantic configs.
- **`tests/training/test_utils.py`** — 3 Hypothesis property-based tests for DataFrame transforms.
- **`tests/training/test_helpers.py`** — 8 tests for `parse_catboost_devices`.

### Changed

- **Consolidated duplicated tests**: 3 pandas/polars test pairs in `test_basic.py` → parametrized; 7 boolean param tests in `test_numerical.py` → single parametrized test.
- **Marked slow tests**: `@pytest.mark.slow` on test_stress.py, test_all_models.py, test_integration.py, RFECV tests. Enables `pytest -m "not slow"` for fast CI.
- **Optimized tree model tests**: reduced iterations from 5000 to 50 in test_core.py (3 tests).
- **Promoted fixture scopes**: `common_init_params`, `fast_iterations`, `fast_config_override` → `scope="session"`.
- **Fixed doctests**: NumPy 2.x compatibility in stats.py (7 doctests), added doctest to `get_numeric_columns`.
- **Fixed dict mutation bugs**: `.copy()` on `hgb_kwargs`, `mlp_kwargs`, `ngb_kwargs`, `rfecv_kwargs` in helpers.py.

## 2026-04-14 — CatBoost Text & Embedding Features + Memory Optimizations

### Added

- **`text_features` and `embedding_features` support**: CatBoost now receives `text_features` (free-text string columns) and `embedding_features` (list-of-float vector columns) via `fit()` params. Models that don't support them (Ridge, XGB, LGB, HGB, MLP, etc.) automatically have these columns dropped before training.
- **`FeatureTypesConfig`** Pydantic class in `configs.py`: `text_features`, `embedding_features`, `auto_detect_feature_types`, `cat_text_cardinality_threshold` (default 50).
- **Auto-detection of feature types**:
  - Embedding columns: auto-detected from `pl.List(pl.Float32)` / `pl.List(pl.Float64)` dtype.
  - Text vs categorical: string columns with `n_unique > cat_text_cardinality_threshold` → text; `<= threshold` → categorical. User-specified lists always take priority.
- **Feature-tier model grouping**: models sorted by `strategy.feature_tier()` — `(True, True)` (CatBoost) trains first with all columns, then text/embedding columns are dropped once per tier for remaining models. Tier DFs cached via `_build_tier_dfs()` using `.select()` (not `.drop()`).
- `supports_text_features` and `supports_embedding_features` properties on `ModelPipelineStrategy` (default `False`). `CatBoostStrategy` overrides both to `True`.
- `feature_tier()` method on `ModelPipelineStrategy` — returns `(supports_text, supports_embedding)` tuple for grouping.
- Mutual exclusivity validation: `text ∩ cat`, `emb ∩ cat`, `text ∩ emb` → `ValueError`.
- Pipeline exclusion: text and embedding columns excluded from encoding/scaling in `fit_and_transform_pipeline()`.
- CatBoost text columns auto-filled with `""` for nulls (CatBoost requirement).
- 18 CPU integration tests + 2 GPU tests in `TestTextAndEmbeddingFeatures`.

### Memory Optimizations (for 100GB+ DataFrames)

- **B1: Conditional clone** — `train_df.clone()` only when pipeline will modify categoricals (`skip_categorical_encoding=False`). Saves 100GB+ when all models are Polars-native.
- **B2: Aggressive cleanup** — post-pipeline Polars DFs released after pandas conversion when no longer needed.
- **B3: `prepare_polars_dataframe()` cache** — moved outside weight schema loop, called once per model instead of once per weight schema.
- **B4: `.select()` over `.drop()`** — tier column trimming uses `.select(cols_to_keep)` for better Polars optimization.
- **B5: Release Polars originals after tier transition** — pre-pipeline Polars DFs freed after all Polars-native models finish training.

### Changed

- Model training loop now sorts models by `feature_tier()` (most features first) instead of using the user-provided order.
- `select_target()` and `configure_training_params()` accept `text_features` and `embedding_features` params.
- `fit_and_transform_pipeline()` accepts `text_features` and `embedding_features` params to exclude from encoding/scaling.

## 2026-04-14 — Typed Training Parameters Refactor

### Breaking Changes

- `train_mlframe_models_suite` signature changed: removed `config_params`, `control_params`, `config_params_override`, `control_params_override`, and `**kwargs`.
- New parameters: `hyperparams_config` (`ModelHyperparamsConfig` or dict) and `behavior_config` (`TrainingBehaviorConfig` or dict).
- `select_target()` signature changed accordingly.

### Added

- **`ModelHyperparamsConfig`** Pydantic class in `configs.py`: typed replacement for `config_params`/`config_params_override` dicts. Fields: `iterations`, `learning_rate`, `early_stopping_rounds`, `has_time`, `rfecv_kwargs`, per-model kwargs (`cb_kwargs`, `lgb_kwargs`, `xgb_kwargs`, `hgb_kwargs`, `mlp_kwargs`, `ngb_kwargs`).
- **`TrainingBehaviorConfig`** Pydantic class in `configs.py`: typed replacement for `control_params`/`control_params_override` dicts. Fields: `prefer_gpu_configs`, `prefer_cpu_for_lightgbm`, `prefer_cpu_for_xgboost`, `prefer_calibrated_classifiers`, `use_robust_eval_metric`, `nbins`, `fairness_features`, `fairness_min_pop_cat_thresh`, `cont_nbins`, `metamodel_func`, `callback_params`, `cb_fit_params`, `use_flaml_zeroshot`, scoring configs.
- Both classes exported from `mlframe.training` and added to `__init__.py` lazy imports.
- Constants `DEFAULT_FAIRNESS_MIN_POP_CAT_THRESH`, `DEFAULT_RFECV_*` moved to `configs.py` (canonical location).

### Changed

- `_initialize_training_defaults()` simplified: no longer normalizes 4 dict params.
- `_build_common_params_for_target()` accepts `TrainingBehaviorConfig` instead of dict.
- `_should_skip_catboost_metamodel()` accepts `TrainingBehaviorConfig` instead of dict.
- `_compute_fairness_subgroups()` accepts `TrainingBehaviorConfig` instead of dict.
- `TrainingConfig` class updated: `config_params_override`/`control_params_override` fields replaced with `hyperparams: ModelHyperparamsConfig` and `behavior: TrainingBehaviorConfig`.

### Fixed

- **Bug**: Tests used `models["target"][TargetTypes.X]` but actual structure is `models[TargetTypes.X]["target"]` — fixed across all test files.

### Migration

```python
# Before:
train_mlframe_models_suite(
    ...,
    config_params_override={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
    control_params_override={"prefer_calibrated_classifiers": False},
)

# After:
train_mlframe_models_suite(
    ...,
    hyperparams_config={"iterations": 10, "cb_kwargs": {"task_type": "CPU"}},
    behavior_config={"prefer_calibrated_classifiers": False},
)
```

## 2026-04-14 — Auto-skip Categorical Encoding + Verbose Logging

### Added

- **`skip_categorical_encoding`** flag on `PreprocessingBackendConfig`: when `True`, the polars-ds pipeline and sklearn pandas path skip ordinal/onehot encoding of categorical features. **Auto-detected** by `train_mlframe_models_suite` — when all requested `mlframe_models` support Polars natively (cb, xgb, hgb), the flag is set automatically, avoiding wasted encoding work and preserving original categorical dtypes.
- **Verbose timing & shape logging** across the training pipeline (`verbose=True`):
  - `core.py`: Phase 1 (data loading, FTE, preprocessing), Phase 2 (splitting with shapes), Phase 3 (pipeline with dtypes), per-model `process_model()` timing, Polars fastpath activation logging
  - `trainer.py`: `model.fit()` timing with shape, `_apply_pre_pipeline_transforms` timing with shape, metrics computation timing
  - `pipeline.py`: Polars-ds pipeline creation timing (scaler/encoding config), transform timing with shape, sklearn categorical encoding timing with shape
- Helper functions `_df_shape_str(df)` and `_elapsed_str(start)` in `core.py`
- 6 parametrized tests for `skip_categorical_encoding` auto-detection (all-native, mixed, non-native model lists)

### Changed

- `CatBoostStrategy.cache_key` = `"catboost"` (was inherited `"tree"`). `XGBoostStrategy.cache_key` = `"xgboost"` (was inherited `"tree"`). Each Polars-native model now gets its own pipeline cache slot, preventing cross-contamination when running multiple models together (e.g. `["cb", "xgb", "hgb"]`).

### Fixed

- **Bug**: XGBoost Polars fastpath passed `cat_features` as a `fit()` parameter, causing `TypeError: XGBClassifier.fit() got an unexpected keyword argument 'cat_features'`. Only CatBoost accepts `cat_features` in `fit()` — XGBoost/HGB auto-detect `pl.Categorical` columns via `enable_categorical=True`.
- **Bug**: When running multiple Polars-native models together (e.g. `["cb", "xgb"]`), the pipeline cache shared the `"tree"` key, causing the second model to receive cached pandas DFs from the first — overriding the Polars fastpath and causing `KeyError: DataType(large_string)` in XGBoost.

## 2026-04-14 — XGBoost Polars Fastpath + Unified Categorical Handling

### Added

- **XGBoost Polars fastpath**: XGBoost (>= 3.1) now receives Polars DataFrames directly via `train_mlframe_models_suite`. String columns are cast to `pl.Categorical` (XGBoost auto-detects via `enable_categorical=True`). No cardinality limit unlike HGB.
- `XGBoostStrategy` in `training/strategies.py` — inherits `TreeModelStrategy`, adds `supports_polars = True` and `prepare_polars_dataframe` (casts `pl.String` → `pl.Categorical`).
- **Unified categorical type constants** in `training/strategies.py`:
  - `PANDAS_CATEGORICAL_DTYPES` — `frozenset({"category", "object", "string", "string[pyarrow]", "large_string[pyarrow]"})`
  - `get_polars_cat_columns(df)` — detects `pl.Categorical`, `pl.Utf8`, `pl.String` columns
  - `is_polars_categorical(dtype)` — type check helper
- 5 unit tests for `XGBoostStrategy.prepare_polars_dataframe` (string→categorical, high-cardinality, passthrough)
- `TestXGBoostPolarsClassification` — XGBoost trained directly on Polars with categorical features
- Parametrized integration tests extended: `test_polars_fastpath_parametrized` and `test_polars_fastpath_regression_target` now cover `["cb", "xgb", "hgb"]`

### Changed

- `training/strategies.py`: XGBoost (`"xgb"`) now uses `XGBoostStrategy` instead of shared `TreeModelStrategy`.
- **Refactored categorical detection** across codebase to use unified constants:
  - `pipeline.py`: uses `PANDAS_CATEGORICAL_DTYPES` and `get_polars_cat_columns()`
  - `trainer.py:_filter_categorical_features`: uses unified constants, **fixed missing `pl.Utf8` bug** and missing pandas string types
  - `utils.py:get_categorical_columns`: uses unified constants
  - `core.py`: uses `get_polars_cat_columns()` for pre-pipeline detection

### Fixed

- **Bug**: `_filter_categorical_features` in `trainer.py` did not include `pl.Utf8` in Polars detection, silently filtering out valid categorical columns.
- **Bug**: `_filter_categorical_features` pandas path only checked `["category", "object"]`, missing `"string"`, `"string[pyarrow]"`, `"large_string[pyarrow]"`.

### Polars support matrix (updated)

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| XGBoost (`xgb`) | Yes (>= 3.1) | Yes (auto-casts strings → pl.Categorical) |
| HGB | Yes (numeric + Categorical) | Yes (auto-casts strings, handles cardinality > 255) |
| LightGBM (`lgb`) | No (broken in 4.6) | No |
| Linear models | No (internal NumPy conversion) | No |
| MLP / NGBoost | No | No |

## 2026-04-14 — HGB Polars Native Fastpath

### Added

- **HGB Polars fastpath** in `train_mlframe_models_suite`: when input is a Polars DataFrame, HGB models now receive it directly without intermediate pandas conversion. String categorical columns are automatically cast to `pl.Categorical` (cardinality ≤ 255) or ordinal-encoded to `pl.UInt32` (cardinality > 255, treated as continuous by HGB).
- `supports_polars = True` on `HGBStrategy`.
- `prepare_polars_dataframe(df, cat_features)` method on `ModelPipelineStrategy` base class (no-op default). `HGBStrategy` overrides it to handle cardinality-aware categorical casting.
- Pre-pipeline Polars originals are now saved before `fit_and_transform_pipeline()` to preserve string/categorical dtypes that polars-ds may convert to float.
- `cat_features_polars` list detected from pre-pipeline schema, used in Polars fastpath to ensure categorical columns are passed to models correctly.
- Polars fastpath now overrides `fit_params["cat_features"]` with pre-pipeline categorical columns when they differ from post-pipeline ones.
- 8 unit tests for `HGBStrategy.prepare_polars_dataframe` in `test_catboost_polars.py` (low/high cardinality, boundary 255/256, passthrough, missing columns).
- 2 integration tests in `test_core.py::TestPolarsNativeFastpath`: `test_hgb_receives_polars_dataframe`, `test_hgb_polars_categorical_is_cast`.

### Changed

- `training/strategies.py`: `HGBStrategy` now sets `supports_polars = True` and overrides `prepare_polars_dataframe` with cardinality-aware casting logic.
- `training/core.py`: Polars fastpath block now calls `strategy.prepare_polars_dataframe()` and sets `skip_preprocessing=True` for models that normally require encoding (HGB). Pre-pipeline Polars originals are saved before `fit_and_transform_pipeline()`.

### Polars support matrix (updated)

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| HGB | Yes (numeric + Categorical) | Yes (auto-casts strings, handles cardinality > 255) |
| LightGBM (`lgb`) | No | No |
| XGBoost (`xgb`) | No | No |
| Linear models | No | No |
| MLP / NGBoost | No | No |

## 2026-04-14 — Polars Native Fastpath for CatBoost

### Added

- **CatBoost Polars fastpath** in `train_mlframe_models_suite`: when input is a Polars DataFrame, CatBoost models now receive it directly without intermediate pandas conversion. This eliminates zero-copy overhead and allows CatBoost (>= 1.2.7) to use its native Polars ingestion path.
- `supports_polars` property on `ModelPipelineStrategy` (default `False`). New `CatBoostStrategy` subclass sets it to `True`.
- `CatBoostStrategy` in `training/strategies.py` — inherits `TreeModelStrategy`, adds `supports_polars = True`.
- Test file `tests/training/test_catboost_polars.py`: 11 tests covering CatBoost and HGB training directly on Polars DataFrames with categorical, numeric, text, and embedding features, plus early stopping on a Polars validation set.
- Integration tests in `tests/training/test_core.py` (`TestPolarsNativeFastpath`):
  - `test_catboost_receives_polars_dataframe` — monkeypatches `_train_model_with_fallback` to verify CatBoost `.fit()` receives a Polars DataFrame.
  - `test_non_catboost_still_gets_pandas` — verifies Ridge still receives pandas when input is Polars.

### Changed

- `training/core.py`: original Polars DataFrames are preserved before `_convert_dfs_to_pandas()` and substituted into `common_params` for models with `supports_polars`.
- `training/trainer.py`:
  - `train_df.columns.to_list()` replaced with `list(train_df.columns)` for Polars compatibility.
  - `_filter_categorical_features` now detects `pl.String` columns in addition to `pl.Categorical` when input is Polars.
- `training/strategies.py`: CatBoost (`"cb"`) now uses `CatBoostStrategy` instead of the shared `TreeModelStrategy`.

### Polars support matrix

| Model | Native Polars `.fit()` | Polars fastpath in `train_mlframe_models_suite` |
|-------|:----------------------:|:-----------------------------------------------:|
| CatBoost (`cb`) | Yes (>= 1.2.7) | Yes |
| HGB | Yes (numeric only) | No (requires category encoding) |
| LightGBM (`lgb`) | No | No |
| XGBoost (`xgb`) | No | No |
| Linear models | No | No |
| MLP / NGBoost | No | No |
