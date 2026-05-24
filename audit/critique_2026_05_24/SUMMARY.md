# mlframe Critique Wave — Master Summary (2026-05-24)

## Контекст

11 параллельных критик-агентов прошли по `train_mlframe_models_suite` и сопутствующим системам (тесты, монолиты, ML/код-архитектура). Каждый агент создал детальный отчёт в этой директории; этот файл — мастер-таблица с дисposition по каждой находке.

**Verification**: P0 находки выборочно подтверждены через прямое чтение file:line (A3, A4, A5, A6, B1, B2, D2 — все CONFIRMED; A1 partially confirmed via docstring at l.213; полная сверка тел функций отложена до решения о фиксах). Никаких изменений в `src/` не сделано — режим аудита.

## Per-agent итоги

| Agent | Scope | File | P0 | P1 | P2 | Low | Total |
|-------|-------|------|----|----|----|----|-------|
| A1 | Feature selection | `fs-critique.md` | 1 | 4 | ~10 | ~6 | 21 |
| A2 | Feature engineering | `fe-critique.md` | 3 | 9 | 7 | 6 | 25 |
| A3 | Ensembling | `ensembling-critique.md` | 1 | 3 | 8 | 4 | 16 |
| A4 | Perf hotspots | `perf-hotspots-critique.md` | 1 | 7 | 10 | 6 | 24 |
| A5 | Pipeline caching | `pipeline-cache-critique.md` | 3 | 4 | 6 | 3 | 16 |
| A6 | Polars zero-copy | `polars-zerocopy-critique.md` | 8 | ~15 | ~12 | ~5 | 40 |
| B1 | Tests expand | `tests-expand.md` | 3 | ~10 | ~20 | ~17 | ~50 |
| B2 | Tests optimize | `tests-optimize.md` | 4 | 13 | 23 | 10 | 50 |
| C1 | Monoliths split | `monoliths-split.md` | — | — | — | — | 18 file plans + 7 preventive |
| D1 | ML best practices | `ml-best-practices.md` | 0 | 5 | 5 | 3 | 13 |
| D2 | Code/arch standards | `code-arch-standards.md` | 3 | 9 | 7 | 4 | 23 |
| **Σ** | | | **27** | **~79** | **~108** | **~64** | **~278 + 25 split plans** |

(Точные числа P2/Low для A1, A6, B1 — см. соответствующие файлы; tilde-числа из подсчёта по сводкам агентов.)

## P0 Master Table (с disposition)

Каждая P0 → блокирующая проблема (некорректность / leakage / silent corruption / deploy break). Default disposition: **FIX_NOW** если не указано иное.

| # | Src | file:line | Issue | Disposition | Comment |
|---|-----|-----------|-------|-------------|---------|
| S01 | A1 | `training/_pipeline_cache.py:213` | `_pre_pipeline_cache_key` partial cell-sampling of train_target (docstring claims defence-in-depth but cells-only fp can collide on balanced binary) | FIX_NOW | заменить на blake2b full-y по аналогии с `_mrmr_fingerprints._full_y_content_hash` |
| S02 | A2 | `feature_engineering/_filter_to_numeric` | `.copy()` violates 100GB rule | FIX_NOW | views/lazy вместо copy |
| S03 | A2 | PySR FE temp-target leak | mutates `train_df` with temp target, не восстанавливает | FIX_NOW | try/finally inject+remove |
| S04 | A2 | `_phase_composite_discovery.py` | `.copy(deep=False)` + `[target_col]=...` leaks через shared BlockManager | FIX_NOW | full copy or transform on view-aware path |
| S05 | A3 | `models/_ensembling_process_method.py:129-141` | `_oof_or_train` silent fallback to train_probs/train_preds on level-1 — leakage в "train" ensemble метрики | FIX_NOW | strict require oof when stacking; remove silent fallback on level-1 path |
| S06 | A4 | `training/_precompute.py:62-69` | pandas `get_trainset_features_stats` Python `for col: .unique()` loop; polars sibling already batches | FIX_NOW | direct copy-paste fix from polars variant |
| S07 | A5 | `training/_precompute.py:166-180` | `precompute_composite_target_specs` / `precompute_dummy_baselines` NotImplementedError stubs — единственный путь cross-run cache dead | FIX_NOW or DOCUMENT | либо реализовать через DiscoveryCache, либо удалить stubs + обновить docstring |
| S08 | A5 | `training/strategies.py:748-773` `PipelineCache` | plain dict, no size gate, нарушает CLAUDE.md 2GB threshold на 100GB frames | FIX_NOW | wrap with LRU + nbytes gate |
| S09 | A5 | `_phase_helpers_fit_pipeline.py:355-470` `fit_and_transform_pipeline` + `apply_preprocessing_extensions` | PySR/TF-IDF/poly/RBF/PCA recompute каждый run; zero joblib.Memory в training/ | FIX_LATER | архитектурный SuiteArtefactCache + KeyBuilder (см. предложение в A5) |
| S10 | A6 | `_main_train_suite.py:814-816` | `pd.DataFrame(<polars>)` bypass Arrow split-blocks bridge | FIX_NOW | route через `get_pandas_view_of_polars_df` |
| S11 | A6 | `_pipeline_helpers.py:401-402` | same downgrade pattern | FIX_NOW | same |
| S12 | A6 | `_pipeline_extensions.py:497` | same | FIX_NOW | same |
| S13 | A6 | `extractors.py:249,371` | same | FIX_NOW | same |
| S14-S17 | A6 | 4 ещё P0 hotspots — см. `polars-zerocopy-critique.md` | per-call `pl.from_pandas` без `rechunk=False`, per-column `.to_numpy()` loops | FIX_NOW batch | see source file for full list |
| S18 | B1 | `training/crash_reporting.py` | zero tests / Windows-signal+WER path uncovered | FIX_NOW | unit suite (Windows-only marker OK) |
| S19 | B1 | `training/metrics_registry.py` | zero direct unit coverage of register/unregister/iter API | FIX_NOW | unit + biz_value |
| S20 | B1 | `feature_selection/mi.py` | 3 separate MI implementations no cross-parity test | FIX_NOW | golden parity test |
| S21 | B2 | tests `importlib.reload` без snapshot (≥2 sites incl. `test_biz_val_filters_mrmr.py`, `test_automl.py`) | violates CLAUDE.md test pollution rule | FIX_NOW | snapshot/restore pattern + meta-test (D2 #9 закрывает регрессию) |
| S22 | B2 | `_clear_mrmr_fit_cache_between_tests` autouse over entire `feature_selection/` tree но релевантен 1 файлу | over-broad autouse | FIX_NOW | переместить в локальный conftest |
| S23 | B2 | duplicate `IS_FAST_MODE` snapshots в root + fs conftests | split source of truth | FIX_NOW | удалить дубль |
| S24 | B2 | `test_automl.py` 5 reloads / one file без isolation | high test-pollution risk | FIX_NOW | snapshot pattern или session-fresh fork |
| S25 | D2 | `tests/test_sklearn_compliance.py` (entire file) | `check_estimator` НЕ вызван для `CompositeTargetEstimator`, `_LagPredictDeployableModel`, `ESTransformedTargetRegressor`, `RFECV`, `PdOrdinalEncoder`, `PdKBinsDiscretizer` | FIX_NOW | `parametrize_with_checks` over всех wrappers + sklearn-matrix |
| S26 | D2 | `training/_composite_target_estimator.py:105-167+220` | `from_fitted_inner` создаёт instance с attrs не из `__init__` signature → `sklearn.base.clone()` silent data loss | FIX_NOW | document "do not clone" OR override `__sklearn_clone__` |
| S27 | D2 | `pyproject.toml:283` + `test_biz_val_pysr_fe_upgrade.py:17` | unregistered marker `no_xdist_parallel` (vs registered `no_xdist`) под `--strict-markers` | FIX_NOW | rename usage to `no_xdist` OR register |

**Σ P0 = 27**, of which **24 FIX_NOW**, **1 FIX_LATER** (S09 — architectural design), **1 FIX_NOW_or_DOCUMENT** (S07), **1 batch** (S14-S17 = 4 P0).

## P1 Master Table (с disposition)

P1 → измеримая perf/memory потеря, неверный ML API, security gap, deploy risk. Default disposition: **FIX_NOW** для security/correctness, **FIX_LATER** для perf-без-критичности.

### A1 Feature selection (4 P1)
| # | file:line | Issue | Disposition |
|---|-----------|-------|-------------|
| S28 | `feature_selection/general.py` `estimate_features_relevancy` | `.to_numpy(allow_copy=True).copy()` + `np.random.shuffle` на global RNG | FIX_NOW |
| S29 | `_rfecv_fit.py` `cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)` | silent 3 substitute для splitters без `n_splits` (2 guard sites) | FIX_NOW |
| S30 | MRMR `fit(groups=...)` accepted but silently ignored (только `warnings.warn`) | leakage hazard для panel/time-grouped data | FIX_NOW |
| S31 | `np.random.shuffle` global RNG в fleuret/permutation hot kernels | non-deterministic + parallel race | FIX_NOW |
| (+1) | `_rfecv_fit.py` time-series auto-detect silently swaps KFold→TimeSeriesSplit на `cv_shuffle=True` | overlap с D1 #2 | FIX_NOW |

### A2 Feature engineering (9 P1)
| # | Issue (см. fe-critique.md) | Disposition |
|---|----|----|
| S32 | `_DEFAULT_DATE_METHODS` mutable module-level default | FIX_NOW |
| S33 | missing year/week/quarter/is_weekend defaults + no tz handling в `create_date_features` | FIX_LATER |
| S34 | bruteforce `to_pandas()` + sub-frame fillna broadcast-copy на полный frame | FIX_NOW |
| S35 | `pd.DataFrame(list-of-lists, dtype=...)` slow path в timeseries | FIX_LATER |
| S36 | recursive subset copies в `create_aggregated_features` | FIX_LATER |
| S37 | `from_fitted_inner` skips `clone()` asymmetry vs `fit` | FIX_NOW |
| S38 | `_median_residual_fit` Python loop с np.median per bin | FIX_LATER (overlap с A4) |
| S39 | bruteforce rename mutating caller's frame | FIX_NOW |
| S40 | `_ratio_fit` empty-array np.median pitfall | FIX_NOW |

### A3 Ensembling (3 P1)
| # | file:line | Issue | Disposition |
|---|-----------|-------|-------------|
| S41 | `_phase_train_one_target.py:256-267` | `_ENSEMBLE_RANK_METRIC_CANDIDATES` fallback на `val.*`/`test.*` — selection-bias double-dip | FIX_NOW |
| S42 | classifier blend без Isotonic/Platt despite `mlframe.calibration.*` exists | FIX_NOW (overlap D1 #6) |
| S43 | `_rrf_aggregate_probs` для K=1 binary-as-1D возвращает raw RRF без sigmoid/minmax — destroys calibration для logloss/brier | FIX_NOW |

### A4 Perf hotspots (7 P1)
| # | file:line | Issue | Disposition |
|---|-----------|-------|-------------|
| S44 | `_phase_polars_fixes.py:172-234` | per-cat-column sync `.collect()` (sibling pattern @_precompute.py:131-134) | FIX_NOW |
| S45 | `composite_transforms.py:231-236` / `_composite_transforms_nonlinear.py:245-256` | N-bin Python mask loop в `_median_residual_fit`/`_quantile_residual_fit` | FIX_LATER |
| S46 | `composite_transforms.py:30-40` + `_composite_transforms_nonlinear.py:526-548` | `_ewma_kernel`/`_frac_diff_inverse_kernel` njit-only, нарушает CLAUDE.md ladder; missing batched parallel variant + kernel_tuning_cache | FIX_LATER |
| S47 | `_phase_train_one_target_body.py:597-614` | `_filter_polars_cat_features_by_dtype` invariant over weight loop — hoist | FIX_NOW |
| S48 | `_phase_composite_post.py:514-530` | train-pred cache keyed by `id(comp)` misses shim wrappers | FIX_NOW |
| S49 | `_phase_helpers.py:540-548` | `memory_usage(deep=True)` known multi-minute hot point; reuse polars `estimated_size()` уже в той же функции | FIX_NOW |
| S50 | (1 more — см. perf-hotspots-critique.md) | | FIX_LATER |

### A5 Pipeline caching (4 P1)
| # | Issue | Disposition |
|---|-------|-------------|
| S51 | preprocessing pipeline (scaler/imputer/encoder) re-fits на каждый target несмотря на детерминистичность по (df_fp, config) | FIX_LATER (architectural) |
| S52 | pre_screen drops не кэшируются между фазами | FIX_LATER |
| S53 | MRMR scores не shared cross-target когда target_y совпадает (текущий ключ верно учитывает y_hash) — gap скорее в "когда стоит шарить" | DOCUMENT |
| S54 | composite_discovery базовые признаки не кэшируются между разными compositions | FIX_LATER |

### A6 Polars zero-copy (~15 P1 — full list в polars-zerocopy-critique.md)
P1 = средний-размер df в горячем пути; AVOIDABLE + measurable cost. Default disposition: **FIX_NOW (batch)** — все ходим единым PR через bridge.

### B1 Tests expand (~10 P1)
P1 = биз-value gaps (composite_auto_detect U3, FS general U15, optbinning U17 87 LOC 0 tests, importance U16) + weak-assert clusters (W1-W8). Default: **FIX_NOW** для P1 биз-value, **FIX_LATER** для W weak asserts.

### B2 Tests optimize (13 P1)
P1 = no global `--timeout=` в addopts (S55 FIX_NOW), `@pytest.mark.fast` зарегистрирован но 0 применений (S56 FIX_NOW), ~60 per-file synthetic-data re-rolls (S57 FIX_LATER), RFECV+tqdmu monkey-patches papering over upstream defaults (S58 FIX_LATER — нужны upstream PRs), `_force_cpu_training_defaults` session-mutates (S59 FIX_NOW), weak `pytest.skip` swallows (S60 FIX_NOW per memory `feedback_dont_accept_documented_skips`), `_prewarm_numba_once` early-returns на serial runs defeats purpose (S61 FIX_NOW), `[MEM]` print spam (S62 FIX_LATER), autouse `_reset_global_rng_state` fights pytest-randomly (S63 FIX_NOW), + 4 more — see tests-optimize.md.

### D1 ML best practices (5 P1)
| # | file:line | Issue | Disposition |
|---|-----------|-------|-------------|
| S64 | `_phase_helpers_fit_split.py:485-501` | train+val Enum domain widening — biases val ES detector | FIX_NOW |
| S65 | `_rfecv_fit.py:283-294,204-208` | RFECV default cv=int — silent k-fold inside FS when outer suite uses temporal split | FIX_NOW (overlap S29) |
| S66 | `training/drift_report.py:35` + `feature_drift_report.py` | drift report numeric-only, no categorical PSI | FIX_NOW |
| S67 | `_phase_polars_fixes.py:69` + `preprocessing/transforms.py:112,123` + `core/predict.py:341` | `pl.Categorical` cast вместо `pl.Enum(train+val union)` — нарушает project rule | FIX_NOW |
| S68 | `training/evaluation.py` no bootstrap CI на primary metrics | FIX_LATER |

### D2 Code arch (9 P1)
| # | section | Issue | Disposition |
|---|---------|-------|-------------|
| S69 | `pyproject.toml:286-304` | 12 dead registered markers | FIX_NOW |
| S70 | `pyproject.toml:48-78` deps | ~90 deps без upper-bound caps | FIX_NOW |
| S71 | `src/mlframe/__init__.py:63-110` | public API stale vs CHANGELOG | FIX_LATER |
| S72 | `inference/predict.py:43-60` + `training/io.py:284-310` | pickle/joblib RCE — `_verify_sidecar` fail-open + bundle_sha256 unfilled | FIX_NOW (security) |
| S73 | `.pre-commit-config.yaml` | ruff/black/mypy/bandit `continue-on-error` — никогда не блокирует | FIX_NOW |
| S74 | meta-test absent | `del sys.modules` / `importlib.reload` без snapshot — AST scan needed (закрывает регрессию S21) | FIX_NOW |
| S75 | `ci.yml:23-33` | no macOS row | FIX_LATER |
| S76 | `sklearn-matrix-ci.yml:62` | sklearn matrix только py3.11/linux | FIX_LATER |
| S77 | `docs/` | docs не обновлены под Round 5.3/5.4 composite refactor | FIX_LATER |

**Σ P1 ≈ 79**, of which **~45 FIX_NOW**, **~30 FIX_LATER**, **~3 DOCUMENT**, **1 SECURITY priority** (S72).

## P2 / Low — aggregated disposition

Каждый P2/Low имеет individual row в source-файле агента. Default disposition по умолчанию:

| Severity | Default disposition | Override condition |
|----------|---------------------|-------------------|
| P2 | **FIX_LATER** (batch per-agent) | если linked to security/data-integrity → FIX_NOW |
| Low | **FIX_LATER** или **DOCUMENT** | если nice-to-have без leverage → REJECT с обоснованием |

Atomic enumeration P2/Low — см. source files (`tests-optimize.md`, `polars-zerocopy-critique.md`, etc.). Каждый агент следовал требованию emit ВСЕ findings включая Low (per memory `feedback_never_hide_low_findings`, `feedback_show_all_agent_findings`).

## Monoliths (C1)

18 файлов >900 LOC + 7 preventive. Per-file план в `monoliths-split.md`. Priority order (top-5):
1. `_phase_composite_post.py` (1129)
2. `composite_transforms.py` (1142) — partial done
3. `metrics/core.py` (1064)
4. `_setup_helpers.py` (1047)
5. `_target_distribution_analyzer.py` (1017)

Disposition: **FIX_LATER (sequential PRs)** — каждый split = свой PR с identity sensor + LOC budget. 4 tight-coupled (`neural/base.py`, `_composite_target_estimator.py`, `boruta_shap.py`, `baseline_diagnostics.py`) — отдельная очередь с method-rebinding pattern.

## Сross-cutting overlaps (одна баг — несколько агентов)

| Theme | Agents | Findings |
|-------|--------|----------|
| sklearn compliance gap | D2 #1, A2 (clone asymmetry), A1 (RFECV cv handling) | S25, S37, S29 — batch fix |
| polars Categorical vs Enum | D1 #4, A2 implicit | S67 (one batch) |
| OOF discipline / leakage | A3 #1, A5 #3 (cache key), A1 #1 (cache cell-sampling), D1 (provenance) | S05, S01, plus architectural provenance proposal |
| Pipeline cache absent | A5 #3, B2 (`PipelineCache` no size gate), A4 (PrePipelineCache) | S08, S09, S51 |
| Test pollution (sys.modules/reload) | B2 #1, D2 #9 | S21, S74 (meta-test закрывает регрессию) |
| `pd.DataFrame(<polars>)` bypass bridge | A6 ×8 P0 | S10-S17 batch |

## Verification status

Sample P0 verifications passed (file:line confirmed via direct Read/Grep):
- A3 S05: ✓ `_ensembling_process_method.py:129` `_oof_or_train` confirmed
- A4 S06: ✓ `_precompute.py:62-69` Python loop confirmed; sibling polars batch confirmed
- A5 S07: ✓ `_precompute.py:166-180` NotImplementedError docstring confirmed
- A6 S10: ✓ `_main_train_suite.py:814,816` `pd.DataFrame(_lb)` confirmed
- B1 S18: ✓ no `test_crash*` / `test_metrics_registry*` / `test_mi.py` in `tests/training/`
- B2 S21: ✓ 27 files match `importlib.reload`/`sys.modules` pattern в tests/
- D2 S25: ✓ только `tests/test_custom_estimators.py` use `check_estimator` (not the flagged classes)
- D2 S27: ✓ `no_xdist_parallel` встречается только в `test_biz_val_pysr_fe_upgrade.py:17`; не зарегистрирован

A1 S01 — partially confirmed (docstring at l.213 prevents naive cell-sampling claim, но точная сэмплинг-логика тела требует чтения; повторная сверка перед фиксом обязательна).

## Predлагаемый план фиксов (ожидает OK пользователя)

**Wave 1 — Critical correctness/leakage (immediate FIX_NOW)**:
S05 (ensembling leakage), S01 (cache fp), S02-S04 (FE copy/leak), S08 (pipeline cache size gate), S25-S27 (sklearn compliance + marker), S72 (pickle RCE)

**Wave 2 — Polars zero-copy batch**: S10-S17 (8 P0) + A6 P1 batch — единый PR через bridge

**Wave 3 — Test infrastructure**: S18-S24 (B1 P0 + B2 P0) + S55-S63 (B2 P1) + S74 (D2 meta-test)

**Wave 4 — Perf hotspots**: S06, S44, S47-S49 (immediate) + S45-S46 (numba parallel)

**Wave 5 — ML discipline**: S64-S67 (Enum, RFECV cv, drift PSI)

**Wave 6 — Caching architecture (S09, S51-S54)**: SuiteArtefactCache design + один-PR-на-кандидат

**Wave 7 — Monoliths**: 5 carve-friendly first (C1 priority list)

**Wave 8 — Documentation / housekeeping**: P2/Low batch per agent, S69-S71, S75-S77

Каждая Wave — отдельный PR с unit + biz_value + cProfile (per CLAUDE.md). Regression test per fix (per memory `feedback_test_every_bug_fix`).

## Output files

```
audit/critique_2026_05_24/
├── SUMMARY.md                       (this file)
├── fs-critique.md                   (A1)
├── fe-critique.md                   (A2)
├── ensembling-critique.md           (A3)
├── perf-hotspots-critique.md        (A4)
├── pipeline-cache-critique.md       (A5)
├── polars-zerocopy-critique.md      (A6)
├── tests-expand.md                  (B1)
├── tests-optimize.md                (B2)
├── monoliths-split.md               (C1)
├── ml-best-practices.md             (D1)
└── code-arch-standards.md           (D2)
```
