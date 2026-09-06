# Cross-cutting audit: symbol / API drift

**Date:** 2026-09-05
**Scope:** `src/mlframe`, `tests/`, `scripts/`, `benchmarks/`, `profiling/`, `research/`, `docs/`
**Modules parsed:** 6115 (`ast.parse`, no execution)
**Method:** purely **static** AST resolution. Two scanners were written in the scratch area:

1. Import resolver: every `from X import Y` (module-level *and* function-local, via `ast.walk`)
   resolved against a module map built from the file tree, with X's real top-level binding set
   (defs/classes/assignments/imports, including those nested in top-level `if`/`try`/`with`);
   plus `__all__` entries checked against the defining module.
2. Reference resolver: `patch(...)` / `patch.object(...)` / `setattr("...")` string targets and
   docstring/comment dotted paths (`:func:` roles, backticked dotted paths) resolved the same way.

No module was imported, and the test suite was not run.

**Conservatism:** modules that use `from ... import *` or a module-level `__getattr__` are skipped
(unresolvable statically). One dynamic pattern *was* resolved by hand: `_gpu_resident_fe.py` and
`_gpu_resident_select.py` rebind carved-out sibling names via
`for _n in dir(_m): globals()[_n] = getattr(_m, _n)`. This produced 18 apparent import failures;
all 18 names were confirmed to exist in the sibling carve files (`_gpu_resident_basis.py`,
`_gpu_resident_pair_mi.py`, `_gpu_resident_select.py`, `_gpu_resident_materialise.py`,
`_gpu_resident_discretize.py`) and are **not** findings.

**Scanner validation:** the known reference case
(`tests/feature_selection/benchmarks/test_fs_hybrid_run_experiment_auc_mean.py` importing
`compute_auc_mean`) is **already fixed in this tree** -- the function is present at
`src/mlframe/feature_selection/_benchmarks/fs_hybrid/run_experiment.py:64`. The scanner correctly
reports no finding there, and the same rule is what would have caught it while it was broken.

---

### XSD-01 [P1] linear-model-module-does-not-exist
**File:** src/mlframe/estimators/custom.py:154
**Summary:** `from ..linear_model import LinearRegression` resolves to `mlframe.linear_model`, which
does not exist. `src/mlframe/` has no `linear_model.py` and no `linear_model/` package (verified by
directory listing and by a repo-wide grep: this line is the *only* occurrence of
`mlframe.linear_model` anywhere in the tree; every other `linear_model` hit is either sklearn's or a
`linear_model_config` parameter name in `training/core/`).
**Failure scenario:** **call time, on a production path.** The import is function-local, inside the
`fit` body of the transformed-target regressor, in the `if self.regressor is None:` branch -- i.e. it
fires exactly when a caller constructs the estimator without passing an explicit `regressor` and
relies on the documented default. `ModuleNotFoundError: No module named 'mlframe.linear_model'` at
that point, after `_fit_transformer` / `_transform_y` have already run. It is invisible to
import-time checks and to any test that always passes a `regressor`.
**Evidence:** `src/mlframe/estimators/custom.py:154` reads
`            from ..linear_model import LinearRegression`; the enclosing block is
`if self.regressor is None:` (line 153) followed by `self.regressor_ = LinearRegression()` (line 155).
No `mlframe/linear_model*` file or directory exists.
**Suggested fix:** **re-point**, to `from sklearn.linear_model import LinearRegression`. The module
already imports `clone` from sklearn in the sibling branch, the class is used bare with no
mlframe-specific kwargs, and there is no evidence a `mlframe.linear_model` ever shipped (nothing else
references it), so "restore" has nothing to restore to. Deleting the default is not an option -- it
would turn a documented default into a required argument. This needs a regression test that
constructs the estimator with `regressor=None` and calls `fit`.

### XSD-02 [P1] stable-counting-segments-moved-to-grouped-segments
**File:** src/mlframe/feature_engineering/_benchmarks/bench_group_sort.py:10
**Summary:** imports `_stable_counting_segments_int` from `mlframe.feature_engineering.grouped`.
That name is defined in `mlframe.feature_engineering._grouped_segments` (line 38), and `grouped.py`
re-exports only `iter_group_segments` from that module (`grouped.py:67`), not the counting kernel.
**Failure scenario:** **import time**, whenever the bench module is imported or run
(`python -m mlframe.feature_engineering._benchmarks.bench_group_sort`). `ImportError: cannot import
name '_stable_counting_segments_int'`. This module ships inside `src/`, so it is on the package
import path, not merely a repo-root script -- any collector that walks package submodules (a future
"all submodules importable" test, a docs build, a `pkgutil.walk_packages` sweep) would hit it. It is
not currently imported by any test (checked), so it is not a collection break today.
**Evidence:** `grep -rn "_stable_counting_segments" src/mlframe/feature_engineering/` returns exactly
three source hits: the bench import at `bench_group_sort.py:10`, the `def` at `_grouped_segments.py:38`,
and a call site at `_grouped_segments.py:105`. `grouped.py:67` is
`from ._grouped_segments import iter_group_segments`.
**Suggested fix:** **re-point** the bench to
`from mlframe.feature_engineering._grouped_segments import _stable_counting_segments_int as _stable_counting_argsort_int`.
Do not widen `grouped.py`'s re-export surface to satisfy a bench -- `grouped_rank.py:5` documents a
deliberate policy that the segment kernels live in `_grouped_segments` and are imported from there
directly.

### XSD-03 [P1] raw-moments-renamed-to-per-cell-raw-moments-njit
**File:** profiling/bench_binned_numeric_agg_fold_gate.py:17
**Summary:** the import block pulls `_raw_moments` (one of six names) from
`mlframe.feature_selection.filters._binned_numeric_agg_fe`. No `_raw_moments` exists in that module,
or anywhere in `src/mlframe`. The nearest real symbol is `_per_cell_raw_moments_njit`
(`_binned_numeric_agg_fe.py:55`). The other names in the same import (`SUPPORTED_STATS`,
`_derive_cell_stats`, `_global_stat`, `engineered_name_binned_agg`) all resolve.
**Failure scenario:** **import time**, the moment the profiling script is run. `ImportError` on the
first statement of the script, so the fold-gate benchmark cannot be reproduced at all.
**Evidence:** `grep -rn "\b_raw_moments\b" src/mlframe --include=*.py`, excluding the `per_cell`
variant, returns nothing. `_binned_numeric_agg_fe.py` defines `_per_cell_count_sum_njit` (38),
`_per_cell_raw_moments_njit` (55), `_per_cell_centered_moments_njit` (81),
`_per_cell_moments_stable` (101).
**Suggested fix:** **re-point** to `_per_cell_raw_moments_njit`, then check the script's call site
against that signature `(codes, v, n_cells)`; the docstrings at lines 41 and 103 both cite
`_per_cell_raw_moments_njit` as the canonical name, so the bench simply missed a rename.

### XSD-04 [P2] eleven-bench-and-profile-scripts-import-pre-carve-training-modules
**File:** see the table below (11 distinct dead module paths, 20 import sites)
**Summary:** the `mlframe.training` package was reorganised into subpackages (`baselines/`,
`composite/`, `callbacks/`, `slicing/`, `targets/`), and a family of bench/profile scripts outside
`src/` still import the pre-carve flat module names. Every target module is gone; every *symbol* they
wanted still exists, at a new path:

| dead import | import sites | symbol still lives at |
|---|---|---|
| `mlframe.training.baseline_diagnostics` | benchmarks/composite_profile.py:82; benchmarks/profile_composite_new_code.py:79, :212 | `mlframe.training.baselines.diagnostics` (`BaselineDiagnostics`) |
| `mlframe.training.composite_diagnostics` | benchmarks/profile_composite_new_code.py:132, :211 | `mlframe.training.composite.diagnostics` (`plot_predictions_vs_actual`) |
| `mlframe.training._target_distribution_analyzer` | profiling/bench_feature_distribution.py:24 | `mlframe.training.targets._target_distribution_analyzer_features` (`analyze_feature_distribution`) |
| `mlframe.training.dummy_baselines` | profiling/bench_multiclass_bootstrap_logloss.py:24 | `mlframe.training.baselines._dummy_bootstrap` (`_vectorized_bootstrap_logloss_samples`) |
| `mlframe.training.composite_estimator` | profiling/bench_pack_g_watchdog_overhead.py:44 | `mlframe.training.composite.estimator._estimator` (`CompositeTargetEstimator`) |
| `mlframe.training.composite_transforms` | profiling/bench_pack_g_watchdog_overhead.py:45 | `mlframe.training.composite.transforms.naming` (`get_transform`) |
| `mlframe.training.composite_discovery` | profiling/bench_parallel_discovery_diag.py:50, :115; profiling/bench_parallel_discovery_speedup.py:43; profiling/bench_stacked_discovery_default_flip.py:104; scripts/bench_tiny_rerank_parallel.py:21 | `mlframe.training.composite.discovery` (`CompositeTargetDiscovery`, exported from its `__init__.py`) |
| `mlframe.training._callbacks` | scripts/bench_slice_es_100k_kfold10.py:61; scripts/bench_slice_es_synthetics.py:36; _v2.py:29; _v3.py:31, :225 | `mlframe.training.callbacks._callbacks` (`CatBoostCallback`, `LightGBMCallback`, `XGBoostCallback`) |
| `mlframe.training._slice_helpers` | scripts/bench_slice_es_100k_kfold10.py:67; scripts/bench_slice_es_synthetics.py:38; _v2.py:31; _v3.py:33 | `mlframe.training.slicing._slice_helpers` (`build_slice_eval_sets`) |
| `mlframe.tests.training.shared` | src/mlframe/training/_benchmarks/bench_val_size_default.py:39 | `tests/training/shared.py`, or `mlframe.training.extractors._extractors_simple` (`SimpleFeaturesAndTargetsExtractor`) -- note `mlframe.tests` is not a package at all |

**Failure scenario:** **import time, on the affected script only.** None of these files is under
`testpaths = ["tests"]` (pyproject.toml:516) except the last, which is under `src/` but is imported
by nothing, so there is no collection impact and CI stays green. The damage is that ten
reproducibility/regression benchmarks and one profiling harness are silently unrunnable: anyone
returning to re-measure a perf claim gets a `ModuleNotFoundError` instead of a number. Graded P2
rather than P1 because no shipped code path reaches them.
**Evidence:** each dead module path returns nothing from
`find src/mlframe -name "<name>.py" -o -type d -name "<name>"`; each replacement was located by
`grep -rln "^\(def\|class\) <Symbol>"` over `src/mlframe` and `tests`. `mlframe.tests` does not exist
under `src/mlframe/`.
**Suggested fix:** **re-point** all twenty import sites -- the mapping above is one-to-one and
mechanical, and every one of these scripts encodes a measurement that the repo's perf claims rest on,
so deleting them discards evidence. The `mlframe.tests.training.shared` case is the one exception:
that path never existed as an installed package, so `bench_val_size_default.py` should import the
extractor from `mlframe.training.extractors._extractors_simple` rather than from the test tree.
A cheap guard against recurrence: a test that `compile()`s (or AST-resolves, not imports) every
module under `scripts/`, `profiling/`, `benchmarks/`, and `src/**/_benchmarks/`.

### XSD-05 [P3] stale-prose-path-feature-selection-wrappers-rfecv
**File:** src/mlframe/feature_selection/wrappers/rfecv/_fit.py:1
**Summary:** the module docstring opens with "RFECV.fit carved out of
`mlframe.feature_selection.wrappers._rfecv`" -- that module no longer exists; the carve produced the
`wrappers/rfecv/` package this file lives in.
**Failure scenario:** **never fails.** Prose only. It misdirects a reader (and any doc
cross-reference tooling) to a path that cannot be opened.
**Evidence:** no `wrappers/_rfecv.py` on disk; `wrappers/rfecv/` is a package containing `_fit.py`.
**Suggested fix:** **re-point** the prose to `mlframe.feature_selection.wrappers.rfecv`, keeping the
"carved out of" history note -- that provenance is the sentence's whole value, so deleting it loses
more than the stale path costs.

### XSD-06 [P3] stale-prose-path-mlframe-signal-dtw-autotune
**File:** src/mlframe/signal/dtw.py:486
**Summary:** a comment reads "crossovers calibrated via `mlframe.signal._dtw_autotune`". No
`_dtw_autotune` module exists under `src/mlframe/signal/`; the string appears nowhere in the repo
except this one line.
**Failure scenario:** **never fails.** Prose only -- but this one is worse than a rename, because the
cited calibration source appears to be *gone*, not moved, so the provenance of the crossover
constants immediately below is now unverifiable.
**Evidence:** `grep -rln "_dtw_autotune" src/mlframe` returns only `src/mlframe/signal/dtw.py`.
**Suggested fix:** **restore or re-point** -- resolve where the autotune script went (git history for
a deleted `signal/_dtw_autotune*.py`) and cite that; if it was genuinely dropped, say so and record
the measured crossovers inline. Do not simply delete the reference: an unsourced magic crossover is a
worse artifact than a dangling one.

### XSD-07 [P3] stale-prose-path-training-core-short-model-tag
**File:** src/mlframe/training/_format.py:22
**Summary:** the docstring contains a `:func:` role pointing at
`mlframe.training.core._short_model_tag`. The function lives in this very file,
`mlframe.training._format`, not in `mlframe.training.core`.
**Failure scenario:** **never fails at runtime**, but a Sphinx build with `nitpicky` on would emit an
unresolved-reference warning, and the `:func:` role renders as a dead link.
**Evidence:** `grep -rln "_short_model_tag" src/mlframe` returns only
`src/mlframe/training/_format.py` (plus its `.pyc`). `mlframe.training.core` is a package and does
not define or re-export the name.
**Suggested fix:** **re-point** to `mlframe.training._format._short_model_tag`.

### XSD-08 [P3] stale-prose-path-training-utils-compute-config-signature-v1
**File:** tests/training/test_discovery_cache_version_tuple_expanded.py:20
**Summary:** a test docstring cites `mlframe.training.utils._compute_config_signature_v1`. That
symbol exists nowhere in `src/mlframe`.
**Failure scenario:** **never fails.** The test itself does not import the path (it patches by
object, not by string), so the drift is confined to the explanation of what the test intercepts --
which is precisely the part a future reader will trust when deciding whether the test still guards
anything.
**Evidence:** `grep -rln "_compute_config_signature_v1" src/mlframe` returns nothing.
**Suggested fix:** **re-point** to whatever the config-signature helper is called today (resolve from
the test's real patch target), or **delete** the sentence if the v1 signature scheme was removed
outright. Leaving it is the one option that keeps the test's stated rationale wrong.

---

## Checked and explicitly NOT findings

- **`_gpu_resident_fe` / `_gpu_resident_select` cluster (18 apparent hits).** Both modules rebind all
  carved-sibling names into their own namespace at the bottom of the file via
  `globals()[_n] = getattr(_m, _n)`. All 18 names (`gpu_discretize_codes_host`,
  `grand_fused_pair_mi`, `grand_fused_pair_mi_fused`, `_gpu_resident_discretize_codes`,
  `_searchsorted_codes`, `gpu_pairs_fe_mi`, `pair_candidate_mi_dispatch`, `_gpu_route_bases_batched`,
  `_gpu_evaluate_basis_matrix`, `_gpu_evaluate_basis_column`, `gpu_resident_pair_recipes`,
  `gpu_materialise_discretize_codes_host`, `_fe_materialise_block_gpu`, `_resident_operand_table`,
  `build_resident_operand_table`, `register_prebuilt_operand_table`,
  `fe_gpu_pairs_mi_backend_choice`, `fe_gpu_binning_backend_choice`) were individually located in a
  sibling carve file. The facade is intact.
- **tests/training/test_ensembling_caching_future_fixes.py:550** -- `from mlframe.training.strategies
  import get_cache_key` sits inside a `pytest.raises` block; the test asserts the name is *gone* and
  that it is absent from `__all__`. Working as intended.
- **tests/training/composite/cache/test_cache_store_identity.py:25** -- the import of
  `_old_cache_store_cpx28_baseline` is wrapped in `try/except Exception` with an `_HAVE_OLD` flag and
  a `# pragma: no cover - baseline snapshot absent` comment. Deliberate optional snapshot.
- **tests/feature_selection/_benchmarks/wide_data_scaling/test_progress_shared.py:12** -- the target
  `mlframe.feature_selection._benchmarks.wide_data_scaling` has no `__init__.py`, so my static map
  missed it, but it is an implicit namespace package inside a regular package and imports fine;
  `_progress_shared.py` is present. Not a defect. Sibling `fs_quality/` has the same shape, so this
  is the local convention, not an omission -- though neither directory would be picked up by a
  `find_packages()` wheel build, which is harmless since `_benchmarks` is not shipped.
- **pyproject.toml:479 console script** `mlframe-tune-kernels = "mlframe.system.kernel_tuning_cache:main"`
  -- `main` is defined at `src/mlframe/system/kernel_tuning_cache/__init__.py:24`. Resolves.
- **`__all__` entries.** Zero mismatches across all 6115 modules: every string in every statically
  analysable `__all__` is bound in its defining module.
- **`patch` / `monkeypatch.setattr` string targets.** 87 `mlframe.`-rooted string patch targets were
  resolved; **all 87 resolve** to a real module attribute. No silent no-op patches found.
- **Docstring/comment dotted paths** flagged as unresolvable but benign: `mlframe.competition.X` and
  `mlframe.training.core.X` (literal `X` placeholders in "import X from" prose),
  `mlframe.training.trainer.X` (same), `mlframe.training.automl.__dict__` (a dunder, not a
  submodule), and the `_old_cache_store_cpx28_baseline` prose mirroring the guarded optional import
  above.

## Coverage: what I reached and what I did not

**Covered.** (1) Every `from X import Y` in all 6115 parsed modules, module-level and function-local,
absolute and relative, across `src/mlframe`, `tests/`, `scripts/`, `benchmarks/`, `profiling/`,
`research/`. (2) All statically analysable `__all__` lists. (3) `patch` / `patch.object` / `setattr`
string targets. (4) Docstring and comment `:func:` / `:class:` / `:meth:` / `:mod:` roles and
backticked dotted paths beginning with `mlframe.`. (5) The single `[project.scripts]` entry point.

**Not reached.**

- **`import X` plus attribute-access drift.** I resolved `from X import Y` but not `mlframe.a.b.c(...)`
  attribute chains reached through a plain `import mlframe.a`. A function deleted but still *called*
  as `mod.fn()` would not appear here. This is the largest remaining gap for this bug class.
- **Modules using `from ... import *` or a module-level `__getattr__`** are skipped by design; the two
  `globals()`-facade modules were the only dynamic pattern I resolved by hand, so any *other*
  runtime-populated namespace is unverified.
- **Names created by `exec`, `setattr(module, ...)`, or decorator-registry side effects.**
- **`.md`, `.rst`, `.ipynb` and `mkdocs.yml` prose references** -- I scanned dotted paths only inside
  `.py` files. `docs/` markdown sources were not checked for stale API paths.
- **Third-party symbol drift** (sklearn / numba / cupy API renames). Only `mlframe.`- and
  `tests.`-rooted targets were resolved.
- **Class-attribute and method-level drift** (e.g. an attribute removed but referenced in a subclass).
- Nothing was imported and no test was executed, so **conditional import failures at runtime**
  (optional-dependency branches, `if TYPE_CHECKING` blocks that lie) are out of scope.

## Summary

| Severity | Count |
|---|---|
| P0 | 0 |
| P1 | 3 |
| P2 | 1 (covering 11 dead module paths / 20 import sites) |
| P3 | 4 |
| **Total findings** | **8** |

Raw scanner hits: 70 import/`__all__` plus 11 prose. 18 were the `_gpu_resident_*` dynamic facade,
2 were deliberate (negative test, guarded optional snapshot), 1 was a namespace-package blind spot in
my own module map, and 7 prose hits were placeholders or dunders -- all documented above rather than
dropped.
