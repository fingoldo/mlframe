# A10 — Examples, scripts & notebook DX audit (2026-06-04)

READ-ONLY developer-experience review of mlframe's example code: the standalone
scripts under `scripts/` (incl. `scripts/repros/`), the tutorial notebook
`docs/composite_targets_tutorial.ipynb`, the example doc
`docs/examples/composite_targets.md`, and example snippets in `README.md`.

Focus: do they still RUN against the current package? Verified every imported /
called mlframe symbol against the source (`python` 3.14.3 import-only smoke +
grep). Heavy training was NOT executed.

## Method & headline result

- AST-parsed every script: all 17 scripts + the notebook cells parse cleanly (no
  syntax breakage).
- Import-only smoke of every mlframe symbol the scripts use: **all resolve**
  except ONE — the notebook's `from mlframe.tests.training.shared import
  SimpleFeaturesAndTargetsExtractor` (there is no `mlframe.tests` package).
- The bench scripts and `tiny_rerank` bench are CURRENT: every `mlframe.training._callbacks`,
  `_slice_helpers`, `_data_helpers`, `mlframe.metrics.*`, `composite_discovery`,
  and `CompositeTargetDiscoveryConfig` symbol exists with the used names, and every
  config kwarg used (`tiny_rerank_n_jobs`, `tiny_screening_models`, `screening`,
  `mi_estimator`, `top_k_after_mi`, `eps_mi_gain`, `cross_target_ensemble_strategy`,
  `mi_sample_n`, `discovery_n_jobs`, etc.) is a real field on the pydantic config.
- The 10 repro scripts are one-off bug repros for bugs fixed on 2026-05-16
  ("audit-fixes-2026-05-16"). Regression coverage is already in `tests/training/`
  (fuzz suite parametrized over `enumerate_combos(...)` + dedicated
  `test_mrmr_polars_fe.py`). They are DELETE-CANDIDATES (or, where a unique
  invariant is not yet pinned, SHOULD-BE-TEST), and 8 of them hardcode a wrong repo
  path.
- **README.md quickstart + example blocks are badly stale**: the headline
  `train_mlframe_models_suite` example uses an API that does not exist, and 6 of 10
  feature-example imports `ImportError`.

## Per-file table

| File | Classification | What it does | Issues | Recommendation |
|---|---|---|---|---|
| `scripts/bench_slice_es_100k_kfold10.py` | CURRENT | n=100k KFold(10) slice-ES bench, CB/XGB/LGB, regr+drift+classif; ~17 metric kernels; resumable JSON to `benchmarks/slice_es_100k_kfold10.json` | None functional. Output dir/file documented in module docstring. | Keep. Optionally cross-reference from a `scripts/README`. |
| `scripts/bench_slice_es_synthetics.py` | CURRENT | Wave-1 synthetic slice-ES value-prop bench (S1–S4), 30 seeds, Wilcoxon | Docstring run-line hardcodes `D:/ProgramData/anaconda3/python.exe` (anaconda not on this host). | Keep; fix docstring run-line to `python scripts/...`. |
| `scripts/bench_slice_es_synthetics_v2.py` | CURRENT | Wave-2 overfit-prone slice-ES bench; JSON to `benchmarks/slice_es_wave2_bench.json` | None functional. | Keep. |
| `scripts/bench_slice_es_synthetics_v3.py` | CURRENT | Wave-3 heavy-tail / rare-class / CB bench; JSON to `benchmarks/slice_es_wave3_bench.json` | None functional. | Keep. |
| `scripts/bench_tiny_rerank_parallel.py` | CURRENT | Serial-vs-parallel wall-time of `_tiny_model_rerank` on n=200k synthetic; equivalence check | None functional. Uses `df.copy()` on a small synthetic (acceptable for a bench, not prod). | Keep. |
| `scripts/numba_coverage_report.py` | CURRENT | Stdlib-only cobertura XML parser → numba-disabled coverage JSON; CLI argparse | None. Portable, no hardcoded paths, no mlframe import. | Keep. |
| `scripts/run_meta_tests.py` | CURRENT | Pre-commit wrapper running `tests/test_meta/` with NUMBA_DISABLE_JIT=1 via `sys.executable` | None. Uses `sys.executable`, no hardcoded python. | Keep. |
| `scripts/run_numba_coverage.ps1` | STALE-API (env) | PowerShell numba-disabled coverage runner | Hardcodes `D:/ProgramData/anaconda3/python.exe` (line 55) — fails on this host (python=3.14.3 store build, no anaconda). | Replace with `python` / `py` / `$env:PYTHON`. |
| `scripts/run_numba_coverage.sh` | CURRENT | POSIX twin of the above | Uses bare `python` (line 37) — fine. | Keep. |
| `scripts/repros/repro_c0085_text_feature_float.py` | DELETE-CANDIDATE | Manual repro of c0085 (CB text_feature=float); monkeypatches `cb.Pool.__init__` | Hardcodes `D:/Upd/Programming/PythonCodeRepository/mlframe` sys.path (lines 9–10) — wrong repo root. Bug fixed 2026-05-16; covered by fuzz suite. | Delete (fuzz `c0085` combo + suite cover it). |
| `scripts/repros/repro_c0086.py` | DELETE-CANDIDATE | Manual repro c0086 (CB+MRMR+polars_enum+nulls) | Hardcodes `D:/Upd/...` sys.path (15–16). Fixed; fuzz-covered. | Delete. |
| `scripts/repros/repro_c0121_cat_nan.py` | DELETE-CANDIDATE | Manual repro c0121 (NaN in cat_feature); diag monkeypatches `cb.Pool` + `get_pandas_view_of_polars_df` | Hardcodes `D:/Upd/...` (8–9). Note: filename says c0121 but it actually runs combo `c0088_*` (mislabelled). Fixed; fuzz-covered. | Delete. |
| `scripts/repros/repro_mrmr_feature_name_mismatch.py` | DELETE-CANDIDATE | Repro MRMR+linear-multi feature-name mismatch | Hardcodes `D:/Upd/...` (13–14). Fixed; fuzz-covered. | Delete. |
| `scripts/repros/repro_mrmr_no_support.py` | DELETE-CANDIDATE | Repro MRMR fit leaves no `support_` | Hardcodes `D:/Upd/...` (15–16). Fixed; fuzz-covered. | Delete. |
| `scripts/repros/repro_mrmr_polars_native.py` | SHOULD-BE-TEST (already promoted) / DELETE-CANDIDATE | Asserts MRMR.fit/transform on polars makes ≤1 `to_pandas()` call + returns pl.DataFrame | Hardcodes `D:/Upd/...` (18). **Exact invariant already promoted** to `tests/training/test_mrmr_polars_fe.py:64–99` (to_pandas spy + isinstance assert). | Delete (superseded by a stronger test). |
| `scripts/repros/repro_mrmr_xgb_lgb.py` | DELETE-CANDIDATE | "Is c0098 still live?" repro (xgb+lgb polars_utf8) | Hardcodes `D:/Upd/...` (16–17). Fixed; fuzz-covered. | Delete. |
| `scripts/repros/repro_mrmr_xgb_lgb_v2.py` | DELETE-CANDIDATE | c0098 root-cause dig; monkeypatches `MRMR.fit` to print X state | Hardcodes `D:/Upd/...` (13–14). Pure diagnostic scratch. Fixed; fuzz-covered. | Delete. |
| `scripts/repros/repro_polars_utf8_dtypes_object.py` | DELETE-CANDIDATE (clean) | Repro polars_utf8 dtypes-object ValueError; FUZZ_SEED scoped + save/restore | **Portable** — uses `Path(__file__).resolve().parents[2]` (no hardcoded path), scopes env mutation. Best-written repro of the set. Bug fixed; fuzz-covered. | Delete (or keep as the canonical repro template if any are retained). |
| `scripts/repros/repro_schema_bug_linear_polars.py` | DELETE-CANDIDATE | Repro c0011 linear+polars gating bug | Hardcodes `D:/Upd/...` (15–16). Fixed; fuzz-covered. | Delete. |
| `docs/composite_targets_tutorial.ipynb` | STALE-API | 16-cell TVT composite-target tutorial | Cell-4 ImportError (`mlframe.tests.training.shared`); prerequisite hardcodes `pip install -e D:/Upd/...`; outputs are not committed but prose pins "Expected output"; no `--fast` path (3 full suite fits). | Fix import path + pip line; otherwise APIs are current. |
| `docs/examples/composite_targets.md` | CURRENT | Copy-paste recipe doc, 3 tiers + metadata reads + rollback | All imports/kwargs/symbols verified present (`report_to_markdown`, `CompositeSpec`, all config fields). One doc inconsistency: text says `mi_estimator` default is `"knn"` but the code default is `"bin"`. | Keep; fix the `mi_estimator` default note. |
| `docs/examples/README.md` | CURRENT | Index of example docs | Links resolve. | Keep. |
| `README.md` (quickstart + example blocks) | STALE-API | Marketing/quickstart snippets | Quickstart `train_mlframe_models_suite(...)` uses a fabricated signature + result-object API; 6/10 feature-example imports ImportError. | Rewrite snippets to the real API (see A10-09/A10-10). |
| `scripts/` (directory) | — | No `scripts/README` exists | Discoverability gap: bench output locations + "repros are dead weight" status only discoverable by reading each file. | Add a short `scripts/README.md`. |

## Detailed findings

### A10-01 — P1 — `docs/composite_targets_tutorial.ipynb` cell-4 — Notebook ImportErrors on cell 4 (`mlframe.tests.training.shared`)
**What's wrong:** Cell-4 (and the import block reused in cell-8) does
`from mlframe.tests.training.shared import SimpleFeaturesAndTargetsExtractor`.
There is no `mlframe.tests` package — tests live at repo-root `tests/`, not under
`src/mlframe/`. Import smoke: `ModuleNotFoundError: No module named 'mlframe.tests'`.
The notebook dies on the first executable cell, so the entire "16-cell walkthrough"
is non-runnable as shipped. `SimpleFeaturesAndTargetsExtractor` IS publicly importable
from `mlframe.training.extractors` (and `mlframe.training`).
**Concrete fix (update-API):** change cell-4/cell-8 import to
`from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor`.
**Confidence:** High.

### A10-02 — P2 — `docs/composite_targets_tutorial.ipynb` cell-0 (Prerequisites) — Hardcoded D: install path
**What's wrong:** Prerequisites cell instructs `pip install -e
D:/Upd/Programming/PythonCodeRepository/mlframe`. That path is not this repo
(`C:/Users/Admin/Machine learning/mlframe`) and is not portable for any other user.
**Concrete fix (update-API):** `pip install -e .` (run from the repo root) or
`pip install mlframe`.
**Confidence:** High.

### A10-03 — P2 — `docs/composite_targets_tutorial.ipynb` (whole) — No Fast/small-data path; stale committed-prose outputs
**What's wrong:** The notebook runs three full `train_mlframe_models_suite` passes
(baseline + composite + report) on n=1500. There is no `--fast`/`n` knob, and
cell-7 prose pins an "Expected output" (`delta% > 20`, `high_potential`) that is not
regenerated, so it silently rots when defaults shift (e.g. the doc-vs-code
`mi_estimator` drift in A10-08). The cells themselves use current APIs (verified:
`CompositeTargetDiscoveryConfig`, `composite_target_specs`, `composite_target_ensemble`,
`composite_target_y_scale_metrics`, `report_to_markdown`, `CompositeSpec`), so this is
a tutorial (not a scratch pad) — it just isn't CI-validated.
**Concrete fix (update-API / DOC):** parameterize `n` at the top; add a note that
"Expected output" is illustrative; ideally add the notebook to an nbval/execute CI
gate so cell-4's ImportError can't recur silently.
**Confidence:** High (runnability); Medium (CI recommendation).

### A10-04 — P2 — `scripts/run_numba_coverage.ps1:55` — Hardcoded anaconda python
**What's wrong:** Invokes `& "D:/ProgramData/anaconda3/python.exe" @pytestArgs`.
On this host `python`=3.14.3 store build with no anaconda; the script fails to find
the interpreter. The POSIX twin (`run_numba_coverage.sh:37`) correctly uses bare
`python`.
**Concrete fix (update-API):** use `python` / `py -3` / a `$env:PYTHON` override,
matching the `.sh` twin.
**Confidence:** High.

### A10-05 — P2 — `scripts/bench_slice_es_synthetics.py:6` — Docstring run-line hardcodes anaconda python
**What's wrong:** Module docstring says `Run from mlframe root:
D:/ProgramData/anaconda3/python.exe scripts/bench_slice_es_synthetics.py`. Doc-only
(the script itself uses `sys.path` + bare imports and runs fine under `python`), but
it tells a contributor on this host to use a non-existent interpreter.
**Concrete fix (DOC):** `python scripts/bench_slice_es_synthetics.py`.
**Confidence:** High.

### A10-06 — P1 — `scripts/repros/*` (8 of 10 files) — Repro scripts hardcode the wrong repo root
**What's wrong:** Eight repros prepend `r"D:/Upd/Programming/PythonCodeRepository/mlframe"`
(+ its `tests/training`) to `sys.path`:
`repro_c0085_text_feature_float.py:9-10`, `repro_c0086.py:15-16`,
`repro_c0121_cat_nan.py:8-9`, `repro_mrmr_feature_name_mismatch.py:13-14`,
`repro_mrmr_no_support.py:15-16`, `repro_mrmr_polars_native.py:18`,
`repro_mrmr_xgb_lgb.py:16-17`, `repro_mrmr_xgb_lgb_v2.py:13-14`,
`repro_schema_bug_linear_polars.py:15-16`. That directory is not this repo; on this
host the `_fuzz_combo`/`shared` imports come from whatever happens to be installed,
not the working tree. Only `repro_polars_utf8_dtypes_object.py` resolves the root
portably via `Path(__file__).resolve().parents[2]`.
**Concrete fix (delete — see A10-07; if any are kept, update-API):** delete the
repros; for any retained, replace the hardcoded `sys.path.insert` with the
`parents[2]` pattern from `repro_polars_utf8_dtypes_object.py`.
**Confidence:** High.

### A10-07 — P2 — `scripts/repros/*` (all 10) — Repros are dead weight for already-fixed bugs; superseded by the test suite
**What's wrong:** Every repro's header states the bug "passes after the
corresponding fix shipped on 2026-05-16" and "regression coverage now lives in
tests/training/". Verified: the fuzz suite (`tests/training/test_fuzz_suite.py`)
parametrizes over `enumerate_combos(target=150, master_seed=...)` and calls
`train_mlframe_models_suite` per combo — exactly the c0085/c0086/c0088/c0098/c0011
paths these repros walk. The polars-native MRMR invariant from
`repro_mrmr_polars_native.py` is promoted verbatim (with a stronger contract) to
`tests/training/test_mrmr_polars_fe.py:64-99`. Regression-sensor files
(`test_fuzz_regression_sensors.py`, `test_recent_fixes_regression_sensors.py`,
`test_prod_log_review_regression_sensors.py`) exist. The repros add no coverage the
suite lacks; they are diagnostic scratch (several still carry `cb.Pool` / `MRMR.fit`
monkeypatch instrumentation) and `repro_c0121_cat_nan.py` is even mislabelled (runs
`c0088_*`).
**Concrete fix (delete):** delete `scripts/repros/` wholesale. Before deletion,
spot-check that each c00XX combo id is reachable from
`enumerate_combos(target=150, ...)` (it is, per the fuzz suite) so no unique path is
lost. Per the project's "promote-to-test, then delete" + "no padding" memory, this
is a clean delete, not a promote — coverage already exists.
**Confidence:** High (supersession verified for the MRMR-native invariant and the
fuzz-combo paths); Medium (that NO repro carries a still-unpinned assertion — the
fuzz suite is a crash-gate, so a repro asserting a specific *numeric/structural*
invariant beyond "doesn't crash" would warrant promote-not-delete; none of these do
on inspection).

### A10-08 — P2 — `docs/examples/composite_targets.md:128-130` — Doc claims `mi_estimator` default is `"knn"`; code default is `"bin"`
**What's wrong:** The Tier-2 notes say `Stick with "knn" (default)` and
`mi_estimator="bin"` is "38x faster than the kNN Kraskov default". But the config's
real default is `mi_estimator: str = "bin"` (`_composite_target_discovery_config.py:309`,
flipped knn→bin 2026-05-10). The doc's parenthetical "(default)" on knn is wrong and
contradicts the Tier-1 prose elsewhere.
**Concrete fix (DOC):** correct the default-marker to `bin`; keep the heavy-tail
caveat ("set `mi_estimator='knn'` for power-law / heavy-tail targets").
**Confidence:** High.

### A10-09 — P1 — `README.md:100-116` — Quickstart uses a `train_mlframe_models_suite` signature + result API that does not exist
**What's wrong:** The headline quickstart calls
`train_mlframe_models_suite(df=df, target="y", models=[...], regression=False,
cv_folds=5, early_stopping_rounds=50, use_polars=True)` and then uses
`result.models["cb"].metrics["holdout_brier"]`, `result.models["lgb"].calibration_plot()`,
`result.ensemble("stack").predict_proba(...)`. None of these exist. The real
signature (verified `core/_main_train_suite.py:95`) requires positional
`target_name`, `model_name`, `features_and_targets_extractor` and uses
`mlframe_models=`; it returns a `(models, metadata)` **tuple**, not a result object
with `.models[...].metrics` / `.calibration_plot()` / `.ensemble(...)`. Parameters
`target=`, `models=`, `regression=`, `cv_folds=`, `use_polars=` are not accepted.
A new user copy-pasting the README's first code block gets an immediate `TypeError`.
**Concrete fix (update-API):** replace with the real shape (mirror the in-code
docstring example at `_main_train_suite.py:194-198` and the working
`docs/examples/composite_targets.md` Tier-0 block):
`models, metadata = train_mlframe_models_suite(df=df, target_name="y",
model_name="exp", features_and_targets_extractor=fte, mlframe_models=[...])`.
**Confidence:** High.

### A10-10 — P1 — `README.md:124,141,157-158` — 6 of 10 feature-example imports ImportError
**What's wrong:** Import-only smoke of the README feature snippets:
- `from mlframe.estimators.custom import CompositeTargetEstimator` → ImportError
  (class lives in `mlframe.training._composite_target_estimator`).
- `from mlframe.metrics import compute_calibration_report` → ImportError (no such symbol).
- `from mlframe.feature_selection.filters.mrmr import mrmr_classif` → ImportError
  (`MRMR` class exists; `mrmr_classif` function does not).
- `from mlframe.feature_selection.wrappers import RFECVCustom` → ImportError
  (real class is `RFECV` in `wrappers/_rfecv.py`).
- `from mlframe.calibration.post import select_best_calibrator` → ImportError.
- `from mlframe.inference.predict import batch_predict` → ImportError.
- `from mlframe.feature_engineering.financial import compute_market_features` → ImportError.
(Verified OK: `ShapProxiedFS`, `ParamOracle`, `create_aggregated_features`, `MRMR`.)
The modules exist but the named symbols are renamed/relocated/fictional, so these
snippets are illustrative-only, not runnable.
**Concrete fix (update-API):** map each example to a real symbol
(`RFECVCustom`→`RFECV`; `CompositeTargetEstimator` import from
`mlframe.training`; replace `mrmr_classif` with the `MRMR` estimator call; verify or
drop `compute_calibration_report` / `select_best_calibrator` / `batch_predict` /
`compute_market_features` against the actual public API). Add a README smoke test
(import-grep) so example imports stay honest.
**Confidence:** High.

### A10-11 — P2 — `scripts/` — No `scripts/README` (discoverability gap)
**What's wrong:** There is no `scripts/README.md` or `scripts/repros/README.md`.
A maintainer cannot tell, without opening each file, that (a) the `bench_slice_es_*`
trio is a 3-wave slice-ES investigation writing to `benchmarks/slice_es_*.json`,
(b) `bench_tiny_rerank_parallel.py` is a one-shot wall-time check, (c)
`numba_coverage_report.py` + `run_numba_coverage.{ps1,sh}` form a coupled
numba-coverage workflow, or (d) `scripts/repros/*` are dead one-off repros for fixed
bugs. The "cheap now beats expensive later" + discoverability ask applies.
**Concrete fix (add-README):** add a one-paragraph-per-script `scripts/README.md`
(what it measures, where output lands, runtime, whether it's CI or manual). If the
repros are deleted (A10-07), no repros README is needed.
**Confidence:** High (the file is genuinely absent).

### A10-12 — Low — `scripts/repros/repro_c0121_cat_nan.py:1,64-66` — Filename/combo-id mismatch
**What's wrong:** The file is named/docstringed `c0121` but selects combo
`c0088_2fa08bef` from `enumerate_combos(...)`. Misleading if anyone tries to map the
repro back to a fuzz id.
**Concrete fix (delete):** resolved by deleting the repro (A10-07); if kept, rename
or correct the comment.
**Confidence:** High.

## Disposition rollup

12 findings: 4 P1 (A10-01, A10-06, A10-09, A10-10), 7 P2
(A10-02/03/04/05/07/08/11), 1 Low (A10-12).
Classifications: notebook = STALE-API; bench scripts + `composite_targets.md` +
`numba_coverage_report.py` + `run_meta_tests.py` + `run_numba_coverage.sh` = CURRENT;
`run_numba_coverage.ps1` = STALE-API(env); all 10 repros = DELETE-CANDIDATE (one of
which, `repro_mrmr_polars_native.py`, is already promoted-to-test); README
quickstart/examples = STALE-API.
