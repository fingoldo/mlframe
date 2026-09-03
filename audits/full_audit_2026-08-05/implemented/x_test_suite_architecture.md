# Cross-cutting: test-suite architecture & coverage map -- mlframe audit (2026-08-05)

## Scope

`tests/` as a whole (~3200 `*.py` files), excluding `tests/feature_selection/mrmr/`, `tests/feature_selection/mrmr_api/`,
`tests/feature_selection/filters/`, `tests/feature_selection/shap_proxied/`, `tests/feature_selection/golden/` -- the
counterparts of the `src/mlframe/feature_selection/filters/**` and `.../shap_proxied_fs/**` trees, which the task's own
scope note excludes as already covered by a dedicated, closed MRMR audit cycle (`audits/mrmr_audit_2026-07-25/`).

This is the **second** run of this exact cluster: `audits/full_audit_2026-07-21/x_test_suite_architecture.md` covered
the same scope 15 days ago (10 findings, all CLOSED per `_TRACKER.md`, with fixes pinned by
`tests/test_meta/test_x_test_suite_architecture_fixes.py`). Per the task's own "do NOT attempt a file-by-file audit"
instruction, this pass verified every prior finding's fix is still in place (regression-checked, not re-derived from
scratch), then did a fresh structural sweep -- directory census, whole-tree AST/grep pattern scans (skip/xfail/broad-except/
`inspect.getsource`/weak-assertion/mutable-default-argument patterns), and targeted deep reads of newly-surfaced
candidates -- to find issues either introduced or missed since 2026-07-21.

Files read in full or substantial part this session:
- `tests/conftest.py` (887 lines, full) -- re-read for drift since the 2026-07-21 pass (verified `--run-heavy-automl` /
  F8's fix landed; found no new regressions in the root fixture set).
- `tests/test_meta/test_x_test_suite_architecture_fixes.py` (152 lines, full) -- the prior cycle's own regression file.
- `tests/test_meta/test_no_inspect_getsource.py` (235 lines, full) -- the `inspect.getsource`/source-position-proxy
  meta-linter and its whitelist.
- `tests/training/test_stress.py` (lines 440-490).
- `tests/training/test_splitting.py` (lines 1000-1390, four `except Exception: pass` sites around `assert`-bearing `try`
  blocks).
- `tests/feature_engineering/test_pysr_temp_target_restored.py` (lines 130-180).
- `tests/feature_selection/gpu/test_numba_utils_coverage.py` (lines 1-50).
- `tests/feature_selection/stability/test_selectors_shared.py` (lines 270-335).
- `tests/feature_selection/wrappers/test_wrappers_default_args.py` (lines 215-260).
- `tests/feature_engineering/transformer/test_biz_val_row_attention.py` (lines 70-90).
- `tests/feature_selection/biz_val/test_biz_val_e2e_suite_fs_breadth.py`,
  `test_biz_val_hetero_hybrid_determinism.py`, `test_biz_val_imbalanced_rare_class.py` (xfail blocks, ~20 lines each).
- `tests/training/test_feature_importances_multilabel_aggregation.py` (lines 1-100).
- `tests/training/neural/test_recurrent_keras_pickle.py` (lines 1-40).
- `tests/training/neural/test_multigpu.py` (lines 270-340).
- `tests/training/neural/test_integration.py` (lines 330-400).
- `tests/training/test_all_models.py` (lines 1140-1170).
- `tests/training/test_linear_models.py` (lines 145-170).
- `tests/feature_selection/stability/test_stability_transform_validation.py` (lines 60-90).
- `tests/integrations/test_mlflow.py` (function-name census via AST, all 17 `test_*` names read, 224 lines total).
- `src/mlframe/testing/parametric.py` (public-API census via AST) and `tests/training/test_parametric_robustness.py`
  (its sole consumer).

Whole-tree structural census (AST-walk / grep, not content-read of every file, per the task's explicit "systematic
architectural review" instruction): every `conftest.py` (10 files, line-counted and the root + `feature_selection/`
+ `training/` ones opened), every `pytest.mark.skip`/`skipif`/`xfail`/`importorskip`/`pytest.skip(` call site
(15 / 225 / 16 / 633 / 376 respectively), every broad `except Exception:`/bare-`except:` block whose body is exactly
`pass` (76 sites, individually triaged, MRMR/filters/shap_proxied-adjacent ones excluded from findings), every `try`
block containing a top-level `assert` whose exception handler is broad and does not re-raise (5 sites, all read and
triaged), every `inspect.getsource(` **call** (cross-checked programmatically against
`test_no_inspect_getsource.py`'s own `WHITELIST` -- 0 unlisted violations, confirmed clean), every sole-assertion
`test_*` function of the shape `assert X is not None` (151 sites, sampled across `training/`), every
`if hasattr(x, "attr"):` block (no `else`) containing an `assert` (21 sites, sampled), and every function/fixture
definition anywhere in `tests/` with a mutable (`list`/`dict`/`set`) default argument (0 hits -- clean).

Real files opened and read (fully or in a cited excerpt) this session: **22**. Real LOC directly viewed via those
reads: approximately **2000** lines. This is in addition to the whole-tree AST/grep census above, whose *existence,
naming, and pattern-match evidence* (not full content) was used for the remaining findings, consistent with the
"do not attempt a file-by-file read" scope instruction for a >3000-file tree.

## Findings

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| X_TEST_SUITE_ARCHITECTURE-1 | P1 | `tests/training/test_stress.py:449-484` | `test_many_nan_values`'s only assertion (`assert TargetTypes.REGRESSION in models`) sits inside a `try` block whose `except Exception: pass` also swallows `AssertionError` (a subclass of `Exception`) -- the test passes unconditionally regardless of whether `train_mlframe_models_suite` returns correct output, raises, or the assertion itself fails. | Move the assertion outside the `try`, or narrow the `except` to the specific expected failure modes (e.g. `except (ValueError, np.linalg.LinAlgError)`), or split into two tests: one asserting success, one asserting the documented failure mode with `pytest.raises`. | AST scanner: flag any `try` block containing a top-level `ast.Assert` whose matching `except` handler catches `Exception`/`BaseException` (or bare `except:`) and does not `raise`/re-raise -- this exact shape guarantees the assertion's failure is unreachable. (3 more instances of the same shape found this session, see -2/-3 below; a `tests/test_meta/` ratchet catching all of them at once is proposed once, not per-file.) |
| X_TEST_SUITE_ARCHITECTURE-2 | P1 | `tests/training/neural/test_integration.py:376-390` (`test_best_weights_restoration`), `tests/training/neural/test_multigpu.py:318-331` (`test_load_best_weights_in_ddp`) | Both tests explicitly set `load_best_weights_on_train_end=True` to exercise best-weight restoration after early stopping, but the entire assertion body is `if hasattr(clf.model, "best_epoch"): assert clf.model.best_epoch is not None`. If the trained model lacks the `best_epoch` attribute the `if` body never runs and the test passes with zero assertions executed; even when present, "is not None" proves nothing about whether the model's WEIGHTS were actually rolled back to the best checkpoint (vs. left at the final epoch's weights) -- the one behavior the test name and the explicit config flag claim to verify. | Compare model state (a hash of `state_dict()`/weights, or held-out-loss) at the best epoch against the post-restoration state, or at minimum assert equality/closeness to a snapshot taken at the best epoch during a spied callback, and drop the `hasattr` guard (fail loudly if the attribute is genuinely absent for the given estimator config, since `load_best_weights_on_train_end=True` was explicitly requested). | AST scanner: flag `test_*` functions whose ONLY assertion(s) are nested inside an `if hasattr(x, "...")` block with no `else` branch -- same class of "may execute zero assertions" test as F7 in the 2026-07-21 report (stopfile-callback `assert cb is not None`), now recurring in a different subsystem. Generalize `test_no_inspect_getsource.py`'s whitelist-scan pattern into a new `test_meta/test_no_vacuous_hasattr_guarded_assert.py`. |
| X_TEST_SUITE_ARCHITECTURE-3 | P2 | `tests/feature_selection/gpu/test_numba_utils_coverage.py:39-46` (`test_arr2str_uint8`) | `assert isinstance(out, str)` sits inside a `try` whose `except Exception: pytest.skip("uint8 not supported by arr2str dispatch")` converts an assertion failure (wrong return type -- a real bug) into a misleadingly-worded SKIP rather than a FAIL, indistinguishable in the pytest summary from a genuine "uint8 unsupported" environment gap. | Call `arr2str` unconditionally and assert the type outside any `try`; if uint8 is a genuinely-unsupported dtype, use `pytest.importorskip`/an explicit capability probe (e.g. try the dispatch on a throwaway value first, decide skip-vs-run BEFORE the real assertion) rather than wrapping the assertion itself. | Same AST scanner as -1: `try` block with a top-level `assert` and a broad `except` whose body calls `pytest.skip(...)` (not just `pass`) is the same failure-swallowing shape, just routed through skip instead of pass. |
| X_TEST_SUITE_ARCHITECTURE-4 | P2 | `tests/feature_selection/stability/test_selectors_shared.py:313-328` (`test_works_in_sklearn_pipeline`) | `assert preds.shape == y.shape` (the test's real invariant -- a fitted selector chained in an sklearn `Pipeline` must produce correctly-shaped predictions) is inside a `try` whose `except Exception as exc: pytest.skip(f"selector not sklearn-Pipeline-compatible: {exc}")` reports a shape-assertion failure as a compatibility skip rather than a test failure. | Split: probe pipeline construction/fit for compatibility (skip only on that), then assert the shape invariant unconditionally once `fit`/`predict` succeeded. | Same AST scanner as -1/-3. |
| X_TEST_SUITE_ARCHITECTURE-5 | P2 | `src/mlframe/testing/parametric.py` (10 public generators: `adversarial_frame`, `prod_like_frame`, `prod_like_frame_small`, `constant_column`, `categorical_column`, `id_column`, `high_card_text_column`, `inf_heavy_float_column`, `sparse_null_column`, `register_profiles`) | A first-class, shipped-with-the-package module purpose-built for exactly the "edge cases and robustness" scenarios this audit dimension asks about (constant/high-cardinality/inf-heavy/sparse-null columns) is imported by exactly **one** test file (`tests/training/test_parametric_robustness.py`) across the whole ~3200-file tree. Meanwhile 55 files independently define their own `..._constant_column` test scenarios, 18 their own categorical-column cases, 7 their own id-column cases -- confirmed via grep that these are locally-authored test names/fixtures, not calls into the shared module. | Point new/rewritten edge-case tests at `mlframe.testing.parametric`'s generators instead of hand-rolled inline frames; a good first target is the `training/` subtree's own edge-case tests (NaN/constant/high-cardinality), which duplicate the exact scenarios the module already covers. Not a mass rewrite -- a policy note + spot-conversion of the largest duplicative cluster. | A `tests/test_meta/` ratchet counting distinct local "constant column" / "categorical column" fixture-construction call sites (e.g. `pd.Series([x]*n)`-shaped assigns near a `def test_*constant*` function) vs. imports of `mlframe.testing.parametric`, alerting if the ratio doesn't improve release over release -- softer than a hard-fail gate, since not every scenario maps 1:1 onto the shared generators. |
| X_TEST_SUITE_ARCHITECTURE-6 | P2 | `src/mlframe/system/kernel_tuning_cache/__init__.py:22-180` (`main`, `cmd_list`, `cmd_show`, `cmd_explain`, `cmd_refresh`, `cmd_refresh_all`, `cmd_clear`) | Still zero test coverage for the `mlframe-tune-kernels` console-script entry point, 15 days after the 2026-07-21 audit's F6 flagged this exact gap. The fix-tracking file (`test_x_test_suite_architecture_fixes.py`, docstring) explicitly deferred F6 as "a concrete follow-up" needing its own pass rather than a few-line fix -- re-confirmed via grep this session that no test imports `main`/`cmd_*` from `kernel_tuning_cache`, nor invokes `mlframe-tune-kernels` as a subprocess. | Add a smoke test invoking `main(["list"])` / `main(["show", "<kernel>"])` / `main(["explain", ...])` against `capsys`, per the original PR4 proposal -- unchanged recommendation, now overdue. | N/A (this is itself the coverage-gap meta-test the original audit already proposed as PR4; tracking its continued absence is the "meta-test", i.e. a `test_meta/` sensor asserting `main` has >=1 real call site under `tests/` would catch this class of "CLI entry point ships with zero test" gap generically for any `def main(argv...)` found in `src/mlframe/**/__init__.py` or `src/mlframe/**/cli*.py`). |
| X_TEST_SUITE_ARCHITECTURE-7 | P3 | `tests/test_meta/test_x_test_suite_architecture_fixes.py` (whole file, whitelisted in `test_no_inspect_getsource.py`) | This regression file for the prior cycle's F1-F3/F7/F8 fixes is itself built almost entirely on `inspect.getsource()` string-pattern assertions (e.g. `assert "pytest.skip" not in src`, `assert "except (TypeError, ImportError)" not in src`) -- precisely the anti-pattern `feedback_behavioral_tests` (and this repo's own `test_no_inspect_getsource.py` gate) prohibits for every OTHER test file. It is explicitly grandfathered into the meta-linter's `WHITELIST` rather than converted, alongside ~27 sibling `*_fixes.py` files documented as "known debt." | Low priority given the explicit, tracked whitelist status (not a silent violation), but worth eventually converting F1-F3's checks to a real behavioral probe: monkeypatch `train_mlframe_models_suite` to raise a distinctive exception and assert the test functions propagate it (fail) rather than skip -- proves the fix behaviorally instead of by string absence, and would catch a future rewrite that reintroduces the same skip-on-import-failure behavior under different source text (e.g. a helper function renamed, or the `except`/`pytest.skip` tokens split across an f-string/`getattr` indirection that dodges the literal substring check). | N/A -- already tracked by the existing `test_no_inspect_getsource.py` whitelist/debt-list mechanism; flagged here only because this specific file is the one whose whole *purpose* is regression-proofing this cluster's own prior findings, so its own fragility is directly in-scope. |

## Dimensions with no findings this session

- **Mutable-default-argument bugs**: 0 hits across the whole `tests/` tree (AST-walked every function/fixture
  definition's `list`/`dict`/`set` defaults). Clean.
- **Flaky-test markers / retry decorators**: 0 uses of `pytest.mark.flaky` or `reruns=` anywhere in scope -- no
  flaky-marker misuse to report (the suite instead uses `perf_time_budget`/`perf_speedup_floor` xdist-aware budget
  scaling in `conftest.py`, a stronger pattern than blind reruns).
- **`inspect.getsource()` source-text-proxy antipattern**: the existing `test_no_inspect_getsource.py` +
  `test_no_source_text_position_proxy_in_test_files` meta-linters are accurate and current -- a from-scratch AST
  scan cross-checked against their `WHITELIST` found 0 unlisted violations.
- **xfail inventory** (16 sites): all are `strict=False`, individually and substantively documented "aspirational
  gap" markers on `biz_val`/business-value tests (e.g. row-attention lift, RFECV pruning under severe imbalance,
  HybridSelector column-order invariance) -- this is the repo's own documented positive pattern (xfail that still
  runs and surfaces in output, vs. a hard skip), not the "xfail to defer a framework bug" antipattern the project's
  memory explicitly warns against. No findings here.
- **Prior cycle's F1-F4/F7/F8 fixes**: spot-verified still in place (root `conftest.py`'s `--run-heavy-automl` flag +
  `heavy_automl` marker registered per F8; F5's mlflow test gap closed with 17 real behavioral tests in
  `tests/integrations/test_mlflow.py`). No regressions found in the previously-fixed set.

## Coverage notes

- Per the task's scope contract, `tests/feature_selection/mrmr/`, `mrmr_api/`, `filters/`, `shap_proxied/`, and
  `golden/` were excluded from both the broad-except and weak-assertion census passes; several structurally similar
  findings almost certainly exist there too (the same "assert-inside-broad-except" AST scan, run without the
  exclusion filter, surfaced additional hits in `feature_selection/mrmr/`-prefixed paths that were discarded
  unread) -- these are out of this cluster's scope by the task's own instruction, not overlooked.
- The 151-site "sole assertion is `X is not None`" census and the 21-site "`if hasattr(): assert`" census were
  sampled, not exhaustively read, in `training/` (the largest, least-previously-sampled subtree per the 2026-07-21
  report's own coverage notes). The two confirmed real bugs (X_TEST_SUITE_ARCHITECTURE-2) came from this sample;
  the remainder of both lists (most of `training/test_all_models.py`, `test_linear_models.py`, and
  `feature_selection/stability/test_stability_transform_validation.py`'s hits) were read and found to be
  legitimately conditional on genuine API-shape differences (e.g. `.coef_` only exists for linear models,
  `.columns` only exists for DataFrame-shaped transform output) rather than vacuous guards -- not every hit in
  these two AST-scan result sets is a bug, and a full manual triage of all 172 combined sites was not attempted at
  this effort level.
- F9 (499 filler docstrings from the `tests/interrogate.toml` 100%-coverage campaign) and F10 (5 documented "stale
  build path" skips) from the 2026-07-21 report were both explicitly assessed and deferred in that cycle's own fix
  file with stated reasoning (doc-quality nit with no reported bug; already-diagnosed environment quirk) -- not
  re-litigated here since nothing has changed about either since 2026-07-21.
- Did not run pytest or any other test-executing command (read-only audit); all claims are static-analysis-derived
  (AST walks, grep, direct file reads), consistent with the 2026-07-21 cycle's same constraint.
