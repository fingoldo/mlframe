# test_gaps (cross-cutting: what additional tests should exist) — mrmr_audit_2026-09-14

## Scope

Both sides of the question, read-only:

- **Source surface**: 77 files / ~31 440 LOC under `find src/mlframe -ipath "*mrmr*" -name "*.py"`.
  Largest: `filters/mrmr/_mrmr_class.py` (4285), `filters/_mrmr_fit_impl/_fit_impl_core.py` (2543),
  `filters/_mrmr_fe_step/_step_score.py` (1189), `_step_core.py` (1060).
- **Test surface**: 362 files under `find tests -ipath "*mrmr*"`
  (`biz_val` 134, `core` 88, `fe` 27, `caching` 10, `regression` 9, plus ~20 loose `test_audit_*`),
  plus MRMR-touching tests outside that tree (`tests/feature_selection/{contracts,filters,gpu,discretization,
  clustering,screening,info_theory,mrmr_api,regression}/`, `tests/test_meta/`).
- **Prior waves cross-checked**: `audits/mrmr_audit_2026-07-20/_TRACKER.md`, `-07-22/_TRACKER.md`,
  `-07-25/_TRACKER.md`.

**Headline calibration, stated up front so this doc is not read as alarmist**: the MRMR suite is unusually
strong for a module this size. The prior-wave cross-check (section 3) came back **almost entirely clean** —
my first pass, grepping for the *helper name* named in each tracker row, produced false gaps that dissolved
once I grepped for the *behaviour* instead. I record that explicitly because the brief warns that a wrong
finding costs more than a missing one. The genuine findings below are narrower and mostly concentrated in
(a) three live parameter values with literally zero tests anywhere, (b) a handful of assertions that are
vacuous or near-vacuous, and (c) parity axes (polars-vs-pandas selection equivalence, float32 **X**) that the
suite covers on one side only.

Severity per `_BRIEF.md`. Tag prefix `TESTGAP`.

---

## 1. Untested invariants on the public contract

### TESTGAP-1 — `nbins_method="marx"` / `"mah_sci"` / `"sci"` have ZERO tests anywhere in the repo, and the edges layer collapses all three to `"mah"` [P1]

**Where:** `src/mlframe/feature_selection/filters/_adaptive_nbins.py:597-600`

```
    "mah": "mah",
    "mah_sci": "mah",
    "sci": "mah",
    "marx": "mah",
```

These four strings are accepted `nbins_method` values (`filters/mrmr/_mrmr_param_constants.py:30` and `:47`,
`_fit_impl_core.py:1270`, `discretization/_discretization_dataset.py:320`). The constants file annotates
`"marx"` as `# Marx 2021 SCI-guided adaptive` — i.e. the docstring claims a *distinct estimator*.

**What the grep says:** `grep -rl "mah_sci|'marx'|\"marx\"" tests` returns **0 files**. Not one test, not one
biz_value test, not one parity test. Every other `nbins_method` value has coverage (`sturges` 7 files,
`freedman_diaconis` 8, `knuth` 5, `mdlp` 21, `fayyad_irani` 6, `optimal_joint` 3, `blocks` 1).

**Why today's suite would miss the regression:** because there is nothing to regress *from*. A user selecting
`nbins_method="marx"` on the strength of the "Marx 2021 SCI-guided adaptive" comment gets, at the
`per_feature_edges` layer, byte-identical behaviour to plain `"mah"`. Either (i) the SCI-guided variation lives
further up (the `entropy_estimator` axis) and the alias table is correct but undocumented, or (ii) the
advertised mechanism is a silent no-op. **I could not settle which from a read — flagged `unverified` on the
root cause, but the zero-test fact is verified.** What settles it: fit twice, `nbins_method="mah"` vs `"marx"`,
on a column whose SCI-optimal binning provably differs from MAH's, and compare the emitted edges.

**Proposed tests:**

| Test | File | Assertion |
|---|---|---|
| `test_nbins_method_marx_is_not_a_silent_alias_of_mah` | `tests/feature_selection/discretization/test_adaptive_nbins_estimator_aliases.py` (new) | Build a column where SCI and MAH provably disagree; `per_feature_edges(X, y, method="marx")` vs `method="mah"` — assert `not np.array_equal(edges_marx[0], edges_mah[0])`. **If they ARE equal, that is the bug**, and the test is then re-framed to pin the alias explicitly plus a docstring fix at `_mrmr_param_constants.py:30`. |
| `test_nbins_method_sci_produces_finite_monotone_edges` | same file | For each of `("mah", "mah_sci", "sci", "marx")`: `e = per_feature_edges(X, y, method=m)[0]`; assert `np.all(np.diff(e) > 0)`, `np.isfinite(e).all()`, `2 <= e.size <= 257`. |
| `test_biz_val_filters_mrmr_nbins_marx_beats_uniform_on_multimodal` | `tests/feature_selection/mrmr/biz_val/test_biz_val_filters_mrmr_nbins_marx.py` (new) | Synthetic: 3-component Gaussian mixture feature carrying the label, plus 12 noise cols. Baseline `nbins_method="uniform"` at the same `n_bins`. Assert `mi_marx >= 1.10 * mi_uniform` and the signal column is rank-1 in `mrmr_gains_`. Threshold set 5-15 % under the measured value once measured. <5 s at n=4000. |

**Regression that slips past today:** any edit to `_ADAPTIVE_NBINS_ALIASES` — a typo, a dropped key, an alias
repointed at the wrong estimator — is invisible. So is a `ValueError` raised at fit time for these three values.

---

### TESTGAP-2 — no polars-vs-pandas **selection-equivalence** test: polars is tested for *acceptance*, never for *identical `support_`* [P2]

**Where (tests):** polars coverage in the MRMR tree is `tests/feature_selection/mrmr/core/test_mrmr_basic.py:498`
(`test_polars_input_with_pandas_output_preserves_dtypes`), `test_mrmr_edges_coverage.py:140,150,234`,
`test_mrmr_dedup_fegate_adversarial.py:264,277,294,305`, `test_mrmr_input_not_mutated.py:145`,
`test_mrmr_polars_transform_chained_recipe.py:79,108`.

**What:** every one of these asserts polars input is *accepted*, *not mutated*, *auto-collected*, or that
dtypes/recipe names survive. **None** fits the same data twice — once as `pd.DataFrame`, once as the exact
`pl.DataFrame` equivalent — and asserts the two `support_` masks are identical.

**Why it matters here specifically:** `CLAUDE.md` documents that `MRMR.fit` bridges polars → an Arrow-backed
pandas view (`get_pandas_view_of_polars_df`) whenever FE runs, and that `_fe_frame_ops` is a *separate*
format-agnostic fallback seam. Two code paths, one contract. Prior wave `STABILITY_MISC-1`
(`group_aware.py:403-406`, DONE) was **exactly this bug class**: a polars→pandas bridge that collapsed numeric
columns to object dtype and destroyed factorize ordering, changing the clustering and therefore the support.
That fix is pinned, but only for `group_aware` — the general fit-level parity invariant is not.

**Proposed tests:**

| Test | File | Assertion |
|---|---|---|
| `test_polars_and_pandas_fits_select_identical_support` | `tests/feature_selection/mrmr/core/test_mrmr_polars_pandas_selection_parity.py` (new) | `np.array_equal(m_pd.support_, m_pl.support_)` AND `list(m_pd.get_feature_names_out()) == list(m_pl.get_feature_names_out())` AND `np.allclose(m_pd.mrmr_gains_, m_pl.mrmr_gains_, rtol=1e-12)`. Pinned `random_state`. |
| `test_polars_and_pandas_fits_identical_with_fe_enabled` | same | Same, with `fe_hybrid_orth_enable=True` so the Arrow-bridge path (not the fallback seam) is the one exercised; assert equal `_engineered_recipes_` names in order. |
| `test_polars_and_pandas_transform_emit_identical_values` | same | `np.allclose(m_pd.transform(X_pd).to_numpy(), m_pl.transform(X_pl).to_numpy(), rtol=1e-12, equal_nan=True)`. |

**Reuse:** the polars fixtures in `test_mrmr_polars_transform_chained_recipe.py` already build a paired
frame — reuse rather than rebuild.

**Regression that slips past today:** a dtype-collapsing or column-reordering change in the Arrow bridge
selects a *different but still valid-looking* feature set on polars input. Every current polars test passes.

---

### TESTGAP-3 — float32 parity is pinned for `y` but not for `X` [P2]

**Where:** `tests/.../test_biz_value_mrmr_contracts_robustness/test_continuous_regression.py:461`
`test_float32_y_matches_float64_support` — the only float32 selection-parity test.
`tests/feature_selection/mrmr/core/test_mrmr_partial_fit_dtype_mismatch.py:22,47` cover float32→float64
*batch upcasting* inside `partial_fit`, not fit-level `X` dtype parity.

**What:** no test fits `X.astype(np.float32)` against `X.astype(np.float64)` and compares `support_`.
`X` is where the discretization, MI, and pairwise-correlation kernels actually consume the dtype, and the
repo's own acceleration ladder (`CLAUDE.md`) routes float32 differently through the njit/CUDA backends.

**Proposed test:** `test_float32_X_matches_float64_X_support` in
`tests/feature_selection/mrmr/core/test_mrmr_dtype_parity.py` (new) — assert `np.array_equal(support_32,
support_64)` on a well-separated synthetic (signal gap >> float32 eps so the contract is genuinely
deterministic, not a coin flip), and `np.allclose(gains_32, gains_64, rtol=1e-5)`.
**Regression that slips past today:** a float32 code path in a batch MI kernel silently truncating bin codes.

---

### TESTGAP-4 — `partial_fit` is covered for streaming mechanics but not for the frozen-params contract at `transform` [P2]

**Where:** `tests/.../test_biz_value_mrmr_provenance_streaming/test_partial_fit_streaming.py` (the substantial
one, ~450 lines), `tests/feature_selection/mrmr_api/test_coverage_api_partial_fit.py`,
`test_setstate_partial_fit_batch_sizes.py`, `test_mrmr_partial_fit_dtype_mismatch.py`. It already pins
empty-batch and ragged-length raises (`:164`, `:171`), clone semantics (`:387`), and
`test_legacy_fit_byte_identical_when_partial_fit_not_called` (`:447`) — good.

**What has no test:** after `partial_fit(X1,y1); partial_fit(X2,y2)`, calling `transform` on a **row subset** of
`X1` must replay the frozen parameters accumulated across BOTH batches, not re-derive from the subset.
`test_mrmr_endtoend_invariants.py:426 test_I3_slice_replay_byte_exact` pins exactly this invariant — but only
for the single-`fit` path.

**Proposed test:** `test_partial_fit_transform_row_subset_replays_frozen_params_not_refit` in
`test_partial_fit_streaming.py` (existing file, existing fixtures) —
`np.allclose(m.transform(X1.iloc[10:40]).to_numpy(), m.transform(X1).to_numpy()[10:40], rtol=0, atol=0)`,
byte-exact, plus `assert m.n_features_ == n_before` (no refit-widening).
**Regression that slips past today:** a `partial_fit` state-merge change that leaves a per-batch scaler or bin
edge un-frozen; the streaming tests all transform the *full* frame, so a subset-refit is invisible.

---

### TESTGAP-5 — `get_feature_names_out()` has no *unfitted* and no *reordered-columns-at-transform* contract test [P2]

**Where:** `get_feature_names_out` appears in 106 MRMR test files, all post-fit.
`test_mrmr_endtoend_invariants.py:414 test_I2_feature_names_out_equals_transform_columns` pins the
fit-time identity well.

**What has no test:**
1. `get_feature_names_out()` on an unfitted estimator must raise `NotFittedError` (sklearn contract).
   (`test_biz_value_mrmr_provenance_streaming/test_fe_provenance_report.py:483` covers the unfitted *report*,
   not this accessor.)
2. `transform(X[reversed_columns])` — a caller handing back the same columns in a different order. MRMR must
   either reorder by name to the fitted order or raise; silently transforming positionally is a P0-class silent
   wrong result.

**Proposed tests**, both in `tests/feature_selection/mrmr/core/test_mrmr_transform_column_contract.py` (new):

| Test | Assertion |
|---|---|
| `test_get_feature_names_out_unfitted_raises_not_fitted` | `pytest.raises(NotFittedError)`. |
| `test_transform_with_reordered_columns_realigns_by_name` | `np.allclose(m.transform(X[list(reversed(X.columns))]).to_numpy(), m.transform(X).to_numpy(), equal_nan=True)` — or, if the contract is to raise, `pytest.raises(ValueError, match="order")`. Pin whichever the code actually does, having read it; do not assert both. |
| `test_transform_with_extra_unseen_column_ignores_it_and_matches` | Append `X["_unseen"] = rng.normal(...)`; assert transform output byte-identical to the un-appended one. |
| `test_transform_with_missing_fitted_column_raises_naming_it` | Drop a selected column; `pytest.raises(ValueError, match=<the dropped column's name>)` — the message must name the column, not just count. |

**Regression that slips past today:** positional-instead-of-name column alignment. Every current test passes the
identical frame object back to `transform`, so positional and name-based alignment are indistinguishable.

---

### TESTGAP-6 — `mrmr_gains_` monotonicity is asserted only "roughly", with no numeric tolerance [P3]

**Where:** `tests/feature_selection/mrmr/core/test_uaed_auto_size_works.py:70-73` —
docstring: *"The relevance trace should be roughly non-increasing across rounds"*.

**What:** "roughly" is not a contract. Greedy mRMR's per-round gain is non-increasing *by construction* within a
round family; where it is genuinely allowed to rise (an FE candidate admitted mid-fit), the test should say so
and bound it.

**Proposed test:** `test_mrmr_gains_non_increasing_within_raw_screen_prefix` (same file) — restrict to the
raw-screen prefix (FE disabled, `fe_max_steps=0`, which `test_fe_max_steps_zero_disables_all_fe.py` shows is a
supported configuration) and assert strictly `np.all(np.diff(gains) <= 1e-12)`. Keep the loose whole-trace
version only as a separate FE-enabled test with an explicit documented allowance.

---

## 2. Untested edge / degenerate inputs

The suite is **strong** here; I am reporting only what is genuinely absent. Verified present and adequate:
constant column (`test_mrmr_edge_cases.py:29`, `test_mrmr_degenerate_frames.py:54`), all-NaN column
(`test_mrmr_degenerate_inputs.py:50`, `degenerate_frames.py:47`), exact duplicate columns
(`degenerate_inputs.py:68`, `degenerate_frames.py:61`), zero surviving features
(`test_mrmr_edge_cases.py:155 test_no_features_selected_transform`), single row
(`degenerate_inputs.py:25,36`), constant/NaN/inf `y` (`degenerate_inputs.py:88`,
`degenerate_frames.py:120,130`), ragged `y` length (`degenerate_inputs.py:211,229`), empty polars frame
(`test_mrmr_dedup_fegate_adversarial.py:294`), polars struct column rejection (`:264`), row-subset slice replay
(`test_mrmr_endtoend_invariants.py:426`), pickle round-trip
(`test_mrmr_diagnostics_pickle_parity.py`), `clone` dropping fitted state (`degenerate_inputs.py:185`).

Genuinely absent:

### TESTGAP-7 — no single-feature (`n_features_in_ == 1`) fit test [P2]
**Proposed:** `test_single_feature_frame_selects_it_or_raises_clearly` in
`tests/feature_selection/mrmr/core/test_mrmr_degenerate_inputs.py` (existing file, reuse
`simple_classification_data`-style construction). Assert `m.n_features_ == 1`, `m.support_.tolist() == [True]`,
`m.mrmr_gains_.shape == (1,)`, and `m.transform(X).shape == (n, 1)`.
**Regression that slips past today:** a redundancy loop that assumes `>= 2` candidates and `IndexError`s or
silently returns an empty support on a 1-column frame. Every degenerate fixture in the suite is ≥ 3 columns.

### TESTGAP-8 — no `n_samples < n_bins` test [P2]
**Proposed:** `test_n_samples_below_n_bins_degrades_to_valid_binning` in the same file. Fit with
`n_bins=32` on `n=12` rows; assert it does not raise, that `m.n_features_ >= 1`, and that the realised bin
count per column is `<= n_unique` (via `m.discretization_edges_` if exposed, else via
`per_feature_edges` on the same column). Current adjacent coverage
(`test_coverage_kernel_edge_discretization.py`) exercises the kernel, not the fit-level under-sampled path.
**Regression that slips past today:** an empty-bin division producing `nan` MI that silently ranks last instead
of raising.

### TESTGAP-9 — no single-class target test [P1]
**Where the gap is:** `test_mrmr_degenerate_inputs.py:88 test_constant_y_raises_clear_value_error` covers
*constant* `y`. A single-class `y` after a `train_test_split` on an imbalanced frame is the same degeneracy
arriving by a different route, and the `grep -c "single_class|one class"` over the MRMR tree returns 6 files,
none asserting the fit-level contract.
**Proposed:** `test_single_class_target_raises_naming_the_class_count` — `pytest.raises(ValueError,
match="1 class|single class")`, message must state the observed class count. Sits next to the existing
constant-`y` test and reuses its structure.
**Regression that slips past today:** a rare-class path that divides by `(n_classes - 1)` and returns `inf`
relevance for every column, selecting arbitrarily.

### TESTGAP-10 — no 100 %-NaN target test [P2]
**Proposed:** `test_all_nan_target_raises_before_any_fe_work` in the same file. `y = np.full(n, np.nan)`;
assert `pytest.raises(ValueError)` AND — the load-bearing half — that it raises *before* FE runs, via
`assert not hasattr(m, "_engineered_recipes_")`. `test_y_nan_raises_valueerror` (`degenerate_frames.py:120`)
covers *partial* NaN `y` only.
**Regression that slips past today:** the all-NaN case falling through the partial-NaN guard's
`np.isnan(y).any()` check into a full FE pass that then fails deep inside a kernel with an unreadable error.

### TESTGAP-11 — rare-class fixtures below the ~5000-row stability floor [P2]
**Where:** `tests/.../test_biz_value_mrmr_contracts_robustness/test_multiclass_ordinal.py:365`
`pytest.skip(f"holdout has <10 minority rows on seed={seed}; recall estimate would be unstable")`.

**What:** the skip is a symptom, not a fix. A fixture that cannot reliably put 10 minority rows in a holdout is
undersized for a minority-computed metric. Per `CLAUDE.md`'s rare-imbalance rule (~5000 rows before a 1 %
positive rate is stable), the correct remedy is to size the fixture up, not to skip the seeds that fall short.
**This is not an xfail hiding a prod bug** — it is an undersized synthetic, so it belongs here rather than in
section 6.
**Proposed:** resize the `test_multiclass_ordinal` fixture to `n >= 5000` and **delete the skip at :365**,
converting the test to `test_biz_val_mrmr_multiclass_ordinal_minority_recall_stable_across_seeds` with a hard
`assert recall >= <measured - 10 %>` across all parametrised seeds.
The identical pattern at `test_target_leakage.py:449` (`"direct-leak column not selected on this seed"`) is a
*different* problem — a fixture whose leak signal is too weak to be reliably selected; same remedy (strengthen
the leak, drop the skip), listed here as TESTGAP-11b.

### TESTGAP-12 — `pytest.skip` used where the fixture should be made deterministic [P3]
**Where:** `test_biz_value_mrmr_dcd/test_recipe_pool.py:391` `pytest.skip("no swap fired on this fixture")`;
`test_biz_value_mrmr_fe_hybrid_orth/test_hybrid_fe_stress.py:534,566`
`pytest.skip(f"seed={seed}: no mi_greedy_transform recipe in support; cannot test corruption")`.
**Assessment:** these are fixture-determinism problems, **not** bugs being deferred — the mechanism under test
genuinely does not fire on those seeds. Still a real gap: on an unlucky seed the test silently tests nothing.
**Proposed:** construct the fixture so the mechanism provably fires (assert it fired, then test it) —
`assert m.dcd_["n_swaps"] >= 1, "fixture must provoke a swap; strengthen it rather than skipping"`, converting
each skip into a fixture precondition assertion. Same remedy the suite already applies at
`test_fe_provenance_report.py:327` ("strengthen the fixture rather than skipping the test") — this is the
repo's own established pattern, just not applied at these three sites.

### TESTGAP-13 — `pytest.xfail` on a perf assertion [P3, correctly classified, no action beyond a note]
**Where:** `test_biz_value_mrmr_linear_preselect/test_cmim_hotpath_perf.py:231 pytest.xfail(...)`.
**Assessment:** the surrounding comment (`:217-229`) shows the hard `>=0.7x` gate stays live and the xfail
only downgrades a *slow-standalone-host* sensor. This is a hardware-variance concession, not a hidden defect —
**not a finding against the no-xfail rule.** Recorded so the next wave does not re-flag it.

---

## 3. Prior-wave DONE findings — is there a pinning test? (the cross-check)

Method: for each DONE row in `audits/mrmr_audit_2026-07-25/_TRACKER.md` (and spot-checks into `-07-22` /
`-07-20`), grep the tests tree for the *mechanism*, not just the helper name.

**Result: essentially clean.** I found no DONE finding shipped without a pinning test. Detail, so this is
auditable rather than a bare assertion:

| ID | Fix | Pinning test found | Verdict |
|---|---|---|---|
| FIT_IMPL-1 | `_align_mrmr_gains` last before `return self` | `tests/feature_selection/test_group_aware_fe_pair_leak.py:90-106` — asserts `len(m.mrmr_gains_) == m.n_features_ == n_out` with the docstring naming post-demotion desync as the pre-fix failure. Also `test_mrmr_critique_uaed_support_consistency.py:46`, `test_biz_value_mrmr_contracts_robustness/test_degenerate_features.py:444`, `test_target_leakage.py:515-518` | **PINNED** |
| DISCRETIZATION-2 | `bb_subsample_threshold` bounded default | 1 test file references the param | PINNED |
| DISCRETIZATION-5 | `optimal_bin_edges` zero coverage | 1 file now references it | PINNED |
| GPU_INFRA-1 | `dispatch_friend_graph_stats` broaden except | 3 files | PINNED |
| GPU_INFRA-2 | `gpu_globally_disabled()` guard | 15 files | PINNED |
| ORTH-1 / ORTH-2 | `preprocess_params_*` at multi-leg recipe sites + arity≥2 slice-replay parity | 5 files reference `preprocess_params_`; `test_mrmr_endtoend_invariants.py:426 test_I3_slice_replay_byte_exact` | PINNED |
| FE_FAMILIES_B-1 / B-3 | `_maybe_rerank_with_mm` == `_compute_pair_ii_mm` | 5 and 4 files respectively | PINNED |
| STABILITY_MISC-1 | polars bridge via `X.to_pandas()` | 54 files touch `group_aware` | PINNED (but see TESTGAP-2 — the *general* parity invariant is still unpinned) |
| STABILITY_MISC-2 | `_apply_tree_rescue` truncate-after-filter | 2 files | PINNED |
| CORE_CLASS-1 | `ensemble_scorers` default tuple | 2 files | PINNED |
| CORE_CLASS-2 | `set_params(nested_config=...)` expansion | 2 files reference `nested_config` | PINNED |
| FE_STEP-1 | `compute_pair_maxt_floor` chunking | `tests/feature_selection/filters/test_pair_maxt_floor_subsample_and_breaker.py` | PINNED |
| FE_STEP-2 | `_non_numeric_column_indices` logging | 4 files | PINNED |
| FE_PAIRS-1 / -3 | `_GPU_GATE_CACHE` lock + key widening | 2 files | PINNED |
| INFO_THEORY-2 / -3 | `clear_cmi_xc_resident_cache` / `clear_mah_y_binning_cache` locks | 2 / 3 files | PINNED |
| INFO_THEORY-4 | `merge_vars` `min_occupancy` | 4 files | PINNED |
| INFO_THEORY-5 / -6 | `MAX_JOINT_CARDINALITY` + int16 clamp | 2 files; `n_classes_y` in 4 GPU test files; int16/32767 in `test_apply_bin_edges_dtype_nan.py` | PINNED |
| INFO_THEORY-8 / -9 | `renyi_alpha_cmi` nats/bits; `fastmi` seed | `test_renyi_alpha_mi_estimator.py`, `test_mi_dispatch_contract.py`; `test_fastmi_mise_lse_hoist_identity.py`, `test_biz_val_mi_estimators.py` | PINNED |
| SCREEN_CONFIRM-5 | `_EVALUATE_CANDIDATES_POOL_ENABLED` dead path | `test_mrmr_parallel_confirm_equiv.py`, `test_screen_predictors_evaluate_candidates_serial_only.py`, `test_coverage_mm_toggle_parallel.py` | PINNED |
| GPU_INFRA-7 | `free_blocks=True` default | **no test** — but this row is marked **DOC**, not DONE; the bench is committed and the default deliberately not flipped. Correctly untested. | N/A |

### TESTGAP-14 — the one real gap in this section: `CROSS-2` and `CROSS-4` are marked **PARTIAL** and have no gate keeping them from regrowing [P3]

**Where:** `audits/mrmr_audit_2026-07-25/_TRACKER.md`, P3 table — CROSS-2 (audit-metadata comments: markers and
all 185 stale citations gone, ~840 prose date mentions remain) and CROSS-4 (`--` in prose, ~80 % swept).

**What:** both are PARTIAL by design (the remainder needs human reading, not a regex — a reasonable call). But
there is no ratchet pinning the *cleaned* portion, so the finding-ID markers and stale `file:line` citations
can silently regrow to their prior count. The repo already has the machinery: `tests/test_meta/` hosts exactly
this kind of gate, and `test_no_source_text_claims.py` demonstrates the baseline-JSON ratchet pattern
(`_source_text_baseline.json`, keyed `file::function::kind`, with a `--refresh-…-baseline` flag).

**Proposed test:** `test_no_new_audit_metadata_comments` in `tests/test_meta/` (new), baselined as
`_audit_metadata_baseline.json` counting per-file occurrences of finding-ID markers
(`/[A-Z_]+-\d+/` in a comment), `Wave \d`, `loop iter \d`, and `.md finding`. Assert the per-file count never
*rises* above baseline, and that a baseline key for a file with zero remaining occurrences fails as stale —
the exact two-directional ratchet `test_no_source_text_claims.py` already implements.
**Regression that slips past today:** the next audit wave reintroduces 200 finding-ID comments, against the
CLAUDE.md comment rule, and nothing notices.

---

## 4. Existing tests that are weak (anti-patterns, with the tightened assertion)

### TESTGAP-15 — vacuous assertion: always true for every possible value [P2]

**`tests/feature_selection/mrmr/core/test_mrmr_stability_method_validation.py:63`**
```python
assert np.asarray(m.support_).size >= 0
```
`.size` is a non-negative int by definition. This assertion **cannot fail**. It passes on an empty support, a
`None`-valued support coerced to a 0-d array, and a wholly broken stability method.
**Tighten to:** `sup = np.asarray(m.support_); assert sup.dtype == bool and sup.shape == (X.shape[1],) and
sup.sum() >= 1, f"stability method {method!r} selected nothing"` — parametrised over the stability methods the
file already sweeps.

### TESTGAP-16 — near-vacuous disjunct swallows the real invariant [P2]

**`tests/feature_selection/mrmr/core/test_uaed_auto_size_works.py:67`**
```python
assert sel.mrmr_gains_.size == len(sel.support_) or sel.mrmr_gains_.size >= 1
```
The `or` branch is satisfied by *any* non-empty gains array, so the length-alignment invariant — the exact
thing FIT_IMPL-1 was a bug about — is not actually enforced here. The file's own docstring (`:55-57`) states the
contract as "a non-empty float array after fit", but the alignment half is what matters.
**Tighten to:** `assert sel.mrmr_gains_.size == sel.n_features_ == int(np.asarray(sel.support_).sum())`,
dropping the disjunct entirely. (Other files already assert this correctly — `test_group_aware_fe_pair_leak.py:105`
— so this site is simply weaker than its siblings.)

### TESTGAP-17 — type-shape assertion where a value assertion is available [P2]

**`tests/.../test_biz_value_mrmr_dcd/test_accessors.py:362`**
```python
assert cm is None or isinstance(cm, dict)
```
Falsifiable only by a non-dict non-None value. It passes when `cluster_members_` is `None` *because DCD silently
failed to run at all* — which is the regression a DCD test most needs to catch. The `if isinstance(cm, dict) and
cm:` block that follows (`:365-368`) is the real test and is **entirely skipped** on the `None` path.
**Tighten to:** given the fixture sets `dcd_enable=True, fe_hybrid_orth_enable=True` and is constructed to
produce clusters, assert positively: `assert isinstance(cm, dict) and len(cm) >= 1, "dcd_enable=True on a
clustered fixture must produce cluster_members_"`, then run the sub-key check unconditionally.

### TESTGAP-18 — same pattern, two sites: `is None or <trivially-true>` [P2]

**`test_accessors.py:745`** and **`test_biz_value_mrmr_dcd/test_recipe_pool.py:464`**, both:
```python
assert m.dcd_ is None or m.dcd_.get("n_swaps", 0) == 0
```
Intended as "no swaps fired". Passes when `dcd_` is absent entirely — i.e. when the whole DCD mechanism was
disabled by a regression, the strongest failure this assertion could ever face.
**Tighten to:** `assert m.dcd_ is not None, "dcd_ must be populated when dcd_enable=True"` then
`assert m.dcd_["n_swaps"] == 0`. Note `.get("n_swaps", 0)` also defaults a *missing key* to the passing value —
use `m.dcd_["n_swaps"]` so a renamed key fails loudly.

### TESTGAP-19 — `>= 0` on a quantity that is non-negative by construction [P3]

**`test_biz_value_mrmr_provenance_streaming/test_batched_pairwise_su_dispatch.py:502`** and
**`test_biz_val_mrmr_dcd.py:251`**, both `assert sel.dcd_["n_su_calls"] >= 0`.
A call *count* is never negative. The tests sit in files whose whole point is that the batched SU dispatch
*fires*.
**Tighten to:** `assert sel.dcd_["n_su_calls"] >= 1` at minimum; better, pin the expected count for the fixture's
known pair budget (`== n_expected_pairs`), which is what makes the batched-vs-serial dispatch regression
detectable.

### TESTGAP-20 — existence-check masquerading as a behaviour test [P2]

**`tests/feature_selection/mrmr/core/test_mrmr_edge_cases.py:66-73`**
`test_skip_retraining_parameter_exists` — docstring *"Just test that the parameter can be set without error"*,
body ends `assert hasattr(mrmr, "n_features_")`. The name itself admits it (`_parameter_exists`), and it
violates the repo's `test_<failure_mode>` naming rule.
**Note:** the *behaviour* is covered elsewhere (`test_skip_retraining_x_content.py`,
`test_skip_retraining_invalidation.py`), so this is redundancy plus a misleading name rather than a coverage
hole.
**Remedy:** delete it, or rename to `test_skip_retraining_on_same_content_accepted_without_error` and assert
something real — `assert m.get_params()["skip_retraining_on_same_content"] is True` after fit (the param
survives fit unmutated, which is the sklearn contract this actually touches).

### TESTGAP-21 — `isinstance(..., list)` passes on an empty list [P3]

**`test_biz_value_mrmr_fe_mechanisms/test_all_fe_mechanisms_e2e.py:344`**
```python
assert isinstance(getattr(m, attr, None), list), f"{attr} missing or not a list after fit with all FE enabled"
```
Run "with all FE enabled", so every one of the swept `attr`s should be **non-empty**. An FE family that silently
stops producing candidates passes.
**Tighten to:** `val = getattr(m, attr, None); assert isinstance(val, list) and len(val) >= 1, f"{attr} empty
after fit with all FE enabled"` — and where a family legitimately may produce nothing on this fixture, list those
attrs explicitly in an allow-set rather than weakening the assertion for all of them.

### TESTGAP-22 — filesystem-path assertion as a proxy for behaviour [P3]

**`test_biz_value_mrmr_linear_preselect/test_mechanism_dataset_showdown.py:541`**
```python
assert flat.exists() or relocated, "Layer 29 module missing; ..."
```
This asserts a *file exists at a path*, which is the same class of fragility as `inspect.getsource` — it breaks
on a harmless move and passes for a module that exists but is broken.
**Tighten to:** `importlib.import_module(<the module>)` and assert the reference function is callable and returns
the expected shape on a 3-row input. An import is behaviour; a path check is not.

### TESTGAP-23 — `inspect.getsource` in MRMR tests: 2 + 49 call sites, allowlisted as known debt [P2, tracked not new]

**Where:** `tests/feature_selection/mrmr/test_audit_verification_gaps.py:95,212` (2 sites) and
`tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py` (**49 sites** — `:451, 788, 789,
840, 995, 1031, 1032, 1052, 1186, 1241, 1513, 1530, 1548, 1549, 1576, 1607, 1713, 1737, 1780, 1819, 1901, 1921,
1936, 1953, 2001, 2032, 2051, 2112, 2186, 2203, 2375, 2399, 2533, 2578, 2598, 2675, 2685, 2707, 2778, 2829,
2846, 3170, 3678, 3786, 3899, 3968` plus the AST walk at `:1737`).

**Assessment — and this is the honest framing:** the repo **already has a live gate** for this
(`tests/test_meta/test_no_source_text_claims.py`, backed by the shared `py_ci_shared.source_text_claims` AST
detector). Both MRMR files are in its `_ALLOWLIST` with a documented reason ("148 call sites across 28 files,
2026-07 known debt"), and every remaining instance is keyed in `_source_text_baseline.json` so the gate stays
live for NEW files. So this is **tracked debt, not an ungoverned hole** — I am reporting it because the brief
asks me to flag every instance, not because the gate is missing.

**Where the debt is nonetheless real:** 49 sites in one MRMR regression file is by far the largest single
concentration in the repo, and each one asserts that a *string appears in a function body*. Per the brief's own
framing these "break on harmless refactors while passing for implementations that are actually wrong" — and this
file has already been bitten twice, per its own in-file comments at **`:2289`** (*"This used to be checked by
reading `inspect.getsource` for the import lines, which breaks on any…"*) and **`:2725`** (*"unrelated monolith
split, which broke the original `inspect.getsource()` assertion"*). Two self-documented breakages is empirical
proof of the failure mode, in this exact file.

**Proposed:** a scoped conversion campaign, not a single test. Highest-value first, because each pins a
prior-wave P0/P1 fix behaviourally instead of textually:
- `:1513` `_dcd_swap.commit_swap` → call it and assert the swap is applied to the returned state.
- `:3899` `resident_operand` → call twice with identical content, assert one upload via a counter (the cache
  hit/miss counter the CLAUDE.md `_MAX_ENTRIES` lead already asks for) rather than reading the source.
- `:788-789` `gpu_materialise_discretize_codes_host` / `gpu_discretize_codes_host` → assert the emitted codes
  match the CPU reference, not that the source mentions a dtype.
- `:3678` `selection_stability_report` → assert the report's numeric content on a fixture with known stability.
Each converted site should be removed from `_source_text_baseline.json` in the same commit, so the ratchet
tightens.

---

## 5. Missing `biz_value` coverage, per mechanism

Measured by grepping each live parameter value against the `biz_val`/`biz_value` test files. The suite has
**134 biz_value files** for MRMR and the coverage is broad; the gaps below are the mechanisms where the count
is zero.

| Mechanism | Live where | biz_value files | Finding |
|---|---|---|---|
| `nbins_method="marx"` / `"mah_sci"` / `"sci"` | `_mrmr_param_constants.py:30,47` | **0** (and 0 tests of any kind) | **TESTGAP-1** above — the P1 |
| `cluster_scoring="factor_score"` (Bartlett 1-factor combiner) | `_mrmr_param_constants.py:57,70`; `_cluster_aggregate.py:71,257` | **0** (5 non-biz_val files reference it) | **TESTGAP-24** below |
| `bayesian blocks` (`nbins_method="blocks"`) | `_adaptive_nbins.py` | 1 | thin; see TESTGAP-25 |
| `redundancy_policy` `pld_max` / `pld_mean` | `_mrmr_param_constants.py:18-19` | 1 each | thin; see TESTGAP-26 |
| `complementary_pairs` stability method | `_mrmr_param_constants.py:36` | 1 | thin |
| `mean_inv_var` cluster scoring | `:57` | 1 | thin |

### TESTGAP-24 — `cluster_scoring="factor_score"` has no biz_value test [P2]
**The mechanism:** Bartlett 1-factor score as the cluster-representative combiner
(`_cluster_aggregate.py:257` — *"factor_score -> Bartlett"*), one of five options, and the only one with no
quantitative test. Its documented edge over `mean_z` is under **heterogeneous loadings and heteroscedastic
noise** (`_mrmr_class.py:1363` names exactly this).
**Proposed test:** `test_biz_val_mrmr_cluster_scoring_factor_score_beats_mean_z_under_hetero_loadings` in
`tests/feature_selection/mrmr/biz_val/test_biz_val_mrmr_cluster_scoring.py` (new).
**Synthetic:** one latent factor `f ~ N(0,1)` driving `y`; 6 observed columns `x_i = l_i * f + e_i` with
loadings `l = (2.0, 1.7, 1.4, 0.5, 0.35, 0.2)` and *per-column* noise sd `s = (0.2, 0.25, 0.3, 1.4, 1.6, 1.8)`
— heterogeneous in both, which is the regime the docstring claims; plus 10 pure-noise columns. n=4000.
**Baseline:** the same fit with `cluster_scoring="mean_z"` (the closest baseline, and the current default
family), identical `random_state`.
**Assertion:** downstream LogReg ROC-AUC on a held-out split,
`assert auc_factor >= auc_mean_z + 0.02` and `assert auc_factor >= <measured - 10 %>` as an absolute floor.
Not `is not None`; a real number on a real holdout.
**Runtime:** <5 s (n=4000, 16 columns, single fit each).
**Regression that slips past today:** the Bartlett combiner silently degenerating to an unweighted mean —
identical output to `mean_z`, no test anywhere would notice.

### TESTGAP-25 — `nbins_method="blocks"` (Bayesian Blocks) has one biz_value reference and no win-vs-baseline assertion [P3]
Prior-wave `DISCRETIZATION-2` (DONE) changed this path's default `bb_subsample_threshold` from 0 (unbounded
O(N²)) to a bounded value — a behaviour change to the *edges themselves* on large N, with the accuracy side
unpinned.
**Proposed:** `test_biz_val_mrmr_nbins_blocks_beats_uniform_on_piecewise_constant` in the new nbins biz_val
file. Synthetic: a feature that is piecewise-constant across 4 regimes, each regime carrying a different class
rate — the exact shape Bayesian Blocks is designed for. Baseline `nbins_method="uniform"` at matched `n_bins`.
Assert `mi_blocks >= 1.15 * mi_uniform`. Plus
`test_bb_subsample_threshold_bounds_runtime_without_changing_edges_on_small_n`: for `N < threshold`, assert
`np.array_equal(edges_bounded, edges_exact)` — the subsampling must be a no-op below the threshold, which is
the half of DISCRETIZATION-2 most likely to regress.

### TESTGAP-26 — `redundancy_policy` `pld_max` vs `pld_mean` have no comparative biz_value test [P3]
Two policies, one biz_val file each, no test distinguishing them. They differ precisely when a candidate is
strongly redundant with **one** selected feature but weakly with the rest — `pld_max` should reject it,
`pld_mean` should admit it.
**Proposed:** `test_biz_val_mrmr_redundancy_pld_max_rejects_single_strong_duplicate_pld_mean_admits` in
`tests/feature_selection/mrmr/biz_val/test_biz_val_mrmr_redundancy_policy.py` (**existing file — reuse**).
Synthetic: `x_dup = x_selected + 0.05*noise` (r≈0.999 with exactly one selected column, r≈0 with the other
four). Assert `"x_dup" not in names(pld_max)` and `"x_dup" in names(pld_mean)` — a *differential* assertion that
fails if either policy silently becomes the other.
**Regression that slips past today:** `pld_mean` being wired to the `max` reduction (or vice versa); both
single-policy tests still pass, since each only checks its own policy produces *a* reasonable support.

---

## 6. Determinism

The suite already has 129 MRMR files referencing `random_state` / determinism, plus
`tests/test_rng_determinism.py`, and the identity-cache replay tests
(`test_mrmr_basic.py:111-141` — replayed `support_` matches the first fit bit-exact).

### TESTGAP-27 — no test pins MRMR selection as independent of `pytest-randomly`'s global seed [P2]
**What:** `pytest-randomly` reseeds the global numpy/`random` state per test. MRMR's contract with an explicit
`random_state` is that selection is *independent* of that global state. Nothing asserts it. A code path that
falls back to `np.random.default_rng()` (no seed) — the exact bug prior-wave `INFO_THEORY-9` fixed in
`_fastmi.py:146,193`, where a hardcoded `default_rng(0)` had no seed param — would produce a test that passes
under a fixed `-p no:randomly` run and flakes only in CI.
**Proposed test:** `test_selection_independent_of_global_numpy_seed` in
`tests/feature_selection/mrmr/core/test_mrmr_determinism.py` (new):
```
for gseed in (0, 1, 999_983):
    np.random.seed(gseed); random.seed(gseed)
    m = MRMR(random_state=42, ...).fit(X, y)
    supports.append(m.support_.copy()); gains.append(m.mrmr_gains_.copy())
assert all(np.array_equal(supports[0], s) for s in supports[1:])
assert all(np.array_equal(gains[0], g) for g in gains[1:])   # byte-exact, not allclose
```
Byte-exact on gains, not `allclose` — an unseeded RNG leaking into a subsample perturbs gains at ~1e-3, far
above any FP-reorder tolerance, so `allclose` at a loose rtol would mask it.
**Companion:** `test_two_fits_same_random_state_are_byte_identical` (same file) —
`np.array_equal(m1.mrmr_gains_, m2.mrmr_gains_)` on fresh estimators, guarding against the identity cache
being the *only* reason replay currently matches.

---

## Proposed tests — rollup

| # | Test | File | Category |
|---|---|---|---|
| 1 | `test_nbins_method_marx_is_not_a_silent_alias_of_mah` | `discretization/test_adaptive_nbins_estimator_aliases.py` (new) | invariant |
| 2 | `test_nbins_method_sci_produces_finite_monotone_edges` | same | invariant |
| 3 | `test_biz_val_filters_mrmr_nbins_marx_beats_uniform_on_multimodal` | `mrmr/biz_val/test_biz_val_filters_mrmr_nbins_marx.py` (new) | biz_value |
| 4-6 | `test_polars_and_pandas_fits_select_identical_support`, `…_identical_with_fe_enabled`, `…_transform_emit_identical_values` | `mrmr/core/test_mrmr_polars_pandas_selection_parity.py` (new) | parity |
| 7 | `test_float32_X_matches_float64_X_support` | `mrmr/core/test_mrmr_dtype_parity.py` (new) | parity |
| 8 | `test_partial_fit_transform_row_subset_replays_frozen_params_not_refit` | `…/test_partial_fit_streaming.py` (existing) | invariant |
| 9-12 | `test_get_feature_names_out_unfitted_raises_not_fitted`, `test_transform_with_reordered_columns_realigns_by_name`, `test_transform_with_extra_unseen_column_ignores_it_and_matches`, `test_transform_with_missing_fitted_column_raises_naming_it` | `mrmr/core/test_mrmr_transform_column_contract.py` (new) | invariant |
| 13 | `test_mrmr_gains_non_increasing_within_raw_screen_prefix` | `test_uaed_auto_size_works.py` (existing) | invariant |
| 14 | `test_single_feature_frame_selects_it_or_raises_clearly` | `test_mrmr_degenerate_inputs.py` (existing) | edge |
| 15 | `test_n_samples_below_n_bins_degrades_to_valid_binning` | same | edge |
| 16 | `test_single_class_target_raises_naming_the_class_count` | same | edge |
| 17 | `test_all_nan_target_raises_before_any_fe_work` | same | edge |
| 18 | `test_biz_val_mrmr_multiclass_ordinal_minority_recall_stable_across_seeds` (resize fixture, delete skip) | `…/test_multiclass_ordinal.py` | sizing |
| 19 | leak-fixture strengthening, delete skip | `…/test_target_leakage.py:449` | sizing |
| 20-22 | fixture-precondition assertions replacing 3 skips | `test_recipe_pool.py:391`, `test_hybrid_fe_stress.py:534,566` | determinism |
| 23 | `test_no_new_audit_metadata_comments` | `tests/test_meta/` (new) | ratchet |
| 24-31 | 8 tightened assertions (TESTGAP-15…22) | in place | weak-assertion |
| 32-35 | 4 highest-value `getsource`→behaviour conversions | `test_regression_mrmr_audit_2026_07_22.py` | anti-pattern |
| 36 | `test_biz_val_mrmr_cluster_scoring_factor_score_beats_mean_z_under_hetero_loadings` | `mrmr/biz_val/test_biz_val_mrmr_cluster_scoring.py` (new) | biz_value |
| 37-38 | `test_biz_val_mrmr_nbins_blocks_beats_uniform_on_piecewise_constant`, `test_bb_subsample_threshold_bounds_runtime_without_changing_edges_on_small_n` | new nbins biz_val file | biz_value |
| 39 | `test_biz_val_mrmr_redundancy_pld_max_rejects_single_strong_duplicate_pld_mean_admits` | `test_biz_val_mrmr_redundancy_policy.py` (existing) | biz_value |
| 40-41 | `test_selection_independent_of_global_numpy_seed`, `test_two_fits_same_random_state_are_byte_identical` | `mrmr/core/test_mrmr_determinism.py` (new) | determinism |

**41 proposed tests**: 8 invariant, 4 edge/degenerate, 4 parity, 5 biz_value, 2 determinism + 3 skip-removals,
2 fixture-sizing, 8 assertion tightenings, 4 anti-pattern conversions, 1 meta-ratchet.

---

## Verified-clean (so the next wave does not re-audit blind)

- **Prior-wave DONE cross-check**: every DONE row in `-07-25/_TRACKER.md` that touches MRMR has a pinning test
  (table in section 3). The only untested row, `GPU_INFRA-7`, is marked DOC and correctly has none.
- **Degenerate-input matrix**: constant column, all-NaN column, exact duplicates, collinear, zero surviving
  features, single row, constant/NaN/inf `y`, ragged `y` length, empty polars frame, polars struct rejection,
  polars LazyFrame auto-collect, row-subset slice replay, pickle round-trip, `clone` dropping fitted state —
  **all covered**, with real assertions. Only the five in section 2 are missing.
- **`inspect.getsource` governance**: a live shared AST gate (`tests/test_meta/test_no_source_text_claims.py` +
  `py_ci_shared.source_text_claims` + `_source_text_baseline.json`) already covers all three spellings
  (getsource, read_text substring, `.find`/`.index` byte positions). The MRMR files are allowlisted **with a
  documented reason** and baselined. Debt, not a hole.
- **`biz_value` naming and structure**: the 134 MRMR biz_value files follow the CLAUDE.md convention
  (`test_biz_val*`, one concern per file, real AUC/MI numbers) and the file docstrings repeatedly and explicitly
  state "NEVER xfail" — the convention is understood and applied, not merely written down.
- **No xfail hiding a production bug** was found in the MRMR tree. The single `pytest.xfail`
  (`test_cmim_hotpath_perf.py:231`) is a documented hardware-variance sensor sitting *behind* a live hard
  `>=0.7x` gate. The `pytest.skip`s are fixture-determinism and fixture-sizing issues (TESTGAP-11, -12), not
  deferred defects.
- **`mrmr_gains_` length-alignment** (the FIT_IMPL-1 invariant) is asserted correctly in four independent
  places; only the `test_uaed_auto_size_works.py:67` site is weak (TESTGAP-16).
