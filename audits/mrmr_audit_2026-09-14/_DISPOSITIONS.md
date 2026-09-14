# mrmr_audit_2026-09-14 — implementation dispositions

Every one of the 201 findings gets an explicit disposition. No finding is dropped, deprioritised away, or
silently omitted — including P3/Low, per the owner's instruction to implement all of them.

**Disposition vocabulary**
- **RESOLVED** — fixed in source, covered by a regression test, verified.
- **DOC** — the code is correct; the defect was in a comment/docstring/name, now corrected.
- **FUTURE** — genuinely deferred, with the reason and the concrete next action recorded.
- **REJECTED** — examined and found not to be a defect, with the reason recorded.

Batches are committed as they close (the repo's convention: ~20 fixes per commit, each with its test).

---

## Wave 1 — P0 (5/5 closed)

| ID | Summary | Disposition | Evidence |
|----|---------|-------------|----------|
| NUM-1 | FE collinearity-dedup variance from raw power sums (k=2) in all three backends; large-offset column → `nan` → "not a duplicate" → collinear columns survive | **RESOLVED** | `_row_center_finite` added; centring applied once in the dispatcher so all three backends inherit it. Live repro before/after: at offset 1e6 the uncentred form returned \|r\|=2.12 (impossible for a correlation), at 1e8+ it returned `nan`; centred returns 1.0 in every regime. `test_orth_dedup_large_offset_stability.py` (12 tests, incl. a seed-parametrised teeth test in the arithmetically-guaranteed regime and a negative control) |
| NUM-12 | maxT floor substituted with `0.0` on failure — the literal "gate off" sentinel, indistinguishable from the deliberate narrow-pool no-op | **RESOLVED** | Permissive degradation kept (a failed floor must not kill the fit) but recorded as `_pair_maxt_floor_failed_` state and the log now names what is disabled. Flag reset per call so a stale `True` cannot leak across FE steps. `test_pair_maxt_floor_failure_is_distinguishable.py` |
| NUM-13 | the circuit breaker making NUM-12 recoverable had its own trip failure swallowed to `debug` + bare `pass` | **RESOLVED** | Escalated to `logger.warning` naming the exception type and the consequence; dead `pass` removed. Test drives the real branch (resident path forced on → faults → breaker trip raises) rather than asserting on source text |
| PERIPHERY-1 | `transform()` identity fast-path compared only a column COUNT and returned before the column-name check, so a reordered/different same-width named frame came back unchanged, every column mis-mapped | **RESOLVED** | `_identity_fastpath_is_safe` requires exact name-and-order equality for named frames (pandas + polars); ndarrays stay eligible since their width is validated upstream. `test_mrmr_transform_identity_fastpath_identity.py` (6 tests) |
| RO-1 | RelaxMRMR-3D's two conditional-MI terms had transposed argument slots: computed `I(X;Y\|Z)` where the formula needs `I(X;Z\|Y)`; the joint helper structurally could not express the intended quantity | **RESOLVED** | `_joint_cmi_xy_given_zw_njit` replaced by correctly-oriented `_joint_mi_x_zw_given_y_njit`; both call sites fixed. Verified against analytically-known cases (XOR: pair determines X → ln2; X==Y → 0; single member alone → 0). The 7 pre-existing tests pass on BOTH orientations, which is why the new test asserts the quantity, not the sign. `test_relaxmrmr_3d_cmi_orientation.py` (5 tests, incl. an explicit teeth test that the two orientations differ) |

### Notes carried from the verification pass

- **NUM-12 severity nuance (recorded, not dropped):** the outer handler already logged at `warning` with
  `exc_info=True`, so the failure was never invisible in a log — only in the data. The finding stands on the
  data-indistinguishability, and that is what the fix addresses.
- **Test-construction lesson worth keeping:** `pooled_pair_permutation_null_joint_mi_floor` takes DISCRETISED
  codes, a per-column `nbins` vector, and a per-ROW `classes_y`. Passing the natural-looking
  `classes_y=[0,1]` (distinct classes) plus a scalar `nbins` indexes the njit kernel out of bounds and
  **segfaults rather than raising**. Filed as a follow-up candidate below.

---

## New findings raised during implementation

| ID | Summary | Disposition |
|----|---------|-------------|
| IMPL-1 | `pooled_pair_permutation_null_joint_mi_floor` segfaults (no bounds validation) when `classes_y` is passed as the distinct-class list instead of per-row codes, or `nbins` as a scalar | **FUTURE** — a cheap shape/range assert at the Python boundary would convert a segfault into an actionable error. Not bundled into the P0 wave to keep that commit reviewable; to be picked up with the P2 batch touching this file |
| IMPL-3 | Same closure as NUM-14 (`_pairs_setup._prewarp_generalises`) also returns `True` ("accept") when an operand's length differs from the validation frame's `_pw_n`, with the comment "subsample edge -> don't block" | **FUTURE** — same accept-on-anomaly SHAPE as NUM-14, but NOT the same defect: the comment documents a legitimate structural case (a subsampled operand cannot be validated against a full-length held-out mask), not a failure. Flipping it to reject blindly could block every valid warp on the subsample path. Next action: establish whether a length mismatch can arise from anything OTHER than the documented subsample path (e.g. a misaligned operand); if it can, distinguish the two and reject only the misaligned case |
| IMPL-2 | Composite-target discovery diagnostics were written FLAT into the run root as `<data_dir>/composite_<target>_<suffix>.png`, so a multi-target run dumped every composite chart beside the data artifacts instead of grouping them per target under `charts/`, unlike every other chart type | **RESOLVED** — owner-reported, outside the 9-agent scope. Now `<data_dir>/charts/<slugify(target)>/composite_discovery/{mi_gain,tdist_<spec>}.png`, using the same `slugify(target_name)` `_setup_model_directories` uses, so the composite dir is a sibling of that target's per-model chart dirs. Filenames drop the now-redundant target prefix. The existing test pinned the OLD flat layout and was re-framed to the new one (per the repo's "re-frame the stale test" rule), plus two new assertions: nothing composite-shaped may remain loose in the run root, and every path recorded in `metadata["composite_target_diagnostic_charts"]` must actually exist |

---

## Wave 2 — P1 (6 of 25 closed)

| ID | Summary | Disposition | Evidence |
|----|---------|-------------|----------|
| CORE-1 | `sample_weight` / `groups` forwarded into `_stability_outer_fit`, which never referenced `fit_kwargs`; the wrapper's weight resampling runs AFTER that branch returns, so every non-classic `stability_selection_method` selected on unweighted data | **RESOLVED** | The classic path converts weights to a row resample via `_maybe_resample_for_sample_weight` (`_mrmr_class.py:3662`), but `_stability_outer_fit` returns at `:3534` — before it. Fix applies the same resampling inside the method, so it cannot be bypassed by another caller and each replicate inherits the weighting with no inner plumbing. Uniform/absent weights return inputs unchanged, keeping the unweighted path byte-identical. Measured: a split-signal fixture where half A makes `a` predictive and half B makes `b` predictive returned `['a','b']` for BOTH opposite weightings pre-fix; post-fix returns `['a']` and `['b']` respectively |
| FIT_IMPL-1 | stability-replay mask built by comparing `feature_names_in_` indices against cols-space indices | **RESOLVED** | Translated through NAMES, the only space both sides agree on; engineered selections (absent from `feature_names_in_` entirely) now included via `_engineered_features_`. Measured pre-fix on a leading-categorical fixture: report claimed `['n2','n3']` for a support of `['n1','n2']`. `test_stability_replay_mask_index_space.py` (3 tests, incl. one asserting the injected target column is never marked selected) |
| FIT_IMPL-3 | raw-signal-retention re-add did not filter on `_allowed_raw_idx`, re-admitting a column the caller excluded via `factors_names_to_use` | **RESOLVED** | One-line guard matching every sibling re-add pass; `_allowed_raw_idx` was already a parameter of the function and used 233 lines earlier |
| NUM-4 | Surrogate seeder: when every permuted-y run failed it set `perm_std = 0.0` and divided by `perm_std + 1e-9`, so any positive gap became z ~ 1e9 and the PAIR self-gate passed unconditionally -- the opposite of the "nominal spread so the z-gate still applies" its comment claimed | **RESOLVED** | No null, or a degenerate one (std <= 1e-12), now yields an undefined z (`-inf`) that fails the gate, recorded as `info["self_gate_null_available"]`, with a warning naming which case. The analytic baseline is kept for REPORTING only. Verified pre-fix by temp-revert: no-null gave **z = 424999999.99**, degenerate-null **z = 449999999.99**, both gates passed; 4 of 6 tests failed while both healthy-null controls still passed (so the controls are not vacuous). `test_surrogate_seeder_self_gate_needs_a_null.py` (6 tests); the pre-existing `test_biz_value_gbm_seeder_order3_floor.py` (9) still passes |
| NUM-14 | Prewarp held-out validation returned `True` ("accept the warp") on ANY exception, at debug -- promoting an unvalidated operand exactly when the check could not run | **RESOLVED** | Now rejects (`False` is already an ordinary outcome of this closure) with a throttled warning. Verified pre-fix by temp-revert: an ALS fit raising only on the train slice left `spec_by_var = {0: {...}, 1: {...}}` registered; post-fix it is empty. A healthy-validation control proves the setup still admits genuine synergy, so the rejection test cannot pass by the setup simply registering nothing. `test_prewarp_validation_failure_rejects.py` (2 tests). The sibling length-mismatch accept in the same closure is tracked separately as IMPL-3 |
| PERIPHERY-2 | `transform()` deep-copied the WHOLE caller frame on every call (`_X_for_recipes.copy()`) purely to own an append surface, though the replay loop only reads single named columns -- 100+ GB frames | **RESOLVED** | The audit's proposed `copy(deep=False)` was deliberately NOT used: this repo already shipped a bug where a shallow copy shared the source BlockManager and a later setitem leaked the new column back onto the caller's frame (`_build_disc_df_for_target`'s note). Instead the deep copy is kept but narrowed to `_recipe_reachable_columns`: the transitive closure of every recipe's `src_names` (following `nested_parent_a/b`) plus any `extra` string that names a real column, as a deliberate superset so a kind reaching outside `src_names` stays covered. Falls back to the full frame on an empty match or any error. `test_transform_recipe_frame_is_narrowed.py` (8 tests, incl. an explicit caller-frame-not-mutated guarantee and a cyclic-chain termination guard); the existing replay-parity suites (float32 provenance, GPU-fallback dtype, orth-cluster basis — 20 tests) pass unchanged |

## Waves 3+ — remaining P1 / P2 / P3

In progress; each finding is appended here as it closes.

### Incidental fixes made while committing

| Item | Disposition |
|------|-------------|
| Two pre-existing `ISC004` (unparenthesized implicit string concatenation in a collection) in `tests/feature_selection/biz_val/test_biz_val_synthesis_and_drop_matrix.py:171` and `tests/test_meta/test_no_lazy_from_import_under_joblib_delayed.py:125` | **RESOLVED** — surfaced by the blocking `ruff tests/` gate, untouched by this session's commits. Per CLAUDE.md ("just fix what a linter surfaces, regardless of origin") they were fixed rather than worked around |
