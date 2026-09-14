# redundancy_orth — mrmr_audit_2026-09-14

## Scope

| File | LOC |
|---|---|
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py` | 522 |
| `.../_friend_graph_and_redundancy/_group2.py` | 555 |
| `.../_friend_graph_and_redundancy/_group3.py` | 434 |
| `.../_friend_graph_and_redundancy/_group4.py` | 325 |
| `.../_friend_graph_and_redundancy/__init__.py` | 176 |
| `.../_hybrid_orth_family_variants/_group1.py` | 483 |
| `.../_hybrid_orth_family_variants/_group2.py` | 501 |
| `.../_hybrid_orth_family_variants/_group3.py` | 439 |
| `.../_hybrid_orth_family_variants/_group4.py` | 354 |
| `.../_hybrid_orth_family_variants/__init__.py` | 68 |
| `src/mlframe/feature_selection/filters/_relaxmrmr_3d.py` | 214 |
| `src/mlframe/feature_selection/filters/_mrmr_tree_rescue.py` | 169 |

Support reads (not audited, used as evidence): `filters/friend_graph.py`,
`filters/_feature_engineering_pairs/_pairs_core.py`,
`tests/feature_selection/mrmr/core/test_relaxmrmr_3d_redundancy_signs.py`,
`tests/feature_selection/mrmr/core/test_mrmr_tree_rescue.py`.

Totals: **28 findings** — P0 ×1, P1 ×3, P2 ×12, P3 ×12.

---

## Findings

### RO-1 — RelaxMRMR's two conditional-MI terms have their argument slots swapped: the code computes `I(X;Y|Z)` where the formula needs `I(X;Z|Y)`  [P0]

**Where:** `src/mlframe/feature_selection/filters/_relaxmrmr_3d.py:187` and `:205` (kernel contracts at `:44-45`, `:81-83`; formula comments at `:182`, `:190-192`; module docstring `:15`)

**What:** `_cmi_xy_given_z_njit(x, y, z, K_x, K_y, K_z)` is documented (`:45`) and implemented as the plug-in `I(X; Y | Z)` — first slot and second slot are the two mutually-informative variables, third is the conditioning variable. The score function calls it as:

```python
cmi_given_y[j] = _cmi_xy_given_z_njit(x_int, y_int, sel_int[j], K_x, K_y, K_z)   # :187
```

with the inline comment on the destination variable reading `# I(X; X_j | Y)` (`:182`) and the block comment (`:191`) specifying `CO_cond = I(X; Z_i | Y) + I(X; Z_j | Y) - I(X; Z_i, Z_j | Y)`. The call actually yields `I(X; Y | Z_j)` — the target is in the *informative* slot and the selected feature in the *conditioning* slot, i.e. the two are transposed.

The same transposition occurs in the pair term (`:205`):

```python
cmi_ij = _joint_cmi_xy_given_zw_njit(x_int, y_int, col_i, col_j, K_x, K_y, K_i, K_j)
```

`_joint_cmi_xy_given_zw_njit` (`:81-91`) builds the composite from its `z1`/`z2` slots and forwards to `_cmi_xy_given_z_njit`, so this returns `I(X; Y | Z_i, Z_j)`, not the required `I(X; Z_i, Z_j | Y)`. Note the helper's signature *cannot express* the intended quantity — the composite is hard-wired into the conditioning slot — so this is a design-level mismatch, not a single typo'd argument.

The unconditional half is correct: `marg_mi[j] = I(X; Z_j)` (`:185`) and `mi_x_zz = I(X; Z_i, Z_j)` (`:207`) both live in candidate-vs-selected space. So `inter = co_cond - co_uncond` (`:209`) subtracts a quantity measured in *candidate-vs-target* space from one measured in *candidate-vs-selected* space.

Measured directly on the fixture shape the repo's own sign test uses (K=5, n=8000, latent-driver triple, seed 0):

```
as-coded  I(X;Y|Z1) = 0.2542      intended I(X;Z1|Y) = 0.0047     (54x)
as-coded  I(X;Y|Z1,Z2) = 0.1068
```

Full-score impact, redundant triple, alpha=1.0, three seeds:

```
seed 0: shipped -0.1647  vs corrected -0.5684   (alpha=0 baseline 0.2492)
seed 1: shipped -0.2141  vs corrected -0.5965   (alpha=0 baseline 0.2363)
seed 2: shipped -0.1395  vs corrected -0.5375   (alpha=0 baseline 0.2508)
```

The interaction penalty is ~3x too small, and the shipped value is not a scaled version of the correct one (it mixes two incommensurate information quantities), so candidate *ordering* changes, not just magnitude.

**Why it is wrong / costly:** this is the entire redundancy-relaxation half of the RelaxMRMR criterion — the reason the module exists. Nothing crashes, all four tests in `test_relaxmrmr_3d_redundancy_signs.py` pass, and the selector silently ranks candidates on an incoherent score whenever `relaxmrmr_alpha > 0` (wired through `evaluation.py:760-779`). The existing tests pass because they only assert a *sign direction* on fixtures where the wrong quantity happens to move the same way; none pins a numeric value or a term identity.

**Fix:** swap the slots. `cmi_given_y[j]` → `_cmi_xy_given_z_njit(x_int, sel_int[j], y_int, K_x, K_sel[j], K_y)`. For the pair term, either add a composite-in-the-second-slot helper or build `z_comp = col_i * K_j + col_j` in the caller and call `_cmi_xy_given_z_njit(x_int, z_comp, y_int, K_x, K_i*K_j, K_y)`. Rename `_joint_cmi_xy_given_zw_njit` (or repurpose it) so the slot semantics are unambiguous. Note `check_joint_cardinality` at `:168-172` guards the *current* table shapes; the corrected call's table is `(K_x, K_i*K_j, K_y)` — re-derive the guard.

**Test:** `test_relaxmrmr_interaction_terms_are_candidate_vs_selected_not_candidate_vs_target` — build a triple where `I(X;Z|Y)` and `I(X;Y|Z)` differ by >10x (the latent-copy fixture above does), assert the returned score equals a NumPy reference implementation of the docstring's formula to 1e-12. This is mutation-resistant in a way the current sign-only tests are not.

---

### RO-2 — RelaxMRMR's interaction term differences plug-in MIs with wildly unequal table sizes and applies no bias correction, so `inter` carries a systematic positive bias  [P1]

**Where:** `_relaxmrmr_3d.py:196-210`; kernels `:43-122` (no correction anywhere in the file)

**What:** `co_uncond = marg_mi[i] + marg_mi[j] - mi_x_zz` subtracts `I(X; Z_i, Z_j)` — estimated on a `(K_x, K_i*K_j)` contingency table — from two marginal MIs estimated on `(K_x, K_i)` and `(K_x, K_j)` tables. Plug-in MI bias grows with table size (`~(rows-1)(cols-1)/2n`), so the composite term is the most upward-biased of the three; subtracting it makes `co_uncond` biased *downward*, hence `inter = co_cond - co_uncond` biased *upward* — a systematic synergy reward for every candidate, largest where the selected pair has the highest joint cardinality.

**Why it is wrong / costly:** the bias is not a constant offset across candidates (it scales with `K_i*K_j` of the selected pair, which is fixed per scoring round, but also with the candidate's own `K_x`), so it re-ranks candidates with differing cardinality. Worse at small n and high `quantization_nbins` — exactly the regime the module's `min_rows_per_cell`-style guards elsewhere in the repo exist to police. The rest of this module surface explicitly uses Miller–Madow (see `_friend_graph_and_redundancy/_group1.py:158-161`, and `_cat_mm_correction.py` / `_chao_shen.py` exist as siblings); this file uses none.

Note the `max(0.0, ...)` clamps (`:77`, `:122`) are not a substitute: they bound each term from below but do nothing about the differential bias, and they make the estimator non-linear so the three biases do not cancel in the difference.

**Fix:** apply the repo's existing Miller–Madow correction (`_cat_mm_correction.py`) to all four MI/CMI estimates consistently — once each, not telescoped (the -07-25 wave found a double-MM application in a sibling; do not repeat it). Alternatively route these through the already-corrected `_bur_term` / `info_theory` primitives rather than the three private kernels here. Verify the correction is applied at exactly one level by asserting that a shuffled-target fixture drives `inter` to ~0.

**Test:** `test_relaxmrmr_interaction_term_zero_under_independence` — independent `X`, `Z_i`, `Z_j`, `Y`; assert `|inter| < 0.01` at n=2000 with `quantization_nbins=10`. Today `inter` is materially positive there. Pair with `test_relaxmrmr_interaction_bias_invariant_to_selected_cardinality` (same data, `K_sel` 4 vs 12, assert the score gap stays within tolerance).

---

### RO-3 — the raw-protection QR fast path replaced a rank-revealing `lstsq` with `qr` + `solve_triangular`, silently producing garbage on a rank-deficient base design  [P1]

**Where:** `_friend_graph_and_redundancy/_group2.py:456-494` (equivalence claim in the comment at `:456-464`)

**What:** the baseline design `_rp_base` is assembled (`:430-442`) from the continuous values of *every* already-selected column. That set routinely contains exactly-collinear pairs: a raw column and its monotone twin, a count encoding and its frequency twin, an orth-basis column and its source, a cluster aggregate and its members — the surrounding passes exist precisely because such pairs survive together. The code then does:

```python
_rp_Q, _rp_R = _rp_sla.qr(_rp_base_tr, mode="economic")
_rp_coef_base = _rp_sla.solve_triangular(_rp_R, _rp_Qty)
```

`scipy.linalg.qr` without pivoting returns a full-width `R` even on a rank-deficient design; `solve_triangular` raises only on an *exactly* zero diagonal. A near-zero diagonal gives enormous coefficients, so `_rp_r2()` returns a large negative held-out R², `_rp_r2(_rv) - _rp_r2_base` becomes noise, and every candidate is accepted or rejected arbitrarily. The `except Exception` at `:490` never fires — there is no exception, just a bad number.

The prior implementation used `np.linalg.lstsq(..., rcond=None)`, which is SVD-based and *rank-revealing*: it returns the minimum-norm solution on a rank-deficient design. The comment's equivalence evidence ("max coefficient difference ~6e-17 ... verified") is a well-conditioned-design result; QR-update-vs-lstsq equivalence does not hold at rank deficiency, and nothing in the comment or the code covers that case.

**Why it is wrong / costly:** silent, not loud. The whole raw-feature floor-drop protection (`:397-538`) degrades to noise exactly on the multi-signal, many-selected-column frames it was built for — and the `_RAW_PROTECT_MIN_INCR_R2 = 0.005` bar is then meaningless in both directions (genuine raws withheld, or noise raws admitted).

**Fix:** use `scipy.linalg.qr(..., pivoting=True)` and drop columns whose `|R[i,i]|` falls below `rcond * |R[0,0]|` before solving, or gate the fast path on a condition check (`np.linalg.cond(_rp_base_tr)` / the `R`-diagonal ratio) and fall through to the retained `lstsq` path otherwise. Keep the QR win for the well-conditioned case (the "gate a big win on its safe condition" rule).

**Test:** `test_raw_protection_qr_matches_lstsq_on_rank_deficient_base` — construct `_rp_base` containing an exact duplicate column; assert the QR path's held-out R² for a known-good candidate matches the `lstsq` reference to 1e-8 (today it does not) and that the candidate is still admitted.

---

### RO-4 — the hybrid-orth families densify a float target with `astype(np.int64)`, truncating non-integral labels into collisions — ten verbatim copies  [P1]

**Where:** `_hybrid_orth_family_variants/_group1.py:36-47, 155-166, 239-251, 320-332, 400-412`; `_group2.py:33-45, 132-144, 236-248, 324-336, 419-431`

**What:** every one of the ten blocks is byte-identical modulo the variable name:

```python
if _y_for_X.dtype.kind in "fc":
    _n_unique = int(np.unique(_y_for_X).size)
    if _n_unique <= 32:
        _y_for_X = _y_for_X.astype(np.int64)
    else:
        try:
            _y_for_X = pd.qcut(_y_for_X, q=10, labels=False, duplicates="drop").astype(np.int64)
        except Exception as exc:
            logger.debug(...)
            _y_for_X = _y_for_X.astype(np.int64)
```

"Fewer than 33 distinct float values" is not the same predicate as "integral-valued". A float target with labels `{0.0, 0.5, 1.0, 1.5}` (ordinal ratings, half-star scores, a probability-bucketed target, any label set produced by a `/2` rescale) truncates to `{0, 0, 1, 1}` — two classes silently merged. A target scaled to `[0, 1)` with ≤32 levels collapses to a single class, and the family then scores every candidate against a constant target.

The `else` branch is a second instance: the `except` fallback (`:47` etc.) is the *same* truncating cast, so a `qcut` failure on a continuous target maps e.g. `[0, 1)` returns to all-zeros — a non-neutral substitution logged only at `debug`, which is the exact pattern the brief's item 4 and this repo's two shipped MI-backend incidents call out.

**Why it is wrong / costly:** the family's MI scorer then ranks candidates against a degenerate target; every candidate scores ~0, the per-family uplift gate rejects everything, and the stage is a silent no-op — or worse, the merged classes produce a plausible-but-wrong ranking. No exception, no warning. Affects ten of the nineteen hybrid-orth families.

**Fix:** one shared helper in `_mrmr_fit_impl/_helpers.py` (the ten copies are already an independent P3 — see RO-23's sibling RO-26), implemented as `np.unique(y, return_inverse=True)[1]` for the ≤32-level case (label-preserving, collision-free) and `pd.qcut` for the continuous case, with the failure branch raising or logging at `warning` rather than substituting a truncated target.

**Test:** `test_hybrid_orth_densify_preserves_non_integral_float_labels` — call the helper with `np.array([0.0, 0.5, 1.0, 1.5])`, assert 4 distinct output codes. Regression-wise, `test_hybrid_orth_family_runs_on_half_step_float_target` — fit with such a target and assert the family appends at least one column (today it is a silent no-op).

---

### RO-5 — three held-out-R² gates build their baseline design by iterating a Python `set`, so the design's column order — and the gate's accept/reject verdict — is not reproducible across processes  [P2]

**Where:** `_group2.py:163` (`for _sn in _sel_names_now:`, set built at `:138`); `_group2.py:431` (`for _sn in _rp_sel_names:`, set at `:429`); `_group3.py:75` (`for _sn in _cf_sel_names:`, set at `:71`)

**What:** in all three the iterated object is a set of column-name strings. Python's `str` hash is salted per process (`PYTHONHASHSEED`), so the set's iteration order — and therefore the column order of `np.column_stack(_rp_base)` / `_sel_value_cols` — differs between runs of the same fit on the same data.

**Why it is wrong / costly:** OLS is order-invariant in exact arithmetic but not in floating point: the QR/normal-equations solve accumulates in column order, so the held-out R² differs at the ~1e-12..1e-9 level run to run. That is harmless *except* that the value is immediately compared against a hard threshold — `_HINGE_PROTECT_MIN_INCR_R2 = 0.003` (`:244`), `_ORTH_PROTECT_MIN_INCR_R2 = 0.01` (`:305`), `_RAW_PROTECT_MIN_INCR_R2 = 0.005` (`:422`), `_CF_PROTECT_MIN_INCR_R2 = 0.005` (`_group3.py:62`). A candidate sitting within FP noise of its bar flips between admitted and withheld across runs, so `support_` is not reproducible. Every other seeded source in these blocks (the `random_seed` permutations at `:197`, `:425`, `_group3.py:65`) was carefully made deterministic; this one was missed. Worse under many-selected-column fits, where more near-collinear columns make the solve more order-sensitive.

**Fix:** iterate a deterministic sequence — `[cols[i] for i in selected_vars]` preserves selection order and is already available at all three sites (the set is only needed for the `in`-membership tests elsewhere). Keep the set for membership, iterate the list.

**Test:** `test_heldout_gate_design_order_is_selection_order_not_set_order` — monkeypatch the gate to record the design's column names and assert they equal `[cols[i] for i in selected_vars]` (a behavioural pin that survives a refactor). Plus `test_mrmr_support_reproducible_across_hash_seeds` — run the same fit in two subprocesses with `PYTHONHASHSEED=0` and `=1`, assert identical `support_`.

---

### RO-6 — the cat-FE floor-drop gate still refits a full SVD `lstsq` per candidate; the QR-insert rank-1 update was wired into only one of the two identical sibling gates  [P2]

**Where:** `_group3.py:96-108` and its call site `:143`, versus the optimized twin `_group2.py:456-494`

**What:** `_cf_r2(_design)` does, per candidate:

```python
_A = np.column_stack(_design)                 # rebuilds the full (n, p+1) design
_yv = _cf_y[_cf_va]; _ss = ...                # recomputed every call
_coef, *_ = np.linalg.lstsq(_A[_cf_tr], _cf_y[_cf_tr], rcond=None)
```

called as `_cf_r2([*_cf_base, _cvv])` (`:143`) once per candidate. Its sibling in `_group2.py` — same fixed base, same one-extra-column shape, same train/val strides, same purpose — was rewritten to hoist `_yv`/`_ss`/`_y_tr` and the train/val base slices out of the loop (`:444-454`) and to extend a once-computed QR by one column via `scipy.linalg.qr_insert` (`:456-494`), with a measured **20x** at production shape (`p~120, n_tr~53k, 109 candidates: 91s → 4.5s`) recorded in the comment. The group3 copy never got it.

**Why it is wrong / costly:** `np.column_stack` per candidate is an O(n·p) full materialisation and `lstsq` is an O(n·p²) SVD, both repeated per candidate, where the optimized form is one O(n·p²) factorisation plus O(n·p) per candidate. On a wide fit this is the dominant cost of the pass. This is the repo's own documented "already-optimized primitive, just not wired into every call site" gap (see the `_jackknife_ece` entry in `CLAUDE.md`).

**Fix:** extract `_group2.py:444-494` into one shared helper (`_heldout_incr_r2_probe(base_cols, y, tr, va)` returning a closure) and call it from both sites. This also retires the duplicated-logic P3 and gives RO-3's rank-deficiency fix a single home instead of two.

**Test:** `test_catfe_gate_uses_incremental_qr_not_per_candidate_lstsq` — spy on `np.linalg.lstsq` and assert it is called O(1) times, not O(n_candidates), across a fit with ≥10 cat-FE candidates. Plus `test_catfe_gate_verdicts_unchanged_by_qr_rewrite` pinning admitted-set equality against the current `lstsq` results on a fixture.

---

### RO-7 — the post-selection DCD "numeric only" guard tests `data`'s dtype, which is the integer bin-code matrix, so it never excludes anything  [P2]

**Where:** `_group3.py:191-195`, comment at `:184-190`

**What:**

```python
_sel_raw_dcd = [
    int(v) for v in selected_vars
    if 0 <= int(v) < _mask_w0 and cols[int(v)] in _raw_name_set_dcd
    and np.issubdtype(np.asarray(data[:, int(v)]).dtype, np.number)
]
```

The comment states the intent: *"NUMERIC only (a string/categorical raw can never enter the PC1/Pearson aggregate — it would raise 'could not convert string to float' in the swap's combiner)"*, and explicitly notes a prior `getattr(..., None)` attempt here "always silently returned the default and was a dead no-op". The replacement is also a no-op: `data` is a single homogeneous 2-D array of quantization bin codes (`self.quantization_dtype`, default `np.int32`) — see `_group2.py:75-77`, where new columns are appended with `.astype(data.dtype)`. Every column of `data` is integer, so `np.issubdtype(..., np.number)` is unconditionally `True`.

**Why it is wrong / costly:** the categorical exclusion the comment argues for does not happen. A string/categorical raw column that is in `selected_vars` still becomes a DCD anchor/pool member, and the failure mode the comment predicts (`could not convert string to float` inside `commit_swap`'s combiner) is then caught by the blanket `except Exception` at `:277` and reported as "post-selection DCD discovery failed" — the whole pass is lost, not just the one bad column. The second attempt at this guard is as dead as the first.

**Fix:** test the source frame, not the bin codes: `isinstance(X, pd.DataFrame) and pd.api.types.is_numeric_dtype(X[cols[v]])`, with a fallback for the non-pandas path. Alternatively use `_fe_frame_ops.fe_is_numeric_col`, which the hybrid-orth siblings already use for exactly this purpose (`_hybrid_orth_family_variants/_group1.py:63`).

**Test:** `test_post_selection_dcd_excludes_categorical_raw_anchors` — fit with a selected string raw column and assert (a) no warning from `:278` is emitted and (b) the string column is absent from `_sel_raw_dcd` (spy on `discover_cluster_members`'s pool argument).

---

### RO-8 — three different target arrays feed sibling probes inside the same section: `_y_np`, `y`, and `y.to_numpy()`  [P2]

**Where:** `_group2.py:152` (`_yv = _y_np`) vs `_group2.py:415` (`y.to_numpy() if hasattr(y, "to_numpy") else y`) vs `_group3.py:55`, `_group3.py:328`, `_group4.py:64`, `_group4.py:167` (all raw `y`); and `_hybrid_orth_family_variants/_group4.py:192` (`_y_for_ens = y.to_numpy() ...`) against `_y_np` at every one of the other eighteen family sites

**What:** `_y_np` is the fit's prepared numeric target; `y` is the caller's original object, which for a classification fit may be strings, a pandas Categorical, or an object-dtype Series. The hinge/orth gate (`_group2.py:150-158`) uses `_y_np` and works; the raw floor-drop protection 250 lines later (`_group2.py:414-420`) uses `y`, and its `np.asarray(..., dtype=np.float64)` raises on a string target, so `_rp_y` stays `None` and the *entire* raw protection is disabled — logged at `debug` (`:419`). Same for the cat-FE protection (`_group3.py:55-60`) and the raw-redundancy continuous-y re-binning (`_group4.py:64-70`).

In the hybrid-orth package, `_group4.py:192` is the single site out of nineteen that reads `y` instead of `_y_np`, and it also skips the densification block every `_group1`/`_group2` family performs — so the ensemble family receives a differently-prepared target from every one of its siblings.

**Why it is wrong / costly:** whole protection passes silently switch off for an entire target class (string/categorical classification) with no signal above `debug`, while their sibling passes on the same fit keep running. The asymmetry is invisible and the fix is one token.

**Fix:** use `_y_np` everywhere — it is already threaded into both group functions' signatures. If a protection genuinely needs the untransformed target, say so in the code and raise/warn rather than disabling itself.

**Test:** `test_raw_floor_drop_protection_runs_on_string_target` — fit on a string-labelled classification target with a raw column the maxT floor drops, assert the protection's `logger.info` fires (today the pass is silently off). And `test_hybrid_orth_ensemble_uses_same_target_as_sibling_families` asserting the ensemble family's scorer receives `_y_np`-derived codes.

---

### RO-9 — two gates decide whether a helper exists by string-membership in `locals()` / `dir()`; one of the two conditions is now permanently true  [P2]

**Where:** `_group2.py:301` (`("_heldout_incr_over_selected" in locals())`) and `_group2.py:507` (`float(_effective_min_relevance_gain) if "_effective_min_relevance_gain" in dir() else float(getattr(self, "min_relevance_gain", 0.0) or 0.0)`)

**What:** `:301` gates the orth-basis protection on whether the hinge block's closure was defined earlier in the same function body. `:507` picks a relevance floor based on whether a name is in the local namespace — but `_effective_min_relevance_gain` is a *keyword parameter* of `_friend_graph_and_redundancy_passes_group2` (`:34`), so it is always in `dir()`; the `getattr(self, "min_relevance_gain", ...)` fallback is unreachable dead code.

**Why it is wrong / costly:** this is the repo's own documented sibling-split bug class expressed as a latent trap rather than a live `NameError`. `_group2.py` is already 555 LOC; the next split of this file (or any reordering that moves the hinge block to a different group) makes `:301` evaluate `False` and the orth-basis protection silently stops running — no error, just a different `support_`. The comment at `:126-134` records that this exact coupling *already* caused the protection to silently never run whenever `fe_hinge_enable=False`; the fix widened the outer gate but left the `locals()` probe in place, so the mechanism that caused the incident is still there.

**Fix:** at `:301`, restructure so the closure is defined unconditionally at the top of the block (it is a pure function of already-computed state) and drop the `locals()` test. At `:507`, delete the conditional and use the parameter directly.

**Test:** `test_orth_basis_protection_runs_with_hinge_disabled` — fit with `fe_hinge_enable=False` and a DPI-dropped orth basis column, assert it is re-added. This is the existing contract; today it holds only by the accident of both blocks living in one function.

---

### RO-10 — the tree rescue reports no per-feature gain, ranks on in-sample importance, and fits on columns the user excluded via `factors_to_use`  [P2]

**Where:** `_mrmr_tree_rescue.py:139-161`, `:145-149`

**What:** three related gaps in one path.

(a) `m.fit(Xnum, yv)` (`:139`) trains on the *same* rows the MRMR fit consumed; `feature_importances_` (`:140`) is therefore an in-screen quantity with a winner's-curse bias, and the union at `:151-153` re-admits features on it with no held-out check. Every sibling protection pass in this cluster (`_group2.py`, `_group3.py`) gates its re-add on a held-out R² increment; the rescue gates on nothing but `imp[i] > 0`.

(b) The `logger.info` at `:155-161` reports only a count and names — no gain, no importance value, no in-screen-vs-holdout label. The docstring's honest measured numbers (`:14-17`, `madelon 0.6885 → 0.7999`) are external bench results, not something the fit reports; a user cannot tell from a fit log whether a rescued feature was worth anything.

(c) `factors_to_use` is applied to the *ranking* (`:145-149`) but the LGBM is fit on the full `Xnum` (`:139`, `Xf = X.reindex(columns=cols)` at `:98` uses `feature_names_in_`, not the allowed set). Excluded columns therefore still consume split budget and shift the importances of the allowed ones. The comment at `:142-144` is careful about filter-before-truncate ordering but the fit itself was not moved inside the restriction.

**Why it is wrong / costly:** (a)+(b) mean up to `tree_rescue_top_k=20` features join `support_` with no honest evidence recorded anywhere, which contradicts this repo's val/test/OOF strictness; (c) makes `factors_to_use` a display filter rather than a real restriction for this path.

**Fix:** fit the rescue LGBM on the `factors_to_use` subset only. Add a cheap honest gate: hold out a seeded fraction, take importances from a fit on the remainder, and log each rescued feature's importance share explicitly labelled in-screen or holdout. Keep the union behaviour (it is a measured win) but make the evidence visible.

**Test:** `test_tree_rescue_respects_factors_to_use_in_the_fit` — restrict `factors_to_use` to a subset, spy on `LGBMClassifier.fit`, assert the design width equals the subset size. `test_tree_rescue_logs_per_feature_importance` — assert the emitted record carries a numeric gain per rescued feature.

---

### RO-11 — RelaxMRMR's O(|S|²) pair loop is plain Python around three serial `nogil` kernels, with an n-length int64 composite allocated per pair  [P2]

**Where:** `_relaxmrmr_3d.py:197-210`; kernels at `:43`, `:80`, `:94` (all `@njit(nogil=True, cache=True)`, none `parallel=True`)

**What:** per the brief's REJECT rule, I grepped the file: `@njit` is present, `parallel=True` / `prange` / `cuda.jit` / `cupy` / `KernelTuningCache` are **absent** — confirmed by `grep -n "parallel\|prange\|cuda\|cupy\|KernelTuning" _relaxmrmr_3d.py` returning nothing. So this is a valid perf finding, not an already-optimal path.

The `for i in range(n_S): for j in range(i+1, n_S):` loop (`:199-209`) is Python-level and executes `n_S*(n_S-1)/2` iterations, each making three Python→njit round trips (`:205`, `:207`, plus the already-cached `cmi_given_y`). Every pair also allocates a fresh `n`-length `int64` composite inside `_joint_cmi_xy_given_zw_njit` (`:88-90`) — at n=2M that is 16 MB allocated and discarded `n_S²/2` times. The `_mi_x_pair_njit` call (`:207`) re-derives the same composite independently (`:103`), so the composite is computed twice per pair.

The pairs are mutually independent given the fixed `x_int`, `y_int`, `sel_int` — precisely the repo's documented fuse-into-one-`prange` shape. The docstring's own cost note (`:141-143`) acknowledges the O(|S|²) growth and recommends "enable the dispatcher only after the per-screen filter has pruned the pool", i.e. it works around the cost rather than fixing it.

**Why it is wrong / costly:** the score is called once per candidate per greedy round, so total cost is O(p · |S|²) kernel dispatches; with `relaxmrmr_alpha > 0` on a p=500, |S|=20 fit that is ~95k Python→njit round trips per round plus ~3 TB of transient composite allocation over a large-n fit.

**Fix:** one `@njit(parallel=True)` function taking `x`, `y`, the selected columns as a single `(n_S, n)` 2-D int64 array plus `K_sel`, with `prange` over a flattened pair index; build the composite once per pair into a per-thread scratch buffer and reuse it for both the conditional and unconditional terms. Accumulate per-pair contributions into a per-pair output array and sum afterwards (no cross-thread running sum). Bench at n ∈ {2k, 50k, 500k, 2M} × |S| ∈ {5, 20, 50} against the current form and save to `_benchmarks/` alongside the existing `bench_relaxmrmr_3d_score.py` (which today benches only alpha=0 vs alpha=1, not backends). Do not quote a speedup until measured.

**Test:** `test_relaxmrmr_fused_pair_loop_matches_reference` — assert the fused score matches the current Python-loop form to 1e-12 across |S| ∈ {2..8} and varying `K_sel`. Note this must land *after* RO-1, or it fuses the wrong formula.

---

### RO-12 — the hinge/orth held-out gate regenerates an O(n) permutation and two full `(n, k)` design copies on every candidate  [P2]

**Where:** `_group2.py:197-199`, `:207-211`, `:220`, `:237-238`

**What:** `_heldout_incr_over_selected` is called once per hinge leg (`:271`) and once per orth-basis candidate (`:384`). Inside, each call:

- builds `np.random.default_rng(seed).permutation(n)` (`:197`) — an O(n) draw whose result is *identical* on every call (same seed, same n), plus the `va`/`tr` boolean masks (`:198-200`);
- builds `base = [np.ones(n), *_sel_value_cols]` (`:207`) and optionally appends `_sv`, `_sv * _sv` (`:211`);
- calls `_r2(base)` and `_r2([*base, leg])` (`:237-238`), each of which does `np.column_stack(design_cols)` (`:220`) — a full `(n, k)` float64 materialisation — then `A[tr]` and `A[va]` boolean-mask copies (`:221`, `:235`), i.e. **two more** full copies.

So per candidate: one O(n) RNG draw, two `(n, k)` column_stacks, four boolean-mask row gathers. At n=2M with k=120 selected columns each `column_stack` is ~1.9 GB, so a single candidate transiently holds ~5.7 GB and every candidate repeats it.

**Why it is wrong / costly:** the permutation, the masks, `yv`, `ss`, and the entire train/val-sliced base design are candidate-*invariant*. The sibling gate 200 lines below (`_group2.py:444-454`) hoists exactly these — the comment there even spells out why ("each call below re-used the SAME held-out target, its centered SS, and the SAME base design rows") — but the hinge gate it was modelled on was never updated. Against the repo's 100 GB-frame memory discipline, three full `(n, k)` materialisations per candidate is the finding even before the CPU cost.

**Fix:** hoist the permutation, `va`/`tr`, `yv`, `ss`, `_y_for_hinge_gate[tr]`, and the train/val-sliced base matrix into the enclosing block (they depend only on `n`, the seed, and `_sel_value_cols`, all fixed once `_sel_value_cols` is built at `:161-174`). Then reuse the same QR-insert incremental form as `_rp_r2` — this and RO-6 collapse into the one shared helper proposed there. Note the `[src, src^2]` smooth-curve terms (`:208-211`) *are* per-candidate, so the shared helper needs to accept 1–3 extra columns, not exactly 1.

**Test:** `test_hinge_gate_hoists_candidate_invariant_design` — spy on `np.random.default_rng` and `np.column_stack` across a fit with ≥10 hinge/orth candidates; assert the RNG is constructed once and `column_stack` call count does not scale with the candidate count.

---

### RO-13 — the masked-raw rescue's y re-binning swallows every exception at `debug` and falls back to a coarser target that changes the keep-rule verdict  [P2]

**Where:** `_group3.py:327-336`

**What:**

```python
try:
    _pcr_yv = y.values if hasattr(y, "values") else np.asarray(y)
    ...
    if (... int(np.unique(_pcr_yv).size) > max(20, 2 * int(np.unique(_pcr_y).size))):
        _pcr_nb = ...
        _pcr_y = np.ascontiguousarray(_pcr_qbin(...)).astype(np.int64)
except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
    logger.debug("mrmr: post-cluster-rescue y rebinning failed: %r", e, exc_info=True)
    pass
```

On failure `_pcr_y` keeps the screening `classes_y` (`:326`). The surrounding code exists precisely because that value is the *wrong* resolution — the sibling pass in `_group4.py:56-61` documents the same issue explicitly: *"the screening `classes_y` is frequently HEAVILY imbalanced on a skewed regression target (`y=(a**2)/b` puts ~89% of rows in one bin), which crushes the engineered anchor's MI and inflates a subsumed operand's apparent residual fraction."*

**Why it is wrong / costly:** the fallback is not neutral — it biases the keep-rule (`_pcr_keep`, `:401-407`) toward *rescuing* subsumed operands, i.e. in the direction that disables the check. Exactly the brief's item-4 shape. Visible only at `debug`.

**Fix:** log at `warning` (once per fit via `log_throttle`, which `_group1.py:379` already uses for the same class of probe), and name the consequence in the message. The `# nosec B110` marker and the redundant bare `pass` after a logged handler should go with it.

**Test:** `test_masked_raw_rescue_warns_when_y_rebinning_fails` — monkeypatch `_quantile_bin` to raise, assert a `WARNING` record naming the fallback is emitted.

---

### RO-14 — the monotone-twin drop breaks ties on `cached_MIs`, which is empty for every raw a *rescue* pass re-added, so rescued raws systematically lose  [P2]

**Where:** `_group4.py:268-274`, decision at `:304-309`

**What:** `_mt_relevance(_v)` returns `float(cached_MIs.get((_v,), 0.0))` — 0.0 on a cache miss. `cached_MIs` is populated by the greedy screen; a raw column that the screen *dropped* and one of the eight upstream protection/rescue passes (`_group1.py:435`, `:475`, `:505`; `_group2.py:276`, `:389`, `:531`; `_group3.py:148`, `:410`) then re-added may have no entry — and by construction these are the columns the screen did not evaluate to completion.

Consequently at `:304`, `_mt_relevance(_v) > _mt_relevance(_twin_of) + 1e-12` compares a real MI against a default 0.0, so a rescued raw always loses to a screen-selected twin, and two rescued twins both read 0.0 and the tie falls to selection order.

**Why it is wrong / costly:** the pass is supposed to keep *the higher-relevance* twin; instead it keeps the one the screen happened to evaluate, discarding exactly the column the rescue passes worked to recover. Silent — the log at `:313-318` reports the drop as a "raw decoy ... monotone re-encoding of a **higher-relevance** selected raw", which is unverified whenever the cache missed.

**Fix:** on a cache miss, compute the marginal MI directly (the `info_theory.mi` call at `_group4.py:200-201` is already imported in this file for the same purpose) rather than defaulting to 0.0; or, cheaper, skip the drop entirely when either side's relevance is unknown, which is the conservative direction.

**Test:** `test_monotone_twin_drop_keeps_higher_relevance_rescued_raw` — construct a fit where the rescued twin has strictly higher marginal MI than the screen-selected one and assert the rescued one survives. Today it is dropped.

---

### RO-15 — the adaptive-Fourier and missingness-indicator re-adds are unconditional, though their log lines claim held-out validation  [P2]

**Where:** `_group1.py:462-481` (adaptive Fourier) and `_group1.py:488-511` (missingness indicators)

**What:** the adaptive-Fourier block re-adds **every** name in `self._adaptive_fourier_features_` that resolves to a column index and is not already selected (`:467-473`) — there is no gate whatsoever. Its `logger.info` (`:477-481`) nonetheless reports *"re-added %d held-out-validated adaptive Fourier feature(s)"*. The validation it refers to happened in the generating detector, at a different time, against a different baseline (raw x, not the final selected design). The missingness block (`:494-503`) gates only on "the raw source survived the screen" — a membership test, not a usability check.

Every structurally identical sibling protection in this cluster does gate: hinge on `_heldout_incr_over_selected >= 0.003` (`_group2.py:271`), orth-basis on `>= 0.01` (`_group2.py:384`), raw floor-drop on `>= 0.005` (`_group2.py:526`), cat-FE on `>= 0.005` (`_group3.py:143`). The comment at `_group2.py:139-149` argues at length why the membership test alone is insufficient (*"on a MULTI-SIGNAL frame the SELECTED pair composite may already capture the source's structure better than a univariate kink"*) — the same argument applies verbatim to a Fourier leg and to a missingness indicator.

**Why it is wrong / costly:** these two passes can pad `support_` with columns already subsumed by a surviving composite, which is exactly the regression the hinge gate was built to prevent. And the log's "held-out-validated" wording will mislead anyone auditing a fit.

**Fix:** route both through the same `_heldout_incr_over_selected` probe the siblings use (it is in scope at `_group2`; after RO-6/RO-12's extraction it would be a shared helper importable by `_group1` too). If the unconditional behaviour is deliberate — a Fourier sin/cos pair genuinely has near-zero *individual* marginal usability and only works as a pair — say so in the comment and fix the log's wording.

**Test:** `test_adaptive_fourier_readd_rejects_subsumed_leg` — fit where a surviving composite fully spans the Fourier leg's signal; assert the leg is not re-added.

---

### RO-16 — the raw floor-drop protection materialises the selected design three times at full row count  [P2]

**Where:** `_group2.py:452-454`

**What:**
```python
_rp_base_mat = np.column_stack(_rp_base)   # (n, p) float64
_rp_base_tr  = _rp_base_mat[_rp_tr]        # boolean-mask copy, ~2n/3 rows
_rp_base_va  = _rp_base_mat[_rp_va]        # boolean-mask copy, ~n/3 rows
```
Three live `(n, p)`-scale float64 allocations; `_rp_base` itself already holds `p` separate `n`-length float64 arrays built at `:430-442`, so the peak is ~4 copies of the design. At n=2M, p=120 that is ~7.6 GB, and `_rp_base_mat` stays alive for the whole candidate loop (`:513-529`) because the closure captures it.

**Why it is wrong / costly:** against the repo's explicit 100 GB-frame discipline, and it is avoidable: `_rp_base_mat` is never used after the two slices are taken (it is referenced only at `:453-454`). Its memory could be released immediately.

**Fix:** `del _rp_base` and `del _rp_base_mat` after the two slices; better, build the train and val blocks directly with `np.column_stack([c[_rp_tr] for c in _rp_base])` so the full-height matrix never exists. Gate the whole probe on a byte-size ceiling the way the synergy screen at `_group1.py:183-192` already does via `fe_polars_exceeds` — the same reasoning applies here and this site has no such gate.

**Test:** `test_raw_protection_does_not_materialise_full_height_design` — spy on allocation shapes (or assert via `tracemalloc` peak) that no array of shape `(n, p)` is created when `n` is large.

---

### RO-17 — `len(selected_vars) >= 0` is vacuously true  [P3]

**Where:** `_group1.py:172`

**What:** `if int(_iac_max_order ...) >= 2 and len(selected_vars) >= 0:` — the second conjunct is always satisfied. Every sibling gate in this file uses `len(selected_vars)` or `len(selected_vars) > 0`.

**Why it is wrong / costly:** either dead text (harmless but misleading — a reader assumes a real guard) or a typo for `> 0` that would change behaviour. Given the n-way synergy screen deliberately seeds operands the greedy never selected, running it on an empty support may well be intended; if so the clause should just be deleted.

**Fix:** delete the conjunct, or make it `> 0` if the empty case is genuinely unsupported. Add a one-line comment saying which.

**Test:** `test_synergy_seeding_runs_with_empty_selection` — pins whichever semantics is chosen.

---

### RO-18 — `_rr_raw_is_relevant_given_engineered` ignores its first parameter  [P3]

**Where:** `_group1.py:306-332`, called at `:417`

**What:** the signature is `(_raw_idx, _eng_cols)` but the body reads only `_eng_cols` and the enclosing `_n_rows_rr`/`_RR_PROTECT_MAX_N`. `_raw_idx` is never referenced.

**Why it is wrong / costly:** the 20-line docstring describes a per-raw conditional-MI decision, so a reader expects `_raw_idx` to matter; the function is actually a pure function of `(bool(_eng_cols), n_rows)`. Dead parameter on a function whose docstring promises more than it does.

**Fix:** drop the parameter and shorten the docstring to what the code does, or implement the per-raw check the docstring describes.

**Test:** covered by the naming/doc audit; no behavioural test needed.

---

### RO-19 — one log gate reads `self.verbose` while every sibling reads the threaded `verbose` parameter  [P3]

**Where:** `_group1.py:144` (`if getattr(self, "verbose", 0):`) vs `:80`, `:227`, `:428`, `:437`, `:443`, `:477`, `:506` (all `if verbose:`)

**What:** `verbose` is an explicit keyword parameter of the group function (`:32`), derived from the fit body's own local. `self.verbose` is the constructor attribute. The fit may adjust the effective verbosity (e.g. suppressing logs on a nested/inner fit) without touching the attribute, in which case this one block logs when the other seven do not.

**Fix:** use `verbose`.

**Test:** `test_standalone_gate_prune_respects_threaded_verbose` — assert no INFO record when `verbose=0` is threaded but `self.verbose` is nonzero.

---

### RO-20 — the masked-raw rescue's de-dup filter is dead and the count it logs can disagree with what was added  [P3]

**Where:** `_group3.py:408`, `:410`, `:417`

**What:** the loop already skips any `_ridx in _pcr_sel_set` (`:382-383`), so `[i for i in _pcr_readd if i not in _pcr_sel_set]` (`:410`) can never filter anything. If it ever did, the log at `:417` would still report `len(_pcr_readd)` — the pre-filter count — and list `[cols[i] for i in _pcr_readd]`, over-reporting the re-add.

**Fix:** compute the filtered list once, extend from it, and log its length.

**Test:** not worth a dedicated test; fold into the rescue's existing test by asserting the logged count equals the observed `support_` delta.

---

### RO-21 — `_mt_relevance`'s exception handler is unreachable  [P3]

**Where:** `_group4.py:270-274`

**What:** the body is `float(cached_MIs.get((_v,), 0.0))`. `dict.get` with a default does not raise; `float()` on the default `0.0` does not raise. The only raising path would be a non-numeric cached value, which would be a different bug. The handler's `logger.debug("... treating as 0.0")` therefore documents a fallback that never happens — while the *real* silent-0.0 path (the cache miss) has no logging at all. See RO-14.

**Fix:** delete the handler; log the cache miss instead.

---

### RO-22 — a debug message names the wrong variable  [P3]

**Where:** `_group4.py:166-171`

**What:** the `try` coerces `y` (`:167-168`); the `except` logs *"classes_y coercion failed for the floor-drop rescue; falling back to a raw classes_y reshape"* (`:170`). `classes_y` is the fallback, not the thing that failed.

**Fix:** `"y coercion failed ...; falling back to classes_y"`.

---

### RO-23 — `set(self.feature_names_in_)` is rebuilt inside loops  [P3]

**Where:** `_group4.py:121` (inside the `if _dropped_redund_names:` branch, fine) and `_group4.py:159` (inside `for _ei in _kept_redund:`)

**What:** `:159` constructs a fresh set from the full raw feature-name array on every iteration of the loop over kept columns — O(p) per iteration, O(p·|kept|) total. The same set is already built at `:49` as `_raw_names_for_redund` and is in scope.

**Fix:** reuse `_raw_names_for_redund`.

---

### RO-24 — `_mi_x_pair_njit`'s log argument is written as a division by a division  [P3]

**Where:** `_relaxmrmr_3d.py:121`

**What:** `mi += p * math.log(p * n_f / (Px[i] * Pz[j] / n_f))`. Algebraically correct (it equals `v*n/(Px*Pz)`, verified), but it performs four FP operations where two suffice and introduces an extra rounding step, unlike the sibling `_cmi_xy_given_z_njit` (`:76`) which uses the clean `(p_xyz * p_z) / (p_xz * p_yz)` normalised form.

**Fix:** `math.log(v * n_f / (Px[i] * Pz[j]))` with `v` the raw count, or convert to probabilities first as the sibling does. Not a correctness bug — flagged for consistency and because this file's other kernel already shows the preferred form.

---

### RO-25 — a non-positive `alpha` silently disables the interaction term rather than being rejected  [P3]

**Where:** `_relaxmrmr_3d.py:197` (`if n_S >= 2 and alpha > 0.0:`), docstring `:29`, `:137`

**What:** the docstring says *"Default alpha = 1 matches Vinh 2016; higher alpha emphasises higher-order structure"* and describes the term as *signed*. A caller passing `alpha=-1` (a plausible "invert the correction" experiment) gets `inter = 0.0` with no signal, indistinguishable from `alpha=0`.

**Fix:** either allow negative alpha (drop the `> 0.0` guard; the arithmetic is already signed) or raise on it. Document which.

---

### RO-26 — the `_hybrid_orth_family_variants` package docstring justifies skipping the AST gate with reasoning the sibling package's docstring documents as false  [P3]

**Where:** `_hybrid_orth_family_variants/__init__.py:9-13` against `_friend_graph_and_redundancy/__init__.py:10-16`

**What:** the hybrid-orth docstring claims *"each group is a verbatim contiguous slice of the original single-function body, so no per-block free-variable re-analysis was needed: the whole original function only ever closed over the params below, so every slice's free variables are a subset by construction."* That reasoning is unsound — function-local `import`s and locals assigned in an earlier block are *not* free variables of the original function, so they are invisible to a free-variable analysis of the whole and break on slicing. The sibling package's docstring records exactly this happening here: *"a sibling split of `_hybrid_orth_family_variants` was caught doing [this] (Python's function-level, not block-level, scoping let 14 of 19 family blocks there implicitly share one early block's import in the original monolith)."*

**Why it is wrong / costly:** the incident was fixed (see Verified-clean below — the AST walk is clean today) but the docstring still advertises the unsound rule to whoever performs the next split of this package.

**Fix:** replace the justification with the sibling's: a per-submodule AST `Load`-Name audit was run and is the gate, not a subset-by-construction argument.

---

### RO-27 — the standalone-gate prune's warning logs the exception type but discards the message  [P3]

**Where:** `_group1.py:150-151`

**What:** `logger.warning("MRMR fast-search standalone-gate prune skipped (%s); continuing.", type(_sg_exc).__name__)` — every other broad handler in this file (`:89-93`, `:233-234`) also logs the exception itself.

**Fix:** add `, _sg_exc` and a `%s` for it.

---

### RO-28 — two small biases in the tree rescue's ranking: reversed-argsort tie order and NaN→0 fill  [P3]

**Where:** `_mrmr_tree_rescue.py:141`, `:116-117`

**What:** (a) `np.argsort(imp)[::-1]` reverses NumPy's stable *ascending* order, so among equal importances the rescue systematically prefers the *highest* column index — an arbitrary but reproducible bias. `np.argsort(-imp, kind="stable")` prefers the lowest index instead, which at least matches the frame's own order. (b) `Xnum[np.isnan(Xnum)] = 0.0` substitutes a value that is inside the observed range of most features, so LGBM splits on the imputed value as if it were real; the comment at `:119-127` warns about this for the non-numeric fallback path but the fast path fills silently. LightGBM handles NaN natively, so the fill is not required at all for the LGBM call.

**Fix:** (a) use a stable descending sort. (b) drop the NaN fill and let LightGBM's native missing handling apply; if a fill is kept for the coerce path, log the count there too (it already does, at `:122-126`) and use `np.nan`-preserving behaviour on the fast path.

**Test:** `test_tree_rescue_ties_broken_by_column_order` and `test_tree_rescue_does_not_impute_nan_to_zero`.

---

## Proposed tests (beyond the per-finding ones)

- `test_relaxmrmr_score_matches_closed_form_on_a_hand_computed_triple` — a 3-variable fixture small enough to compute all six MIs by hand; assert the score to 1e-12. The existing suite asserts only sign directions and finiteness, which is why RO-1 survived; this is the mutation-resistant replacement.
- `test_relaxmrmr_score_symmetric_under_selected_set_permutation` — shuffle `selected_cols`/`nbins_selected` together; the score must be invariant (it is a symmetric average over pairs). Would catch any future index-pairing regression in the `i<j` loop.
- `test_friend_graph_group_split_has_no_unbound_names` — the repo's documented gate, as an executable test: AST-walk every `Load`-context `Name` in all nine `_groupN.py` files, assert none is unbound. I ran this manually (0 findings); nothing in the suite pins it, so the next split has no backstop.
- `test_friend_graph_groups_are_order_dependent_only_as_documented` — call `_friend_graph_and_redundancy_passes` with the four groups invoked in a permuted order and assert `support_` changes, pinning that the sequential contract in the package docstring is real rather than incidental. A silent reordering today would be undetected.
- `test_every_readd_pass_is_idempotent` — run the whole section twice on the same state; `selected_vars` must not grow the second time. Nine passes append to `selected_vars` and each maintains its own `_sv_set`; there is no shared guard, and `_group3.py:410` already shows one pass hand-rolling a de-dup filter.
- `test_protection_passes_never_net_grow_support_beyond_screen_plus_k` — a coarse budget invariant across the eight re-add sites; none of them consults an aggregate cap today.
- `test_hybrid_orth_family_target_preparation_is_shared` — assert all nineteen family stages receive a target derived from `_y_np` by the same helper (covers RO-4 and RO-8 together, and prevents the ten copies from drifting again).
- `test_monotone_twin_drop_is_transitive_on_a_triple_of_twins` — three mutually-monotone raws: assert exactly one survives. The `break`-at-first-twin logic at `_group4.py:291-298` combined with the displacement branch at `:304-307` is not obviously transitive; I could not convince myself either way by reading, and nothing pins it.

## Prior-wave findings touching this cluster

The `-07-25` tracker's cluster-relevant items were checked against current source:

- **Double Miller–Madow bias correction (MM entropies plus a telescoped subtraction).** Does **not** reproduce in this cluster's files. `_relaxmrmr_3d.py` has *no* bias correction at all (RO-2 — the opposite defect), and the friend-graph/hybrid-orth groups delegate all MI estimation to `info_theory` / `permutation.mi_direct` / the per-family `_orthogonal_*_fe` modules rather than correcting inline. I found no site in the twelve assigned files applying a correction twice. The MM-corrected joint-MI path referenced at `_group1.py:158-161` is inside `detect_synergy_combos`, outside this cluster.
- **Sibling-split `NameError` on a moved function** (the `_hybrid_orth_family_variants` incident named in `_friend_graph_and_redundancy/__init__.py:10-16`). **Fixed and holding.** A full AST `Load`-Name walk over all nine `_groupN.py` files plus both `__init__.py` files, with correct nested-scope handling (params, comprehension targets, `except` bindings, walrus, `global`/`nonlocal`, lambdas, class bodies), reports **zero** unbound names. Residual risk is documented as RO-9 (two `locals()`/`dir()` guards that would silently change behaviour, not raise, under the next split) and RO-26 (the docstring still advertises the unsound justification).
- **`cuda` branch catching only `(ValueError, RuntimeError)` while `CudaAPIError` escapes.** Not applicable to this cluster: `grep` for `cuda`/`cupy`/`cuda.jit` across all four `_hybrid_orth_family_variants/_groupN.py`, all four `_friend_graph_and_redundancy/_groupN.py`, `_relaxmrmr_3d.py` and `_mrmr_tree_rescue.py` returns **no matches**. There is no GPU path in these files; GPU dispatch lives one level down in the per-family `_orthogonal_*_fe` modules and in `friend_graph_gpu.py` (the `gpu_backend` argument is merely forwarded at `_group1.py:72`), all outside this cluster's scope.

## Verified-clean

Checked and found genuinely fine — do not re-audit blind:

- **Unbound-name audit across the split, all nine `_groupN.py` files + both `__init__.py`.** Zero `Load`-context names that are not locally bound, imported, a builtin, or a legitimate closure reference. The scope walker handled nested `def`s, lambdas, comprehension targets and `except ... as` bindings correctly (an initial naive version produced 19 false positives from flattening nested scopes; those were all confirmed to be parameters of inner functions such as `_r2(design_cols)` and `_heldout_incr_over_selected(_leg_vals, _src_vals)`).
- **Re-export surfaces.** Each package's `__init__.py` imports exactly one symbol per group module and exposes exactly one public entry point. `_friend_graph_and_redundancy` is imported only by `_fit_impl_core.py:2243-2246`; `_hybrid_orth_family_variants` only by `_fe_stage_cascade_early_a.py:501-503`. Both call sites use the single facade function. **No symbol is defined in two group modules**, so the "same name, two definitions" hazard does not exist here.
- **Parameter threading through the four friend-graph groups.** All twenty keyword arguments are forwarded identically to each group (`__init__.py:80-175`), and the `(selected_vars, cols, data, nbins)` tuple is correctly rebound between groups — the growth paths that motivated the docstring's BUG FIX note (`_group2.py:68` `cols = [*cols, _hn]`, `_group2.py:75-83` `data`/`nbins` append, `_group3.py:262-264` the DCD swap rebind) all propagate correctly through the chain and out of the facade.
- **`_abs_corr_finite_njit`, used by the monotone-twin drop** (`_group4.py:275`, `:295`). Returns `|Pearson r|` (so anti-monotone twins are correctly caught) via a numerically-stable **two-pass** mean-then-centred-moment accumulation — the docstring records the catastrophic-cancellation incident that forced the two-pass form. `min_n=2` is the documented correct value for a corrcoef-replicating call site. Pearson-on-average-ranks is the standard Spearman definition, so the "Spearman rho" naming is accurate. No cancellation bug here.
- **`prune_by_friend_graph`'s reporting.** `_group1.py:82-85` logs `_fg.pruned`, which `friend_graph.py:600` sets to the *actually-removed* names after `protect_indices` (`:579`, `:599`) is applied — so the cluster-aggregate protection at `_group1.py:77` is correctly reflected in the log. I initially suspected over-reporting; it is correct.
- **`_mi_x_pair_njit`'s log argument** (`_relaxmrmr_3d.py:121`). Algebraically verified equal to `p·log(p/(p_x·p_z))`. Awkward form only (RO-24), not a bug.
- **`_nbins_syn` adaptive-binning formula** (`_group1.py:206`). `int((n / (mrpc * 5.0)) ** (1/order))` matches the comment's `floor((n / (5*min_rows_per_cell))^(1/order))` — multiplication commutes, and the `max(2, min(quantization_nbins, ...))` clamp at `:207` matches the documented `[2, quantization_nbins]` range. Verified against the comment's own worked example (n=2000, order=3 → 4 bins).
- **Synergy-screen memory gate** (`_group1.py:180-192`). Correctly refuses to `fe_to_pandas` a large polars frame and warns at `UserWarning` rather than silently skipping — the right shape, and the right log level. Worth copying to the un-gated sites named in RO-16.
- **No GPU/CUDA/cupy/njit code in any of the twelve files** except `_relaxmrmr_3d.py`'s three `@njit(nogil=True, cache=True)` kernels. The REJECT-gate grep (`@njit`, `parallel=True`/`prange`, `cuda.jit`, `cupy`, `KernelTuningCache`/`get_or_tune`) was run over all twelve; the only hits are those three `nogil` kernels, which is what makes RO-11 a valid perf finding rather than an already-optimal path.
- **`MRMRTreeRescued._get_param_names`** (`_mrmr_tree_rescue.py:64-69`). Correctly unions `MRMR._get_param_names()` with `_TREE_RESCUE_PARAMS` so sklearn `get_params`/`set_params`/`clone` round-trip despite the `*args`/`**kwargs` constructor. The `sorted(set(...))` result is deterministic. This is the right fix for the varargs-ctor introspection problem.
- **`_tree_rescue_should_fire`'s gate arithmetic** (`_mrmr_tree_rescue.py:72-85`). Mode parsing handles `False`/`None`/`"off"`/`"false"`/`"none"` and `True`/`"always"`/`"true"`; `p <= tree_rescue_min_p` short-circuits before the auto branch; `floor = max(min_features, ceil(min_ratio * p))` matches the docstring's stated rule exactly. No off-by-one.
