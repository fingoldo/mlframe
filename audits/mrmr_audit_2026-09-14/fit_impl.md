# fit_impl — mrmr_audit_2026-09-14

## Scope

| File | LOC | Read |
|---|---|---|
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py` | 2543 | head 1-1600 + tail 1600-2543 (full) |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_helpers.py` | 472 | full |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_finalise.py` | 651 | full |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_assign_support.py` | 726 | full |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_assign_support_tail.py` | 617 | full |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/__init__.py` | 26 | full |
| `src/mlframe/feature_selection/filters/_mrmr_degenerate.py` | 271 | full |
| `src/mlframe/feature_selection/filters/_mrmr_sis_screen.py` | 367 | full |
| `src/mlframe/feature_selection/filters/_mrmr_sis_apply.py` | 93 | full |
| `src/mlframe/feature_selection/filters/_mrmr_passthrough.py` | 95 | full |

Cross-read for index-space / contract verification (not in scope, not audited):
`_mrmr_stability_report.py:109-164`, `_mrmr_validate_transform.py:318-360`, `mrmr/_mrmr_class.py:3576-3595, 3702-3716`,
`_fe_stage_cascade_early_b.py` (roster registration).

**Scope note on the greedy loop itself.** The brief's "does the implemented score match the claimed mRMR/MIQ/MID/JMIM
variant, is redundancy averaged or summed, is the argmax invariant enforced" question is NOT answerable inside this
cluster: `_fit_impl_core.py:1766` delegates the entire greedy selection to `screen_predictors(...)`
(`mrmr_relevance_algo` / `mrmr_redundancy_algo` / `use_simple_mode` are passed straight through), which lives in the
`screen_confirm` cluster. What this cluster owns is everything BEFORE (candidate-pool construction, discretisation,
FE-append) and everything AFTER (the ~20 post-screen support-mutation passes). All findings below are in that surface.

## Findings

### FIT_IMPL-1 — stability-replay `selected_mask` is built by comparing feature_names_in_ indices against cols-space indices  [P1]
**Where:** `_mrmr_fit_impl/_assign_support.py:586-590` → `_mrmr_fit_impl/_helpers.py:287-291`
**What:** `_assign_support` calls `_build_stability_replay_state(..., selected_vars=selected_vars)`. By that point
`selected_vars` has been rebound to **feature_names_in_ index space** (`_fit_impl_core.py:2333`,
`selected_vars = original_indices`, where `original_indices` comes from `_fni_idx.get(col)` at 2313-2315). But
`_build_stability_replay_state` treats them as **cols-space** indices into the augmented screening matrix:

```python
sel_set = set(int(v) for v in np.asarray(selected_vars, dtype=np.intp).ravel())   # _helpers.py:287
cand_cols = [c for c in range(n_cols) if c != t_idx]                              # _helpers.py:289
selected_mask = np.array([c in sel_set for c in cand_cols], dtype=bool)           # _helpers.py:291
```

`cols` is `categorize_dataset`'s output: it carries the injected `targ_*` column(s), every engineered column, and
`categorize_dataset` reorders categorical columns (the fit body's own comment at `_fit_impl_core.py:2271` says so —
that reorder is exactly why the name-based remap at 2274/2312 exists).
**Why it is wrong / costly:** silent wrong result in a user-facing diagnostic. `selected_mask` feeds BOTH outputs of
`MRMR.selection_stability_report`: `n_selected = selected_mask.sum()` (`_mrmr_stability_report.py:112`) is the top-K cut
size used on **every** bootstrap replay, and `selected_features` (`:142`) is the reported point selection. When the two
spaces diverge the report names the wrong columns as selected and ranks against the wrong K — it does not raise, it
returns a plausible-looking table. The two spaces coincide only in the narrow case of an all-numeric frame with zero
engineered columns and the single target appended last; with any categorical column or any FE family on (≈30 are
default-ON) they diverge.
**Fix:** pass the pre-remap **cols-space** selection, or translate inside `_build_stability_replay_state`:
`sel_set = {name_to_col[str(self.feature_names_in_[v])] for v in selected_vars}` using the `name_to_col` map the
function already builds at `_helpers.py:305`.
**Test:** `test_stability_replay_mask_matches_support_names` — fit on a frame with a leading string categorical (forcing
the categorize reorder) plus one FE family on; assert
`set(np.asarray(m._stability_replay_state_["cand_names"])[m._stability_replay_state_["selected_mask"]])` equals the raw
names in `m.support_`, and that `selection_stability_report(as_text=False)["n_selected"] == len(m.support_)`.

### FIT_IMPL-2 — the SIS front gate narrows X before `feature_names_in_` is set, breaking the ndarray fit/transform round-trip  [P1]
**Where:** `mrmr/_mrmr_class.py:3706-3711` → `_mrmr_sis_apply.py:88-93` → `_fit_impl_core.py:1111-1112`
**What:** `_apply_sis_screen` returns `X[:, survivors]` and `fit` rebinds `X` to it *before* `_fit_impl` runs, so
`self.feature_names_in_` / `self.n_features_in_` are computed from the **survivor subset**
(`_fit_impl_core.py:1107-1112`). `self.sis_survivors_` is stored (`_mrmr_sis_apply.py:81`) but is read by **nothing**
(grep across `src/` + `tests/`: only the two assignment sites and two `hasattr` assertions in tests).
**Why it is wrong / costly:** `_mrmr_validate_transform.py:338-347` raises
`ValueError: X has {p} features, but MRMR is expecting {m} features as input` for any non-named-frame input whose width
differs from `n_features_in_`. The gate is a **fastest-default dispatch, not opt-in** (`sis_screen_threshold: int = 20000`,
`_mrmr_class.py:1227`), so `MRMR().fit(X_ndarray_100k_cols, y).transform(X_ndarray_100k_cols)` raises on the caller's own
training matrix. The pandas path survives only because the named-frame branch at `:346` skips the shape check and the
survivor names are present in the full frame — i.e. the contract holds by accident of container type, not by design.
**Fix:** either keep `n_features_in_` at the pre-screen width and apply `sis_survivors_` as a positional pre-slice at
the top of `transform`, or (simpler) record `self.sis_n_input_features_` as the sklearn `n_features_in_` and translate
`support_` back into pre-screen positions before it is frozen.
**Test:** `test_sis_screen_ndarray_fit_transform_roundtrip` — fit an `MRMR(sis_screen_threshold=200)` on a
`(n, 300)` **ndarray**, then `transform` the same array; assert it returns `(n, m.n_features_)` rather than raising
`ValueError`.

### FIT_IMPL-3 — the raw-signal-retention augmentation bypasses the pinned `factors_names_to_use` / `factors_to_use` search space  [P1]
**Where:** `_mrmr_fit_impl/_assign_support_tail.py:556-563` (vs the chokepoint at `_assign_support.py:399-415`)
**What:** `_assign_support.py:389-398` documents that the search-space restriction is enforced "ONCE here — the single
chokepoint right before `support_` is frozen", precisely because downstream re-add passes do not consult it. Every other
post-chokepoint re-add honours it explicitly: emit-both (`_assign_support.py:495-496`), usability raw retention
(`_assign_support_tail.py:323-324`), the empty-support rescue (`_finalise.py:118-129`). The raw-signal-retention
augmentation does **not**:

```python
_to_add = [i for i, _name, m in sorted(_raw_mi_aug, key=lambda kv: (-kv[2], kv[0]))
           if m > _floor_aug and i not in _selected_set and _name in _eng_tokens
           and _name not in _aug_excluded_names
           and _name not in _redund_dropped_names
           and not (_aug_large_n and _name in _surviving_eng_operands)]
if _to_add:
    selected_vars.extend(_to_add)
    self.support_ = np.array(selected_vars, dtype=np.int64)
```

`_allowed_raw_idx` is threaded into this function (signature line 42) and used 233 lines earlier at `:323`, but not here.
**Why it is wrong / costly:** a raw column the caller explicitly excluded from the candidate pool re-enters `support_`
whenever its name appears as a source token of a confirmed engineered recipe and its marginal MI clears the floor — the
exact leak the chokepoint's own comment says the design exists to prevent. It is also cache-persistent: the fit is
stored in `MRMR._FIT_CACHE` and replayed.
**Fix:** add `and (_allowed_raw_idx is None or int(i) in _allowed_raw_idx)` to the `_to_add` comprehension.
**Test:** `test_augmentation_honours_pinned_search_space` — fit with `factors_names_to_use=["good1"]` on a frame where
`good2` is an operand of a surviving engineered recipe and clears the relevance floor; assert
`set(m.feature_names_in_[m.support_]) <= {"good1"}`.

### FIT_IMPL-4 — the SIS survivor floor reads a non-existent attribute, so `k_target` is always `None`  [P2]
**Where:** `_mrmr_sis_apply.py:59-63`
**What:** `k_target = getattr(self, "n_features", None)`. `MRMR.__init__` has **no** `n_features` parameter
(grep of `mrmr/_mrmr_class.py` for `n_features` outside `n_features_in_` / `n_features_` / `*_selection_rule` returns only
docstring text and `fe_gbm_seeder_min_features`); sklearn's `BaseEstimator` does not define one either. The `try/except`
at 61-63 only guards a bad cast, not a missing attribute.
**Why it is wrong / costly:** `k_target` is always `None`, so `survivor_count`'s floor
(`_mrmr_sis_screen.py:167`, `floor = max(20 * int(k_target or 0), 1000)`) is permanently `1000`, and the docstring's
"feeds the `20*k_target` survivor floor" contract is dead. On a caller that genuinely wants e.g. 200 final features the
intended 4000-survivor floor silently degrades to 1000 — the "never starve the downstream pool" guarantee the design doc
specifies is not in force. Not a crash and not obviously wrong at a glance, which is why it survived.
**Fix:** read the attribute that actually exists (`min_features_fallback`, or a new explicit `n_features_to_select`), or
delete the parameter and the floor branch rather than leave a dead knob. A `getattr` for an attribute no ctor defines
should never be the silent-default form.
**Test:** `test_sis_k_target_resolves_from_a_real_param` — monkeypatch `sis_screen` to capture its `k_target` kwarg,
fit an MRMR whose requested feature count is 200, assert the captured value is 200 (not `None`).

### FIT_IMPL-5 — the empty-support rescue tokenises the recipe `repr()` instead of its column name  [P2]
**Where:** `_mrmr_fit_impl/_finalise.py:101-107`
**What:**
```python
for _en in getattr(self, "_engineered_recipes_", {}) or {}:
    for _tok in _RESC_TOK_SPLIT.split(str(_en)):
```
`self._engineered_recipes_` is a **list of `EngineeredRecipe` objects** (built at `_fit_impl_core.py:2305/2320`), not a
name→recipe dict. `str(_en)` is therefore the full dataclass `repr`, not the column name. The module already imports
`_engineered_recipe_name` for exactly this purpose (`_finalise.py:32`) and uses it correctly 100 lines later at
`_finalise.py:205-206`; `_assign_support.py:97` and `:132` do the same. The `{}` default and `or {}` also signal the
author believed this was a dict.
**Why it is wrong / costly:** the token set is a strict **superset** of the intended operand names — a repr carries
`kind=`, `extra={...}` keys/values and every nested `src_names` entry. Any raw column whose name coincides with a repr
token (a short name like `a`, `b`, `x1`, or any string appearing inside `extra`) is added to `_rescue_redund_dropped`
(`:107`) and is then **skipped** from the rescue candidate list at `:131-132`. The rescue exists precisely for
"0 raw survived despite recoverable signal"; over-exclusion here makes it return fewer features (or fall through to the
never-empty single-column branch at `:280`). Silent under-selection, no log.
**Fix:** `for _en in (_engineered_recipe_name(_r) for _r in (getattr(self, "_engineered_recipes_", None) or [])):` —
the same idiom already used at `:205-206`.
**Test:** `test_rescue_operand_exclusion_uses_recipe_name_not_repr` — construct a fitted MRMR whose
`_engineered_recipes_` holds one recipe with `src_names=("a",)` and `extra={"note": "b"}`, force the empty-raw rescue,
and assert `b` is still eligible (i.e. rescued when it clears the floor) while `a` is excluded.

### FIT_IMPL-6 — a failed SIS scoring block silently zero-scores up to `chunk_width` columns, and the warning is throttled  [P2]
**Where:** `_mrmr_sis_screen.py:303-311`
**What:** each column block's MI and propensity calls are individually wrapped:
```python
except Exception as exc:  # never let one block kill the whole screen
    log_throttle(logger, "sis_screen_mi_block_failed", logging.WARNING, "... scored 0", ...)
```
`mi`/`prop` were pre-allocated as zeros (`:287-288`), so the block keeps its `0.0` scores.
**Why it is wrong / costly:** this is the repo's own twice-shipped bug class (CLAUDE.md, 2026-08-02 × 2) in its
substituted-value form. `0.0` is **not** neutral here: `fuse_scores` z-scores each channel and takes the max
(`:140`), so a zero block lands at or below the median and is cut by `survivor_count`. A transient failure on one
`_mi_classif_batch` call therefore **permanently removes up to `chunk_width` (default 256-8192) candidate columns from
the entire fit** — they never reach MRMR at all. The `log_throttle` key is per-message, so a screen that fails on 40
blocks reports it once, and the run looks clean. There is no post-hoc count of failed blocks and no abort threshold.
**Fix:** count failed blocks; if any block failed, either fall back to the full-width path (the caller at
`_mrmr_class.py:3712` already has that fallback for a whole-screen raise) or force the failed columns into the survivor
set rather than scoring them 0. At minimum log an unthrottled end-of-screen summary naming the failed column ranges.
**Test:** `test_sis_block_failure_does_not_silently_drop_columns` — monkeypatch `_mi_classif_batch` to raise on the
second block only; assert the failed block's columns are either all retained or the screen reports a non-zero failure
count on the estimator, and that the planted signal in that block still reaches `support_`.

### FIT_IMPL-7 — the default-ON degenerate-column audit materialises ~2× the frame in float64, gated on `p` only  [P2]
**Where:** `_mrmr_degenerate.py:213-217, 242-248, 63-69`; call site `mrmr/_mrmr_class.py:3586-3591` (unconditional)
**What:** three unbounded-in-`n` allocations:
1. `:213-217` — every numeric column is kept as a **float64 copy** (`v = values.astype(np.float64)`) plus a bool
   `finite` mask, held in `numeric_cols` for the whole scan: ~9 bytes per element of the numeric frame.
2. `:242-248` — `M = np.empty((len(live), n_rows), dtype=np.float64)` plus a further `col = v.copy()` per column: a
   second full float64 image of the same data.
3. `:63-69` — the polars branch of `_column_arrays` calls `X.to_numpy()`, materialising the **entire** frame as one
   dense ndarray before yielding column views.

The only guard is `max_collinearity_cols = 4000` (`:176, 227`), which bounds **p** and says nothing about `n`. The
docstring at `:188-193` justifies it as "a dense `(p, n)` Gram-matrix input — unbounded on a genuinely wide raw frame",
i.e. the `n` axis was simply not considered.
**Why it is wrong / costly:** CLAUDE.md's Memory/RAM discipline is explicit that frames can be 100+ GB and that eager
conversion must be gated on **byte size**. At n=10M × p=500 numeric columns this diagnostic — which by its own module
docstring "does NOT remove columns or alter which features MRMR selects" — allocates ~40 GB in `numeric_cols` and another
~20 GB in `M`. It is wrapped in a bare `try/except` at the call site (`_mrmr_class.py:3587-3590`) that debug-logs, so an
`MemoryError` here degrades to a silent missing diagnostic; a paging stall or an OOM that lands in a *sibling* allocation
does not.
**Fix:** gate on `n_rows * len(live) * 8` bytes (the repo's ~2 GB eager threshold) in addition to `max_collinearity_cols`,
row-subsample for the correlation pass (a few thousand rows is ample for detecting `|corr| == 1`), drop the float64 copy
in `numeric_cols` (recompute per column inside the M build), and replace the polars `X.to_numpy()` with per-column
`X.get_column(name).to_numpy()`.
**Test:** `test_degenerate_audit_bytes_gated_on_n_not_just_p` — call `audit_degenerate_columns` on a tall frame whose
`(p, n)` float64 image would exceed the gate; assert the collinearity pass is skipped (info-logged) and
`all_nan`/`constant`/`duplicate` reasons are still returned.

### FIT_IMPL-8 — the group-aware final demotion drops engineered recipes without pruning the public engineered rosters  [P2]
**Where:** `_mrmr_fit_impl/_finalise.py:553-561` vs `_mrmr_fit_impl/_assign_support.py:595-623`
**What:** `_assign_support.py:606-623` reconciles all 28 public rosters (`hybrid_orth_features_`, `kfold_te_features_`,
`wavelet_features_`, …) against `self._engineered_features_`, with the comment "Runs AFTER the additional_rfecv rescue …
so `_engineered_features_` is final here." It is not: the group-aware demotion in `_finalise.py:553-556` later removes
names from **both** `_engineered_recipes_` and `_engineered_features_`, but touches none of the rosters.
**Why it is wrong / costly:** under `group_aware_mi=True` with a between-group-only leak feature, the demoted column
disappears from `get_feature_names_out()` / `transform` output but remains in the user-facing
`m.hybrid_orth_features_` (and whichever family roster produced it). The roster is documented as "engineered columns
that actually survived into the output" — it now over-reports. Same class as the prior wave's FIT_IMPL-1 (which was
about `mrmr_gains_` and *is* fixed by the second `_align_mrmr_gains(self)` at `_finalise.py:577`); the roster half of
the same "last mutation has no reconciliation after it" problem was not closed.
**Fix:** re-run the roster intersection (factor `_assign_support.py:606-623` into a helper) immediately after the
demotion, next to the existing `_align_mrmr_gains(self)` at `:577`.
**Test:** `test_group_demotion_prunes_public_rosters` — same fixture as the prior wave's proposed gains test; assert
`set(m.hybrid_orth_features_) <= set(m.get_feature_names_out())` after a demotion fires.

### FIT_IMPL-9 — the SIS redundancy dedup gathers an `(n, m)` float64 sub-matrix whose size is unbounded in `n`  [P2]
**Where:** `_mrmr_sis_screen.py:347`
**What:** `surv_df = pd.DataFrame(np.asarray(Xarr[:, survivors], dtype=np.float64), columns=...)`. The comment at
`:341-346` calls this "the SMALL (n x m) survivor sub-matrix (m … a few thousand)" and justifies the float64 upcast as
load-bearing for the `|corr|` threshold — the upcast reasoning is correct, the "small" claim is not: only `m` is
bounded (`_ram_cap_survivors` caps it by free RAM at `:183-187` — but that cap is computed for an **int16** `(n, m)`
matrix at `n*m*2` bytes, while this allocation is `n*m*8`, a **4× underestimate** of the actual peak).
**Why it is wrong / costly:** at the module's own design point (`n=4000, p=100k`, `m≈2000`) this is only 64 MB, so it
never showed up; at `n=1M, m=2000` it is 16 GB, and the RAM cap that was supposed to bound it budgeted 4 GB. The
default `sis_dedup_corr_thr=0.92` means this runs on every gated fit.
**Fix:** either make `_ram_cap_survivors` budget the float64 dedup gather (`n*m*8`) rather than the int16 Gate-B pool,
or row-subsample the dedup correlation (the threshold is 0.92 — a few tens of thousands of rows settle it), or chunk the
gather. `corr_clusters` needs a DataFrame but not the full `n`.
**Test:** `test_sis_dedup_gather_respects_the_ram_cap` — assert the dedup gather's byte size is `<= _ram_cap_survivors`'s
own budget for the (n, m) it was given, across a tall/narrow and a short/wide shape.

### FIT_IMPL-10 — `_is_constant` reports an all-infinite float column as neither constant nor all-NaN  [P3]
**Where:** `_mrmr_degenerate.py:95-99` (and `_is_all_nan` at `:82`)
**What:**
```python
finite = values[~np.isnan(values)]      # keeps +inf / -inf despite the name
...
return bool(np.ptp(finite) == 0)
```
For an all-`+inf` column, `_is_all_nan` is `False` (inf is not NaN), `finite` is the full inf array, and
`np.ptp([inf, inf])` is `inf - inf == nan`, so `nan == 0` is `False`.
**Why it is wrong / costly:** the column is genuinely zero-variance but is reported as non-degenerate, so the
diagnostic's `constant` reason misses it (it then falls through to duplicate/collinear, where the standardisation at
`:252-256` produces a NaN `std` and `good[j]` is False, so nothing is reported at all). Purely diagnostic — no selection
impact — hence P3. The local's name (`finite`) actively misleads about what it holds.
**Fix:** `finite = values[np.isfinite(values)]` and treat an empty result as `all_nan`-adjacent (or add an explicit
`all_inf` reason); rename the local.
**Test:** `test_degenerate_audit_flags_all_inf_column` — assert an all-`inf` column gets a non-empty reason.

### FIT_IMPL-11 — three `"<name>" in dir()` guards are dead: the names are function parameters  [P3]
**Where:** `_assign_support.py:252` (`"cached_MIs" in dir()`), `:422` (`"data" in dir()`), `:434-435`
(`"cached_MIs"`, `"cols"`); `_assign_support_tail.py:591` (`"data"`), `:599` (`"cols"`)
**What:** `dir()` with no argument returns the local namespace. `data`, `cols` and `cached_MIs` are all keyword-only
**parameters** of these functions, so each guard is unconditionally `True`.
**Why it is wrong / costly:** no behaviour change, but the guards are survivors of the pre-split monolith (where the
names might legitimately have been unbound) and now falsely advertise a defensive branch that cannot fire; a reader
paying attention to them wastes time, and `"x" in dir()` is a fragile idiom that would silently start returning `False`
again under any future rename.
**Fix:** delete the `in dir()` conjuncts; keep the `isinstance(cached_MIs, dict)` check at `:252`/`:434`, which is real.
**Test:** none warranted (dead-branch removal); covered by existing support-assignment tests.

### FIT_IMPL-12 — `_align_mrmr_gains`'s comment claims a descending-sort trim; the code slices the head  [P3]
**Where:** `_fit_impl_core.py:61-65` vs `_finalise.py:467-469`
**What:** the call-site comment says "keep the top screening gains (**descending** — what the UAED elbow already
consumed) and pad any FE tail with 0.0". The implementation is a positional head slice:
`self.mrmr_gains_ = _g[:_nf_final]`. The greedy log is in **selection order**, which is descending in gain only if the
greedy never re-ranked — not a property this cluster can assert (the redundancy term can make a later pick's gain
exceed an earlier one's).
**Why it is wrong / costly:** the contract `len(mrmr_gains_) == n_features_` holds either way, so no live bug; but when
the final count is SHORTER than the greedy log (degenerate trim / p>=n cap / UAED elbow) the *retained* gains are the
first-selected ones, not the largest ones, and they are not re-paired with `support_` — a caller zipping
`support_` with `mrmr_gains_` after a p>=n cap (which re-**sorts** `selected_vars` by relevance at
`_assign_support.py:444`) gets a mis-paired table.
**Fix:** either make the comment match (say "keep the first `n_features_` greedy-log entries"), or — better — re-pair
the gains with the final `support_` order wherever a pass re-sorts it.
**Test:** `test_mrmr_gains_pair_with_support_order_after_pgn_cap` — fit in the p>=n regime, assert that for each
`i`, `mrmr_gains_[i]` is the gain recorded for `feature_names_in_[support_[i]]` in `_predictors_log_`.

### FIT_IMPL-13 — group-demotion threshold is `<= 0.0` while every comment says "EXACTLY zero"  [P3]
**Where:** `_finalise.py:551` (`if _grp_mi_f == _grp_mi_f and _grp_mi_f <= 0.0:`) vs the comments at `:499-500` ("EXACTLY
zero … a column with any real, however small, within-group signal survives") and the log text at `:564`.
**What:** `<= 0.0` also demotes a **negative** group-MI, which a debiased/permutation-corrected estimator can legitimately
produce for a weak-but-real signal.
**Why it is wrong / costly:** widens the demotion beyond the documented contract in exactly the low-signal regime where
the estimator noise floor lives. Whether `group_relevance_mi` can return negative values was not verified in this cluster
— **unverified**; settled by grepping `info_theory/_group_mi.py` for a `max(0, ...)` clamp on the return value.
**Fix:** if the estimator can go negative, decide deliberately (`== 0.0` per the comment, or keep `<= 0.0` and fix the
comment + log wording). Do not leave code and comment disagreeing on a demotion threshold.
**Test:** `test_group_demotion_threshold_matches_documented_contract` — feed a stubbed `group_relevance_mi` returning
`-1e-12` and assert the documented behaviour.

### FIT_IMPL-14 — `run_additional_rfecv` computes `n_unexplored` across two different index spaces  [P3]
**Where:** `_fit_impl_core.py:2363`
**What:** `n_unexplored = X.shape[1] - len(selected_vars)`. At this point `X` is the **working frame** (passthrough
columns removed at `:401`, engineered columns appended by ~30 FE families) while `selected_vars` indexes
`feature_names_in_` (raw user columns, passthrough included, engineered excluded — set at `:1103-1112`). The comment 65
lines later at `:2428-2430` correctly flags this exact mismatch for the *name* lookup, but the count above it was not
fixed.
**Why it is wrong / costly:** only gates whether the rescue runs (`if n_unexplored > 0`) and a log line, and the actual
pool is built correctly by name at `:2462`, so no wrong selection. But the number is meaningless and the guard can be
satisfied by engineered columns that the pool then excludes, running an RFECV over an empty `temp_columns`.
**Fix:** `n_unexplored = len(temp_columns)` — move the guard below the pool construction.
**Test:** `test_rfecv_rescue_skipped_when_pool_is_empty` — a fit where every raw column is selected but engineered
columns exist; assert `RFECV` is not constructed.

### FIT_IMPL-15 — the `_eng_drop` recipe-dict cleanup is 30 copy-pasted 3-line blocks; the sibling pass already has the loop form  [P3]
**Where:** `_fit_impl_core.py:810-920` (30 near-identical `for _c in list(_X_pre_recipes.keys()): if _c in _eng_drop: _X_pre_recipes.pop(...)` blocks) and `:777-809` (20 near-identical roster-filter lines)
**What:** the Layer-91 unified-gate pass immediately below solves the identical problem in 6 lines by iterating a tuple
of the dicts (`:996-1027`) and a tuple of the roster attribute names (`:969-990`). The tuple of all recipe dicts is even
already built for the dependency-closure guard at `:744-767` — and is then **not reused** by the 30 blocks that follow.
**Why it is wrong / costly:** ~140 lines of duplicated logic that must be edited in lockstep every time an FE family is
added. It has already drifted: `_all_pre_recipe_dicts` (`:744-767`) omits `_temporal_agg_pre_recipes`… no — it includes
it at `:760`; but the roster list at `:777-809` omits `modular_features_`, `group_distance_features_`,
`rare_category_features_`, `conditional_residual_features_`, `conditional_dispersion_features_`, `wavelet_features_`,
`rankgauss_features_`, `temporal_agg_features_` — all eight of which the *unified-gate* sibling at `:969-990` does
prune. A column dropped by the Spearman dedup therefore stays in those eight rosters until the later reconciliation at
`_assign_support.py:606-623` catches it (it does, so no user-visible leak today — but the divergence is invisible and
one reconciliation pass away from mattering).
**Fix:** reuse `_all_pre_recipe_dicts` for the pop cascade and hoist the roster-name tuple to a module constant shared by
both passes and by `_assign_support.py:607-620`.
**Test:** `test_all_engineered_rosters_are_pruned_by_every_drop_pass` — a meta-test asserting the three roster tuples
(`_fit_impl_core.py` dedup, `_fit_impl_core.py` unified gate, `_assign_support.py` reconciliation) are the same set.

### FIT_IMPL-16 — `_fit_impl_core.py` is 2543 LOC (2.5× the budget) and the file's own comments carry stale line numbers  [P3]
**Where:** `_fit_impl_core.py` (2543 LOC); stale references at `_fit_impl_core.py:235` ("line 144 below"), `:2155`
("line ~5085"), `_assign_support.py:659` ("line 1020+"), `_assign_support_tail.py:229` ("~line 8527"), `:235`
("~line 8498-8514"), `:339` ("~line 7915")
**What:** the prior wave's FIT_IMPL-2 (audit/process metadata in comments) is **still open** and was partly made worse
by the Tier-F splits: `grep -cE "Wave [0-9]|loop iter [0-9]|[0-9]{4}-[0-9]{2}-[0-9]{2}|Layer [0-9]|A1#[0-9]|BUG[0-9]|iter[0-9]{2}|FIT_IMPL_"`
gives 59 / 5 / 2 / 3 / 4 / 0 / 1 hits across `_fit_impl_core.py`, `_assign_support.py`, `_assign_support_tail.py`,
`_finalise.py`, `_helpers.py`, `_mrmr_degenerate.py`, `_mrmr_sis_screen.py`. Six comments additionally cite line numbers
from the **pre-split 10056-line monolith** that point nowhere in the current files.
**Why it is wrong / costly:** CLAUDE.md's Comment style rule bans date stamps / Layer-N / wave markers / finding IDs
outright. Stale line numbers are worse than no reference: `_assign_support_tail.py:229-236` builds a load-bearing
argument ("this read happens AFTER it, so the freshly repopulated attribute is authoritative") on line numbers a reader
cannot check. The 1000-LOC budget is also breached by `_fit_impl_core.py` — the `while True` screen loop
(`:1741-2160`) and the post-screen tail (`:2162-2543`) are both clean carve candidates, and the ~140 duplicated lines
from FIT_IMPL-15 would remove ~5% on their own.
**Fix:** strip the marker prefixes (keep the WHY), replace every line-number citation with a function/section name, and
carve the screen loop into a `_screen_fe_loop.py` sibling.
**Test:** the existing `test_no_file_over_1k_loc.py` backstop already covers the LOC half; add
`test_no_stale_line_number_references_in_mrmr_fit_impl` asserting no `line ~?\d{3,}` occurrences.

### FIT_IMPL-17 — the SIS screen's relevance statistic is deliberately inconsistent with the main loop's, and nothing tests the consequence  [P3 / design]
**Where:** `_mrmr_sis_screen.py:282-285` (the CAVEAT block) and `_mrmr_sis_apply.py:65-72`
**What:** the screen scores the MI channel on **raw columns, fixed quantile nbins=10, quantile-binned y**, while
`screen_predictors` scores on **`categorize_dataset`'s codes (default supervised MDLP)**. Both files explicitly document
that the two MI values differ — `_mrmr_sis_apply.py:69-71` says substituting the screen's MI into `cached_MIs`
"would CHANGE selection", which is the same statement read from the other direction: the screen's ranking is not the
main loop's ranking. The screen also permanently discards everything below the cut.
**Why it is wrong / costly:** this is an honest, documented tradeoff (the whole point of the gate is to avoid the
expensive supervised pass), not a bug — but the consequence "a feature the main loop would have selected can be cut by a
statistic the main loop disagrees with" is untested. The `max`-of-z-scores fusion mitigates it for interaction operands;
nothing bounds it for a main effect whose MDLP binning finds signal that a fixed 10-bin quantile does not (exactly the
regime CLAUDE.md's "adaptive_nbins_large_n_reg" campaign found MDLP winning on classification).
**Fix:** none required. Add a recall test so a future binning change cannot silently degrade it.
**Test:** `test_sis_screen_retains_features_mdlp_would_rank_top` — plant a signal whose MDLP MI is high but whose
10-bin quantile MI is near the noise floor (a sharp threshold effect inside one quantile bin); assert it survives the
screen at the default `sis_screen_threshold`/`dedup_corr_thr`.

## Proposed tests (beyond the per-finding ones)

1. `test_support_and_n_features_agree_after_every_mutation_pass` — parametrised over the passes that rewrite
   `self.support_` (`_assign_support.py:556`, `_assign_support_tail.py:329/463/563/610`, `_finalise.py:303/450`):
   after fit assert `len(m.support_) + len(m._engineered_recipes_) == m.n_features_ == len(m.get_feature_names_out()) == m.transform(X).shape[1]`
   **and** `len(m.mrmr_gains_) == m.n_features_`. Today only the gains half is pinned (`TestSupportGainsAlignment`); the
   `transform().shape[1]` leg is what would have caught the UAED lockstep desync the comment at `_finalise.py:439-444`
   describes.
2. `test_selected_vars_index_space_is_documented_at_every_handoff` — a source-level invariant is the wrong tool here;
   instead, a behavioural test: fit with a leading string categorical (forcing the `categorize_dataset` reorder) and
   assert that every consumer that receives `selected_vars` post-`original_indices` (`_build_stability_replay_state`,
   `_assign_support_tail`'s `_post_sel_raw_names`, the `_rr_sel_names` build) reports the **same** name set as
   `m.feature_names_in_[m.support_]`. This is the generalisation of FIT_IMPL-1 and would catch the next instance.
3. `test_pgn_cap_is_not_exceeded_after_any_retention_pass` — p>=n fixture with real leftover linear-usable raw signal
   (so both the usability raw retention and the augmentation fire); assert
   `len(m.support_) + len(m._engineered_recipes_) <= max(20, p // 3)`. The cap is applied twice
   (`_assign_support.py:424-449`, `_assign_support_tail.py:593-617`) with different `cached_MIs` sources (local vs
   `self.`), which is exactly the shape of a drift bug.
4. `test_fit_cache_replay_is_not_hit_when_only_groups_differ` — already reasoned correct by the prior wave
   (`_fit_impl_core.py:261-263`); pin it, since the `_groups_sig` fold is conditional on `group_aware_mi` and a future
   default flip would silently widen the key's blind spot.
5. `test_nullable_densify_threshold_switches_to_per_column` — `_fit_impl_core.py:423-427` branches on
   `len(X) * len(_nullable_num) * 8 <= 2 GiB`; assert both branches produce an identical frame (the per-column
   `assign`-in-a-loop path is O(n_cols) frame rebuilds and is never exercised by the suite at a size that reaches it).
6. `test_passthrough_detector_samples_are_bounded` — `_mrmr_passthrough.py:61,72` uses `step = max(1, n // 50)` and
   `s.iloc[::step]`, which yields ~50 rows for `n > 50` but the **whole column** for `n <= 50`; assert the cell-inspect
   loop never touches more than `_SAMPLE_ROWS + 1` cells regardless of `n`, and that a 60%-vector / 40%-scalar column is
   classified as an embedding column (the `n_vec * 2 >= n_seen` majority rule at `:89` is untested at the boundary).

## Prior-wave findings touching this cluster

| ID | Status in current source |
|---|---|
| `FIT_IMPL-1` (2026-07-25, P1 — `mrmr_gains_` not re-aligned after the group-aware demotion) | **FIXED.** The trim/pad is now a named helper `_align_mrmr_gains` (`_fit_impl_core.py:51-67`) called twice in `_finalise_fs_results`: once at `:471` and again at `:577`, the second call explicitly placed after the demotion with the comment "the group-aware demotion just above is the LAST `n_features_` mutation". This is exactly the prior wave's proposal #2. Verified: no `n_features_` write exists between `:577` and `return self`. |
| `FIT_IMPL-2` (2026-07-25, P3 — audit/process metadata in comments) | **STILL OPEN**, see FIT_IMPL-16 above. 74 marker hits across the cluster's five `_mrmr_fit_impl` files, plus six now-stale pre-split line-number citations introduced by the Tier-F carve. |
| `FIT_IMPL_B-1` (2026-07-22, LOC exemption) | **PARTIALLY ADDRESSED.** The monolith went 10056 → 2543 LOC via the `_assign_support` / `_assign_support_tail` / `_finalise` / cascade-sibling carves (prior-wave proposal #3, "carve the post-selection tail"). Still 2.5× the 1k budget. |
| `FIT_IMPL_B-2` (p>=n cap re-application) | **HOLDS.** Both caps present (`_assign_support.py:417-449`, `_assign_support_tail.py:584-617`); the second is still the last raw-selection mutation (only the UAED elbow follows, and it only shrinks). The `# FIT_IMPL_B-2 fix` marker comment is gone (good). |
| `_finalise_empty_support_fallback` "no bug found" (2026-07-25 non-finding) | **SUPERSEDED** by FIT_IMPL-5 — the prior wave audited the index-space translation and the gate polarity (both correct) but did not check that `_engineered_recipes_` is a list of objects rather than a name dict at `:101`. |

## Verified-clean

- **Fit-cache key + lock discipline** (`_fit_impl_core.py:246-292`, `_finalise.py:603-649`). Lookup + `move_to_end` +
  `_replay_fitted_state` are inside one `_MRMR_FIT_CACHE_LOCK` section; the store side holds the same lock across
  `setdefault` (first-writer-wins) / `move_to_end` / LRU `popitem` / byte-cap eviction. `fit_cache_max=0` clears rather
  than silently restoring 4; `None` folds to 4; `_skip_fit_cache` skips only this instance. Publishing `self` only after
  every post-fit mutation (the documented fix for the `vars()` torn-read `RuntimeError`) is genuinely the last statement.
- **In-object skip signature** (`_fit_impl_core.py:178-239`). Folds X content hash, y content hash, column-name tuple,
  shapes, and the **pre-override** ctor-params snapshot (`_pre_fit_ctor_params_snapshot_`, avoiding the transient-override
  trap) — and refreshes the params slot post-fit at `_finalise.py:403-408`. Empty X hash falls through to a full fit.
  Every failure path substitutes a unique sentinel (`object()`), which forces a miss — the conservative direction.
- **`_mrmr_instance_state_size_bytes`** (`_helpers.py:171-223`). `vars(instance).copy()` (C-level, no iterator protocol)
  rather than `list(vars(...).values())` — the reasoning in the comment is correct and the failure it prevents is real.
- **`fe_decide_on_subsample`** (`_helpers.py:341-462`). Closed-form-only boundary documented; `shared_subsample_idx`
  bounds-checked (`ndim == 1`, `0 < n_idx < n`, `max() < n`); empty-result treated as a valid decision (no wasted full-n
  re-run); partial-recipe-coverage and replay errors fall back to the full-data decision **with a WARNING**, not a debug
  line. No whole-frame copy on any path.
- **`survivor_count` / `fuse_scores` / `_zscore`** (`_mrmr_sis_screen.py:124-180`). No cancellation-prone form:
  `_zscore` uses `np.std` (two-pass, stable), not `E[x²] − E[x]²`; the MAD scale uses the standard 1.4826 factor; zero
  spread returns zeros rather than NaN. `m = np.clip(m, 1, p)` and `target_survivors` clipping both guarantee the screen
  **cannot** return an empty survivor set — the brief's "can screening drop ALL features?" question is a No.
- **SIS determinism.** No RNG anywhere in `_mrmr_sis_screen.py`; block order is ascending; the top-m cut is
  `np.lexsort((np.arange(p), -fused))` (explicit ascending-index tie-break, not `argsort`'s implementation-defined
  order); survivors re-sorted ascending. The dedup representative is `max(mem, key=lambda nm: fused[...])` over a
  `members` mapping — ties there are broken by `max`'s first-wins over `mem`'s iteration order, which is
  `corr_clusters`-owned (out of scope, noted).
- **SIS y-encoding chain** (`_mrmr_sis_screen.py:257-281`). The four-way branch (nominal / low-card integer /
  continuous-or-high-card / the moderate-integer gap) is exhaustive and each branch's comment matches its predicate;
  the `else` gap-closing branch's claim that cardinality is `<= max(nbins, 2)` there is correct given the preceding
  `elif` chain.
- **Passthrough re-attach** (`_fit_impl_core.py:386-406, 2348-2355`). `feature_names_in_` is rebuilt from
  `_passthrough_full_columns_` (the pre-narrow order) at `:1103-1107` so the sklearn input-width contract is preserved,
  and the indices are looked up in that same space at `:2352` — the two halves agree. Column-subset selection shares
  buffers (no row copy).
- **Engineered-name routing into `feature_names_in_`** (`_fit_impl_core.py:1049-1051`). `_engineered_names_set` is only
  `hybrid_orth_features_ | mi_greedy_features_`, which initially reads as under-inclusive — but every other FE family
  also appends into `hybrid_orth_features_` (verified: `_fe_stage_cascade_early_b.py:117,163,245,285,329,440,469,499,603,636,669,704`
  all do `self.hybrid_orth_features_ = list(...) + list(_appended)`). So `feature_names_in_` correctly excludes all
  engineered names, and the RFECV rescue's `_fni_idx[feature]` at `:2491` cannot `KeyError` (`temp_columns` excludes
  both rosters at `:2448-2449`). **Not a finding** — recording it because it looks like one on a first read.
- **Never-empty raw representative** (`_assign_support.py:87-213`). The cols-space → `feature_names_in_` remap at
  `:178-179` is present and correct (the comment names the IndexError it fixes); `_redundancy_emptied_raw_` correctly
  suppresses the re-attach when the empty support is the sweep's deliberate outcome; the `elif` at `:187` populates
  BOTH `_raw_redundancy_dropped_` and `_redundancy_emptied_raw_` so the downstream rescue agrees.
- **Empty-support rescue fail-closed handlers** (`_finalise.py:227-238, 249-258`). Both substitute the value that makes
  the gate **reject** (`p = 1.0`, `pair_mi = inf`) and log at WARNING naming the exception type — the correct polarity
  for this repo's documented bug class, and the comments explain why. Contrast with FIT_IMPL-6, where the SIS screen
  does the opposite.
- **`_pgn_raw_budget`** (`_helpers.py:465-472`). Pure, floored at 0, shared by both cap sites — no drift possible in the
  arithmetic itself (the drift risk is in the `cached_MIs` source, see proposed test 3).
- **Perf / `@njit` check (per the brief's REJECT rule).** Greps for `@njit` / `parallel=True` / `prange` / `cuda.jit` /
  `cupy` / `KernelTuningCache` across the cluster: the only first-party kernels are
  `_eng_dedup_batch_corr.py:32-58` (`njit(cache=True, parallel=True)` + `prange`, covering the O(K²) engineered-dedup
  correlation) and `_mrmr_degenerate.py:147-173` (`cupy` GEMM behind the `fe_gpu_strict_enabled` work-floor gate).
  `_mrmr_sis_screen.py` uses `KernelTuningCache` for chunk width (`:98-111`) and is pure glue over two already-njit
  sibling kernels (`_mi_classif_batch`, `second_moment_propensity`) — its own module docstring's "there is nothing to
  optimize in the glue" claim is confirmed by reading it. **No unoptimised Python-level hot loop calling an njit kernel
  per iteration was found in this cluster**; the per-family FE work happens in the cascade siblings (other clusters).
  The only measurable-cost first-party loops here are the 30-block `_eng_drop` cascade (FIT_IMPL-15 — O(families), not
  O(n)) and the degenerate audit's per-column Python loop at `_mrmr_degenerate.py:199-217`, whose body is one
  `_content_key` hash plus two dtype checks — bounded by `p`, not `n`, so the `prange`-fusion lever does not apply.
  **No speedup number is claimed anywhere in this document; none was measured.**
