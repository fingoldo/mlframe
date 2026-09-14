# fe_cascade — mrmr_audit_2026-09-14

## Scope

| File | LOC |
|---|---|
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_early_a.py` | 685 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_early_b.py` | 721 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_mid_a.py` | 802 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_cascade_mid_b.py` | 876 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_stage_temporal_agg.py` | 115 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_scan.py` | 161 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_eng_dedup_batch_corr.py` | 96 |
| `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fe_roster_attrs.py` | 70 |
| `src/mlframe/feature_selection/filters/_mrmr_fe_provenance.py` | 594 |
| `src/mlframe/feature_selection/filters/_mrmr_artifacts.py` | 234 |

Cross-read for call-site facts only (not audited): `_fit_impl_core.py:300-333, 434-545, 660-790`, `_fe_deadline.py`.

## Findings

### FEC-1 — wavelet + rankgauss auto-scope excludes only `hybrid_orth_features_`, so MI-greedy engineered columns become nested-recipe sources (transform-time KeyError)  [P1]
**Where:** `_fe_stage_cascade_mid_b.py:788-790` and `:846-848`
**What:**
```python
_wv_already = set(getattr(self, "hybrid_orth_features_", None) or [])
_wv_cols = [c for c in X.columns if c not in _wv_already] or None
```
(identical shape for `_rg_already` / `_rg_cols`). Every other FE family in the cascade pushes its names into `hybrid_orth_features_` (e.g. `early_b.py:117, 163, 245, 285, 329, 440, 469, 499, 603`; `mid_a.py:105, 179, 254, 313, 370, 423, 472, 590, 650, 706, 781`). **`mi_greedy_features_` is the one roster that is never merged in** — `early_a.py:588` assigns `self.mi_greedy_features_ = list(_mig_appended)` and `early_a.py:667` appends the CMI winners, and grepping the whole package shows no site that adds those names to `hybrid_orth_features_`.
**Why it is wrong / costly:** `_fe_stage_cascade_early_a` (MI-greedy, Layer 26/60) runs *before* `_fe_stage_cascade_mid_b` (`_fit_impl_core.py:484` then `:518`), so at the wavelet/rankgauss stage `X.columns` contains `mi_greedy_transform` columns that the exclusion set misses. A Haar leg or RankGauss column built on one of them produces a 2-deep recipe whose parent the 1-deep replay cannot order — precisely the failure the surrounding comment (`mid_b.py:775-783`) says it is preventing, and precisely the `KeyError('x0__p2sin1')` the 2026-06-10 fix was written for. Bites whenever `fe_mi_greedy_enable`/`fe_mi_greedy_cmi_enable` is on together with `fe_wavelet_enable` (default ON per its own comment) or `fe_rankgauss_enable`.
**Fix:** use the same source the sibling stages already use — `set(_raw_input_cols_pre_fe)` (as at `mid_b.py:298, 366, 434, 504, 573, 641, 712`) — or at minimum union `hybrid_orth_features_ | mi_greedy_features_`. `_raw_input_cols_pre_fe` is already a parameter of this function.
**Test:** `test_wavelet_scope_excludes_mi_greedy_columns` — fit with `fe_mi_greedy_enable=True, fe_wavelet_enable=True` on a frame where an MI-greedy transform is emitted; assert no recipe in `_produced_recipes_` has a `src_names` entry that is in `mi_greedy_features_`, and that `transform(X_test)` does not raise.

### FEC-2 — count/frequency-encoding auto-detect path does not exclude engineered columns (explicit-config path does)  [P1]
**Where:** `_fe_stage_cascade_early_b.py:226-231` and `:266-271`
**What:**
```python
if _cnt_cfg:
    _cnt_cols = [c for c in _cnt_cfg if c in X.columns and c not in _engineered_seen_l34]
else:
    _cnt_cols = auto_detect_te_cols(X, min_card=5, max_card=500)
```
`_engineered_seen_l34` (`:220`) is built and then applied **only** on the explicit branch. The auto-detect branch is handed the already-augmented `X`.
**Why it is wrong / costly:** by this point `X` carries the hybrid-orth, MI-greedy and k-fold-TE columns. A `{col}__te` or `mi_greedy` column with 5-500 distinct values is a valid `auto_detect_te_cols` hit, so `count_encode_with_recipes` emits a `count_encoded` recipe whose `src_names` is an engineered column → `KeyError` at transform replay. The sibling `_resolve_missing_cols` in the same file (`:412-418`) filters BOTH branches (`return [c for c in auto_detect_missing_cols(...) if c not in _engineered_seen_l37]`), which is the correct pattern — so this is an inconsistency, not a deliberate design.
**Fix:** wrap both auto-detect calls the same way `_resolve_missing_cols` does: `[c for c in auto_detect_te_cols(...) if c not in _engineered_seen_l34]`.
**Test:** `test_count_encoding_autodetect_skips_engineered_columns` — enable `fe_kfold_te_enable` + `fe_count_encoding_enable` with no `fe_count_encoding_cols`; assert every `count_encoded` recipe's `src_names` ⊆ `_raw_input_cols_pre_fe`.

### FEC-3 — four whole-frame `pd.concat` copies in the discrete-structural stages, where every sibling stage uses `fe_append_columns`  [P1]
**Where:** `_fe_stage_cascade_mid_a.py:586-588, 646-648, 702-704, 777-779`
**What:** `X = pd.concat([X, pd.DataFrame(_pm_new, index=X.index)], axis=1)` (and the `_il_new` / `_am_new` / `_cg_new` twins). Every other append site in the cascade goes through `fe_append_columns(X, fe_extract_columns(X_src, names))` (`early_a.py:235, 374, 587, 657`; `early_b.py:112, 162, 243, 283, 327, 438, 467, 497, 601, 634, 667, 702`).
**Why it is wrong / costly:** `pd.concat(..., axis=1)` materialises a brand-new frame — a full copy of every existing column, not just the appended ones. Four of these run back-to-back when the discrete-structural operators fire (default-ON master, `mid_a.py:498`), i.e. up to four full duplications of an already FE-augmented frame in one fit. CLAUDE.md's Memory/RAM discipline forbids exactly this on 100+ GB frames.
**Fix:** route all four through `fe_append_columns` like the siblings; the payload is already a `{name: array}` dict, which is what that seam takes.
**Test:** `test_discrete_structural_stages_do_not_whole_frame_copy` — monkeypatch `pd.concat` to raise inside `_fe_stage_cascade_mid_a` and assert a fit with `fe_pairwise_modular_enable=True` still appends its columns (mutation-resistant: asserts the append happened, not just that no exception escaped).

### FEC-4 — dedup's rank buffer is a `K × n` float64 `np.empty` allocated unconditionally, with no byte-size gate  [P1]
**Where:** `_eng_dedup_scan.py:45`
**What:** `_eng_rank_buf = np.empty((len(_eng_cols_appended), len(X)), dtype=np.float64)` — allocated at function entry, before any column is inspected, sized by the *total* appended-column count even though rows are only ever written for fully-finite columns (`:151, :158`).
**Why it is wrong / costly:** the module's own docstring cites "~200 engineered columns a wide fit can produce". At n = 2 M that is 3.2 GB; at the 100 M-row frames CLAUDE.md says this codebase targets it is 160 GB. The allocation happens even when only one column turns out fully finite, and even when the batched path is never taken (`:96` requires `_eng_next_free_row > 0`). On Windows `np.empty` commits pages against the paging file, which is the documented `WinError 1455` failure mode in this repo.
**Fix:** gate on bytes the way the rest of the FE seam does (`fe_polars_exceeds`'s ~2 GB rule): allocate lazily on first write, grow geometrically, and fall back to the unchanged per-pair `np.corrcoef` path above the byte budget.
**Test:** `test_eng_dedup_rank_buffer_respects_byte_budget` — call `scan_engineered_duplicates` with a stub frame whose `len(X)` × K would exceed the budget and assert no allocation larger than the cap occurs (spy on `np.empty`), and that the returned keep/drop sets match the per-pair reference path.

### FEC-5 — the missingness stage writes back into the working frame, mutating the caller's DataFrame and coercing its dtype  [P1]
**Where:** `_fe_stage_cascade_early_b.py:386-393`
**What:**
```python
_restored = _col_now.to_numpy().astype(np.float64, copy=True)
_restored[_mask] = np.nan
X[_mc] = _restored
```
**Why it is wrong / costly:** two problems. (a) `X` at this point may still be the user's own `DataFrame` (no defensive copy is taken anywhere in the cascade — the whole design is X-in/X-out by reassignment), so `X[_mc] = ...` is an observable side effect on the caller's data: an int/categorical column silently becomes float64 and gains NaNs after `fit()` returns. (b) Under the polars bridge (CLAUDE.md: `get_pandas_view_of_polars_df` gives an Arrow-backed *view*) this is a write into a view of the user's polars frame. The restore is also all-or-nothing (`if not _col_now.isna().to_numpy().any()`), so a *partially* imputed column is left partially wrong rather than restored.
**Fix:** build the restored columns into a local dict and feed the missingness family a frame assembled from them (`fe_extract_columns` + a local overlay), never assigning back into `X`. Drop the `astype(np.float64)` for columns that are already float.
**Test:** `test_missingness_stage_does_not_mutate_caller_frame` — fit with `fe_missingness_indicator_enable=True` on a frame snapshot; assert `X.equals(snapshot)` and `X.dtypes.equals(snapshot.dtypes)` afterwards.

### FEC-6 — the wall-clock FE budget gates 4 of ~35 family stages; the rest of the cascade is unbounded  [P2]
**Where:** `_fe_budget_ok` is defined at `_fit_impl_core.py:313-320` and threaded into all three big cascade modules (`early_a.py:25`, `mid_a.py:37`, `mid_b.py:40`), but is *consulted* at only four sites: `early_a.py:126` (hybrid/univariate basis), `early_a.py:422` (hinge), `mid_a.py:498` (discrete-structural master), `mid_b.py:760` (wavelet).
**What:** `_fe_stage_cascade_early_b` does not even receive `_fe_budget_ok`; `mid_b`'s ten other Layer-104 families (rare-category, conditional residual/dispersion/quantile-rank, ordinal-pattern, random-Fourier, SIR, LOF, Mahalanobis, rankgauss) and all of `mid_a`'s Layer 87/93/88/89/94/90/95A families check only `_fe_family_on`, which is a pure `fe_max_steps > 0` test with no time component.
**Why it is wrong / costly:** the helper's own docstring says its purpose is "so an oversized fit handed a small `max_runtime_mins` aborts within a small multiple of the budget". With 31 of 35 stages unchecked, `max_runtime_mins` is close to advisory across the pre-FE cascade. The per-candidate `fe_deadline_passed()` checks inside the family implementations (`_target_encoding_fe.py:429`, `_conditional_gate_fe.py:799`, `_hinge_basis_fe.py:537`, …) mitigate the families that have them, but ~20 of the families reached from this cascade have no deadline consumer at all (grep of `fe_deadline_passed` shows 14 consumer modules against ~35 families).
**Fix:** make the between-stage gate uniform — either fold `_fe_budget_ok()` into `_fe_family_on` (one change, covers every family), or add the explicit `and _fe_budget_ok()` to each stage guard and thread the helper into `early_b`.
**Test:** `test_fe_cascade_respects_max_runtime_for_every_family` — parametrised over the `fe_*_enable` flags: set `max_runtime_mins` to an already-elapsed budget via a frozen clock and assert each family's roster comes back empty.

### FEC-7 — the thread-local FE deadline does not cross the joblib worker boundary, and the cascade never re-publishes it  [P2]
**Where:** `_fe_deadline.py:13-14` states "A thread-local is sufficient because every `fe_deadline_passed()` consumer runs INLINE on the MAIN thread"; `polynom_pair_fe.py:444-446, 517` is the one place that works around this by reading `_fe_deadline._state.deadline` and passing it as an explicit argument into the `delayed()` payload, re-publishing via `fe_deadline_scope` at `:395`.
**What:** none of the cascade stages does that. They call family entry points (`hybrid_grouped_agg_fe`, `hybrid_cat_pair_fe`, `hybrid_conditional_gate_fe_with_recipes`, …) with no deadline argument, and those families' own `fe_deadline_passed()` calls are therefore only effective while they stay on the main thread.
**Why it is wrong / costly:** any family that fans out over `joblib` internally silently loses the deadline — the budget is not "approximate", it is absent on that path, with no warning. This is a latent version of the bug `polynom_pair_fe.py:248-251` documents.
**Fix:** unverified which of the cascade-reached families actually dispatch to joblib; settle it by grepping each family module for `Parallel(`/`delayed(` and, for each hit, adopt the `polynom_pair_fe` explicit-argument + `fe_deadline_scope` pattern.
**Test:** `test_fe_deadline_reaches_joblib_workers` — set a passed deadline, run the family under `n_jobs=-1`, assert the worker-side `fe_deadline_passed()` returns True (spy in the payload).

### FEC-8 — six shipped recipe kinds are unmapped in `_RECIPE_KIND_TO_ORIGIN`, so their columns report as `engineered_unknown`  [P2]
**Where:** `_mrmr_fe_provenance.py:102-165`
**What:** the emitted kinds `mahalanobis_density`, `random_fourier`, `lof_score`, `conditional_quantile_rank`, `sir_direction`, `ordinal_pattern_te` (grepped from `_mahalanobis_density_fe.py`, `_random_fourier_features_fe.py`, `_lof_fe.py`, `_conditional_quantile_rank_fe.py`, `_sliced_inverse_regression_fe.py`, `_ordinal_pattern_fe.py`) appear in none of the map's keys. `FE_ORIGIN_LABELS` (`:65-96`) likewise has no bucket for them.
**Why it is wrong / costly:** all six families are wired in `_fe_stage_cascade_mid_b.py` (`:338, :406, :476, :545, :614, :683`) and each has its own roster in `FE_ROSTER_ATTRS`, so they genuinely fire. Their surviving columns collapse to `engineered_unknown`, which is exactly the regression `get_unlabeled_recipe_kinds` (`:523`) was written to detect and which its docstring claims is currently only `{"factorize": N}`. The docstring is therefore also wrong.
**Fix:** add the six kinds with dedicated origin labels (or to `extra_fe`, matching the treatment of `rare_category`/`conditional_residual`/`conditional_dispersion` at `:150-153`) and extend `FE_ORIGIN_LABELS`.
**Test:** `test_every_emitted_recipe_kind_has_an_origin_label` — meta-test that greps `kind="..."` literals across `filters/` and asserts each is a key of `_RECIPE_KIND_TO_ORIGIN`; this generalises the bug class rather than pinning six names.

### FEC-9 — `get_unlabeled_recipe_kinds` keys recipes by RAW name but looks them up with SIMPLIFIED names  [P2]
**Where:** `_mrmr_fe_provenance.py:552-557` vs `:447-448` and `:563-565`
**What:** the self-audit builds `recipe_by_name[str(nm)] = r` (raw), while `compute_fe_provenance` builds its index as `{simplify_fe_name(str(getattr(r, "name", ""))): r}` and therefore writes SIMPLIFIED names into `prov["feature_name"]`. Line 563 then does `recipe_by_name.get(str(name))` with those simplified names.
**Why it is wrong / costly:** for any engineered column whose name canonicalises (the `abs(div(sqr(a),neg(b)))` → `abs(div(sqr(a),b))` case the surrounding comments describe), the lookup misses and the kind is recorded as `"<no-recipe>"` instead of the real kind. The guardrail that is supposed to name the unregistered family reports a useless bucket — directly compounding FEC-8.
**Fix:** apply `simplify_fe_name` on the key at `:557`, mirroring `:447`.
**Test:** `test_unlabeled_kinds_resolves_simplified_names` — construct a recipe whose name simplifies, force it unlabeled, assert the returned dict is keyed by the recipe's `kind`, not `"<no-recipe>"`.

### FEC-10 — `_greedy_rank_for_name` is O(n_names × n_predictors) with a `simplify_fe_name` call per pair  [P2]
**Where:** `_mrmr_fe_provenance.py:324-351`, called once per name at `:467`
**What:** the function linearly scans `predictors` and calls `simplify_fe_name(str(_en))` on every entry, for every name.
**Why it is wrong / costly:** the predictor log's simplified names are invariant across the outer loop — they are re-derived `n_names` times. On a wide kitchen-sink fit (hundreds of produced-but-screened-out engineered names × hundreds of predictors) this is ~10^5 parser calls of pure waste. The file already contains the identical fix for the sibling O(n·m) problem — `_build_roster_membership_sets` (`:281-301`) exists precisely because "`_origin_from_rosters` used to … do an O(len(roster)) membership test from scratch, PER NAME" — so the pattern was fixed on one path and missed on this one.
**Fix:** build `{simplify_fe_name(str(e["name"])): idx for idx, e in enumerate(predictors)}` once in `compute_fe_provenance` (keeping the FIRST index on duplicates, to preserve today's first-match semantics) and pass it in; keep the scanning fallback for other callers exactly as `_origin_from_rosters` keeps its own.
**Test:** `test_greedy_rank_lookup_is_hoisted` — spy on `simplify_fe_name` and assert the call count is O(n_names + n_predictors), plus an equivalence assertion that the resulting `support_rank` column is unchanged vs the scanning path.

### FEC-11 — `set(...)` rebuilt once per column inside four list comprehensions  [P2]
**Where:** `_fe_stage_cascade_mid_a.py:569, 630, 691, 752`
**What:** `_pm_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]` — the `set()` constructor is in the *condition*, so it is re-evaluated for every column.
**Why it is wrong / costly:** O(n_cols × n_engineered) set construction where O(n_cols + n_engineered) suffices. With the ~200 engineered columns the dedup module's docstring cites and a wide frame this is hundreds of thousands of wasted tuple-to-set builds per stage, four stages over. `mid_b.py:789` and `:847` show the correct hoisted form in the same codebase.
**Fix:** hoist to a local above each comprehension.
**Test:** covered by a micro-benchmark rather than a unit test; no behaviour change, so a perf-regression test is not warranted.

### FEC-12 — dedup survivorship is emission-order-dependent because Spearman ≥ 0.99 is not transitive  [P2]
**Where:** `_eng_dedup_scan.py:111-156` combined with `_fit_impl_core.py:712-732`
**What:** each candidate is compared only against the columns currently in `_eng_keep`; a candidate that loses is dropped immediately (`:142`), a candidate that wins evicts every colliding kept column (`:144-148`).
**Why it is wrong / costly:** with a near-duplicate relation that is not transitive (A~B, B~C, A≁C — routine at a 0.99 threshold), the surviving *set size* depends on the order `_eng_cols_appended` happens to be built in (`_fit_impl_core.py:670`: `hybrid_orth_features_` then `mi_greedy_features_`). Order A,B,C keeps {A, C}; order B,A,C keeps {B} alone. The `_eng_dedup_prefer` tie-break docstring (`:715-718`) promises determinism "byte-for-byte" within a stage, which holds, but says nothing about this cross-cluster instability — and the cluster composition changes whenever an upstream family's `top_k` shifts.
**Fix:** make the policy a documented single-pass greedy over a deterministic order (e.g. sort candidates by descending `_eng_mi` before the scan, so the highest-relevance column is always processed first and is never evicted), which makes the survivor set a function of the values rather than of emission order.
**Test:** `test_eng_dedup_survivor_set_is_order_invariant` — build three synthetic engineered columns with the intransitive correlation structure above, run `scan_engineered_duplicates` over all 6 permutations of `_eng_cols_appended`, assert the returned `keep` set is identical each time.

### FEC-13 — raw-floor frames are built with `X[[...]]`, which copies every selected column  [P2]
**Where:** `_fe_stage_cascade_early_b.py:427`, `_fe_stage_cascade_mid_b.py:152`, `:220`; also `mid_a.py:231`
**What:** `_raw_floor_X = fe_to_pandas(X)[[c for c in _raw_input_cols_pre_fe if c in X.columns]]`
**Why it is wrong / costly:** pandas fancy column indexing returns a copy of the selected block, so each of these materialises the *entire raw input frame* a second time purely to compute an MI noise floor. Three such sites fire in one fit when the indicator / rare-category / conditional-residual families are on.
**Fix:** pass the raw column NAME list plus the already-materialised frame down to the family and let it slice per-column as it scores, or build the floor from a subsample (the MI floor is a rank statistic; the cascade already has `fe_decide_on_subsample` for exactly this).
**Test:** `test_raw_floor_does_not_copy_full_frame` — assert the peak allocation during the indicator stage stays below a multiple of one column's size (spy on `DataFrame.__getitem__` with a list key).

### FEC-14 — `fe_to_pandas(X)` is re-invoked per family and twice per call in one place  [P2]
**Where:** `_fe_stage_cascade_early_b.py:91, 151, 235, 275, 314, 418, 427, 428, 463, 493, 595, 628, 661, 695`
**What:** fourteen independent `fe_to_pandas(X)` calls in one function; `:427-428` calls it twice for the same `X` within two adjacent expressions, and `_resolve_missing_cols` (`:418`) calls it once per family that uses it (three families).
**Why it is wrong / costly:** for a pandas input this is cheap (identity), but for a polars input under the ~2 GiB gate each call is a real conversion. The block already decided once, at `:52`/`:193`/`:367`/`:559`, that the frame is small enough to materialise — it should materialise once and reuse.
**Fix:** hoist `_X_pd = fe_to_pandas(X)` immediately after each `fe_polars_exceeds` gate and use it throughout the block. Note this interacts with FEC-5: today `X[_mc] = _restored` writes to `X`, not to the converted copy, so the hoist must be done alongside that fix, not before it.
**Test:** `test_early_b_converts_frame_once_per_block` — spy on `fe_to_pandas` under a polars input with several Layer-34/37/38 families on; assert the call count is one per gated block.

### FEC-15 — artifacts computes per-column entropy in a Python loop of `np.bincount` calls, with no njit/prange anywhere in the file  [P2]
**Where:** `_mrmr_artifacts.py:146-192`
**What:** the loop does, per feature, `x_bins = data[:, data_col]` (a strided, non-contiguous column view), `np.bincount(...)`, a `> 0` mask, a `log`, and a `sum`. Grep of the file confirms no `@njit`, no `parallel=True`/`prange`, no `cuda.jit`, no `cupy`, no `KernelTuningCache` — so the brief's REJECT precondition is not met.
**Why it is wrong / costly:** this is the repo's signature win shape at column granularity — `n_features_in` independent reductions over a shared matrix, each currently paying Python dispatch plus a strided gather. `_extremality_matrix_njit` (`row_wise_extremality.py`, CLAUDE.md 2026-08-04 entry) is the exact precedent: `prange` over COLUMNS of a shared matrix, 2.11x. Only pays when `retain_artifacts=True`, so the blast radius is bounded.
**Fix:** one `@njit(parallel=True)` kernel taking `data`, the column-index array and `nbins`, `prange` over columns, accumulating counts into a per-column private buffer and returning `h_x` directly. Bench plan: n ∈ {500, 2k, 10k, 100k, 1M} × n_features ∈ {20, 200}, warm, best-of-10, against the current loop, with a bit-identity gate on `su_to_target`.
**Test:** `test_artifact_entropy_kernel_matches_bincount_reference` — bit-identical `h_x`/`su_to_target` vs the current implementation across constant, all-one-bin, and NaN-shifted-code columns.

### FEC-16 — `_ROSTER_ATTR_TO_ORIGIN` covers 13 of the 39 rosters in `FE_ROSTER_ATTRS`  [P2]
**Where:** `_mrmr_fe_provenance.py:183-198` vs `_fe_roster_attrs.py:19-59`
**What:** the roster→origin fallback lists 13 attributes; `FE_ROSTER_ATTRS` lists 39. Missing include `wavelet_features_`, `rankgauss_features_`, `conditional_gate_features_`, `row_argmax_features_`, `integer_lattice_features_`, `temporal_agg_features_`, `grouped_agg_features_`, `cat_pair_features_`, and every Layer-104 roster.
**Why it is wrong / costly:** the fallback fires whenever a name has no recipe. Also, `_final_feature_order` (`:402-411`) drains names from *this same 13-entry tuple*, so 26 rosters' survivor names reach the provenance frame only if they happen to carry a recipe — a family that stores "roster names + replay logic" without an `EngineeredRecipe` (the case the comment at `:400-401` calls out explicitly) is simply absent from the audit trail.
**Fix:** drive both loops from `FE_ROSTER_ATTRS` with an explicit label map, keeping the documented ordering invariant (specific rosters before the two catch-alls). `_fe_roster_attrs.py`'s docstring already says a meta-test pins that tuple against what the cascades seed — extend it to pin this map too.
**Test:** `test_every_fe_roster_has_a_provenance_origin` — assert `set(FE_ROSTER_ATTRS) - {"_adaptive_fourier_features_", "_hinge_features_"} ⊆ {a for a, _ in _ROSTER_ATTR_TO_ORIGIN}`.

### FEC-17 — `try/except NameError` used as control flow, and its fallback inverts the documented gate  [P3]
**Where:** `_fe_stage_cascade_early_a.py:261-264` and `:291-294`
**What:**
```python
try:
    _extra_basis_scorer_ok = _default_scorer == "plug_in"
except NameError:
    _extra_basis_scorer_ok = True
```
**Why it is wrong / costly:** `_default_scorer` is only bound at `:185`, inside the polynomial stage's `try:`; if the import at `:132` or the y-densification raised, it is unbound. The fallback then sets `True`, which the comment at `:288-290` says is precisely the wrong answer — under a non-plug-in `fe_hybrid_orth_default_scorer` the Fourier extra basis "would emit columns the routed scorer never selected". So a transient failure in the polynomial stage silently re-enables a path the config disabled. The same pattern at `:262` is benign but equally fragile. This also defeats the CLAUDE.md AST name-gate, which cannot distinguish a deliberate unbound-name guard from a split regression.
**Fix:** bind `_default_scorer` and `_y_for_hybrid` **before** the `try:` at `:131`, and delete both `except NameError` handlers.
**Test:** `test_extra_basis_skipped_when_poly_stage_raises_under_alternate_scorer` — force the `_orthogonal_univariate_fe` import to raise with `fe_hybrid_orth_default_scorer="jmim"`; assert no `orth_fourier` recipe is produced.

### FEC-18 — Symmetric Uncertainty is silently clamped into [0, 1]  [P3]
**Where:** `_mrmr_artifacts.py:182` — `su_to_target[orig_idx] = max(0.0, min(1.0, 2.0 * mi_val / denom))`
**What:** any `2·MI/(H(X)+H(y))` outside [0,1] is replaced with the boundary value with no log.
**Why it is wrong / costly:** SU > 1 is mathematically impossible; observing it means the cached MI and the recomputed marginal entropies disagree (different binning, a stale cache entry, or a estimator bias). Clamping converts that diagnostic signal into a plausible-looking 1.0 that a consumer treats as "perfectly informative". This is the repo's documented "fallback substitutes a non-neutral value silently" class.
**Fix:** keep the clamp but `logger.warning` with the feature name and the pre-clamp value when it actually fires.
**Test:** `test_su_clamp_warns_on_out_of_range` — feed a `cached_MIs` entry larger than `(h_x + h_y)/2` and assert a warning is emitted naming the column.

### FEC-19 — artifacts use only `target_indices[0]`, but the schema advertises "SU(X_j, y)"  [P3]
**Where:** `_mrmr_artifacts.py:128-139` (`y_idx = int(target_indices[0])`) vs the parameter doc at `:96-97` ("shape (1 or k,)") and the schema strings at `:54-55`.
**What:** for a multilabel/multi-target fit, `h_y` and every SU value are computed against the first target column only; the remaining `k-1` targets are silently ignored and nothing in the exported dict records that.
**Why it is wrong / costly:** a consumer (`ShapProxiedFS(precomputed=...)`) reading `su_to_target` for a multilabel MRMR gets a single-label statistic labelled as the general one.
**Fix:** either raise/warn when `len(target_indices) > 1`, or add a `target_index_used` key to the exported dict and document it in `_ARTIFACT_SCHEMA`.
**Test:** `test_artifacts_multitarget_declares_which_target` — fit multilabel with `retain_artifacts=True`; assert the exported dict records the target used (or that a warning fired).

### FEC-20 — missingness count and pattern families read the *indicator* family's column config  [P3]
**Where:** `_fe_stage_cascade_early_b.py:460` and `:489` — both call `_resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))`
**What:** there is no `fe_missingness_count_cols` / `fe_missingness_pattern_cols`; a user enabling only `fe_missingness_count_enable` must set a parameter named for the indicator family to scope it.
**Why it is wrong / costly:** a user who sets `fe_missingness_indicator_cols` to scope the indicator and enables count/pattern gets that scope silently applied to families they never configured. Not a wrong result, but an undocumented coupling between three "independent master switches" (the comment at `:346-347` calls them independent).
**Fix:** document the shared parameter in the flag docstrings, or add per-family overrides falling back to the shared one.
**Test:** none needed beyond a docs change if the shared-config reading is intended; if per-family overrides are added, `test_missingness_count_honours_its_own_cols`.

### FEC-21 — `_gate_raw_operands_` / `_gate_col_src_vars_` are seeded in one cascade module and consumed in another  [P3]
**Where:** seeded at `_fe_stage_cascade_early_b.py:542, 548`; mutated at `_fe_stage_cascade_mid_a.py:712-713, 787-788`
**What:** `mid_a` does `self._gate_raw_operands_.update(...)` with no `getattr` default, relying on `early_b` having run first.
**Why it is wrong / costly:** `early_b` has no unconditional early return today, so this holds — but it is the same class of implicit cross-sibling contract the monolith-split rule warns about, and `mid_a` is separately callable. The other 39 rosters are protected by the explicit `seed_empty_fe_rosters` contract in `_fe_roster_attrs.py`; these two are not, and are not listed in `FE_ROSTER_ATTRS`.
**Fix:** seed both in `seed_empty_fe_rosters` (or at the top of `mid_a` with a `getattr` guard), and add them to the meta-test's coverage.
**Test:** `test_gate_operand_attrs_present_after_multioutput_fit` — mirrors `test_fe_roster_attrs_exist_after_fit.py` for these two attributes.

### FEC-22 — module docstrings carry split/process metadata, against the repo's comment rules; two files are near the LOC gate  [P3]
**Where:** `_fe_stage_cascade_mid_a.py:1-2`, `_fe_stage_cascade_mid_b.py:1-2, 15-17`, `_fe_stage_temporal_agg.py:2-4` ("Carved verbatim out of the giant `_fit_impl` orchestration body … (Tier E partial split)"), `_eng_dedup_scan.py:2-6`; plus dozens of dated markers in code comments (`early_a.py:39, 105, 251, 405, 491, 513, 524, 604`; `mid_a.py:499`; `mid_b.py:775`).
**What:** CLAUDE.md's comment-style rule bans "process/audit metadata … no phase/wave markers … refactor-history narration … date stamps" — "that belongs in git history / the PR description".
**Why it is wrong / costly:** quality only, but it is a named repeated complaint in the project conventions, and the volume here is large. Separately, `mid_a.py` (802) and `mid_b.py` (876) are inside the "carve before ~800-900 LOC" band the same document specifies, so the next family added to either trips the 1k backstop.
**Fix:** strip the refactor-history and dated markers on next touch; plan the next carve of `mid_b` (the Layer-104 block from `:109` onward is a natural seam) rather than waiting for `test_no_file_over_1k_loc.py`.
**Test:** n/a (lint/meta).

### FEC-23 — `validate_artifact_dict` ignores `n_samples_at_fit`, which the schema tells consumers to check  [P3]
**Where:** `_mrmr_artifacts.py:208-234` vs the schema string at `:60` ("consumers should warn / discard on shape mismatch")
**What:** the validator checks only `su_to_target` presence/shape against `feature_names`. It never looks at `n_samples_at_fit`, `schema_version`, or the `bins`/`nbins_per_feature` pairing invariant the schema states ("Present together with 'bins'").
**Why it is wrong / costly:** the one helper written to save consumers from a malformed dict does not check the two things the schema explicitly asks consumers to check; a version-1 consumer handed a future version-2 dict passes validation.
**Fix:** add a `schema_version` check (warn-and-reject above `ARTIFACT_SCHEMA_VERSION`) and a `bins`⇔`nbins_per_feature` co-presence check; leave the row-count comparison to the caller (it needs the consumer's own `n`) but expose a `n_samples` parameter for it.
**Test:** `test_validate_artifact_dict_rejects_future_schema_version` and `test_validate_artifact_dict_rejects_half_present_bins`.

### FEC-24 — the temporal-agg stage re-implements `_fe_family_on` inline instead of calling it  [P3]
**Where:** `_fe_stage_temporal_agg.py:48` — `if bool(getattr(self, "fe_temporal_agg_enable", False)) and int(getattr(self, "fe_max_steps", 0) or 0) > 0:`
**What:** byte-for-byte the body of `_fe_family_on` (`_fit_impl_core.py:333`), duplicated because the helper is not threaded into this stage (`:33` takes only `self, X, _y_np, verbose, _temporal_agg_pre_recipes`).
**Why it is wrong / costly:** duplicated gate logic that must be kept in sync by hand — the comment at `:46-47` even says "mirrors `_fit_impl._fe_family_on`". If FEC-6 is fixed by folding the budget into `_fe_family_on`, this copy silently keeps the old semantics.
**Fix:** thread `_fe_family_on` in from `_fit_impl_core.py:540` and call it.
**Test:** covered by FEC-6's parametrised test once the helper is shared.

### FEC-25 — the batched and per-pair dedup paths use different degeneracy thresholds  [P3]
**Where:** `_eng_dedup_batch_corr.py:72` (`if saa <= 1e-24 * n or sbb <= 1e-24 * n: continue`, leaving `out[j] = 0.0`) vs `_eng_dedup_scan.py:119, 132` (`if _a.std() <= 1e-12 or _b.std() <= 1e-12: continue` and the same on ranks)
**What:** the kernel's floor is on the *sum of squared deviations* relative to `n`; the Python path's is on the *standard deviation*. `saa <= 1e-24·n` is `std <= 1e-12`, so they do agree numerically — but they also differ in outcome shape: the kernel writes `0.0` (read as "not a duplicate") while the Python path `continue`s (also "not a duplicate"). Verified equivalent; flagged only because the two expressions are written differently enough that a future edit to one will not obviously need the other.
**Why it is wrong / costly:** no live divergence found. The risk is maintenance: the identity test that pins the kernel against a per-pair reference is the same test the `_eng_dedup_scan.py` docstring (`:3-6`) says *already drifted out of date once*.
**Fix:** none required; optionally express the kernel floor as a named shared constant.
**Test:** `test_dedup_batched_and_pairwise_agree_on_near_constant_columns` — a column with `std` straddling `1e-12`, asserting both paths return the same keep/drop verdict.

## Proposed tests (beyond the per-finding ones)

- `test_fe_cascade_stage_handles_zero_output_from_previous_stage` — run each of the four cascade functions with an `X` containing only raw columns and every prior roster empty; assert none raises and each returns a frame with the same columns it was given. Today the "previous stage produced nothing" path is only exercised incidentally by whichever fixture happens to disable a family.
- `test_cascade_rosters_and_recipes_stay_consistent_after_dedup` — after a fit with dedup dropping at least one column, assert `set().union(*rosters) == {r.name for r in _produced_recipes_} ∩ X.columns`, i.e. no roster entry lacks a recipe and no surviving recipe names a dropped column. The dependency-closure loop at `_fit_impl_core.py:768-775` is the only thing keeping this true and has no direct test.
- `test_fe_cascade_is_deterministic_across_repeat_fits` — two fits with the same `random_seed` on the same frame produce identical `fe_provenance_["feature_name"]` order. Guards FEC-12 and the several `set()`-derived scopes.
- `test_no_cascade_stage_swallows_an_exception_without_warning` — meta-test over the four cascade modules asserting every `except Exception` handler's body contains a `logger.warning` (all 30-odd currently do; the test locks it in, since this repo has shipped a silent handler twice).
- `test_fe_provenance_survives_pickle_roundtrip` — `mechanism_details` is stringified specifically so the frame pickles; nothing asserts it.

## Prior-wave findings touching this cluster

- `audits/mrmr_audit_2026-07-25/fit_impl.md:40` — "`_eng_dedup_batch_corr.py`: two-pass centered Pearson with a relative-variance floor, append-only buffer + active-mask (no per-candidate re-copy); assumes finite inputs, which the caller guarantees. Clean." **Still holds** for the kernel itself (re-verified: `_eng_dedup_scan.py:96` gates the batched path on `_eng_fully_finite[_c]` and only rows for fully-finite columns are ever written, so the finiteness precondition is genuinely enforced). The prior wave audited the kernel but not its caller's `np.empty` sizing — see FEC-4.
- `audits/mrmr_audit_2026-07-25/fit_impl.md:8` lists `_fe_stage_temporal_agg.py` in scope but records no finding against it. Re-read here: the one issue found is FEC-24 (duplicated gate), which is new/not previously raised.
- No prior-wave doc mentions `_fe_stage_cascade_early_a/b` or `_fe_stage_cascade_mid_a/b` by name (grep across the three prior audit directories), so the 3084 LOC of carved cascade in this cluster is being audited for the first time.

## Verified-clean

- **Monolith-split `NameError` gate (the cluster's highest-priority check): CLEAN.** AST-walked every `Load`-context `Name` in all ten files, resolving against local bindings (args, assignments, comprehension targets, `except` names, `with` targets, lambda args, `global`/`nonlocal`), module-level bindings, and builtins. **Zero unresolved names.** Note the two deliberate `except NameError` guards at `_fe_stage_cascade_early_a.py:263, 293` are a *different* thing (conditionally-bound locals, not missing imports) and are reported separately as FEC-17.
- **Recipe dict mutation-vs-reassignment contract:** every `_*_pre_recipes` parameter is only ever mutated (`d[name] = r`), never rebound, in all four cascade modules — the docstrings' claim is accurate, so the "no return needed" contract holds.
- **Dedup drop propagation:** `_eng_drop` is applied to all 39 rosters and to every one of the 38 pre-recipe dicts (`_fit_impl_core.py:744-919`), and the `while True` dependency-closure loop (`:768-775`) correctly un-drops any column a surviving recipe consumes via `src_names` before `X.drop`. No dangling-recipe / `raw_recipes[col]` KeyError path found in this cluster.
- **Adaptive-Fourier force-keep:** `_eng_dedup_scan.py:52-60` records the force-kept column's array so later candidates are still deduped *against* it, and correctly skips giving it a rank-buffer row — the batched path therefore never compares against it, and the per-pair path computes its ranks lazily at `:127`. Correct.
- **Dedup uses full-column Spearman, not a subsample** — the decision-stability concern in the brief does not apply: `_arr_c` / `_arr_k` are full columns (`:70`, `:114`).
- **`_eng_dedup_prefer` symmetry:** within-stage pairs deterministically fall back to first-appended (`_fit_impl_core.py:722-723`), and the cross-stage MI tie-break is antisymmetric (`:728-731`). No self-contradictory verdict possible for a given ordered pair.
- **Shared class-MI target binning** (`_fe_stage_cascade_mid_a.py:525-544`) is computed once and reused by all four discrete-structural operators, as its comment claims. No per-operator re-binning.
- **NaN handling in the batched correlation kernel:** the two-pass mean-then-center form is the numerically-stable one (no `E[x²] − E[x]²`), so the repo's catastrophic-cancellation class does not apply here; no additive-epsilon denominator padding anywhere in the dedup path (`denom = (saa * sbb) ** 0.5`, `_eng_dedup_batch_corr.py:74`).
- **No `sum(x^k)`-minus-power-of-mean sites** in any of the ten files (grepped); the cascade modules delegate all moment computation to the family implementations.
- **`seed_empty_fe_rosters`** (`_fe_roster_attrs.py:62-70`) does what its docstring says, and `FE_ROSTER_ATTRS` (39 names) matches the union of rosters actually seeded across `early_a.py:52/57/63/72, 531`, `early_b.py:48, 183-185, 351-353, 521-535, 549`, `mid_b.py:111-121`, plus `grouped_agg`/`composite_group_agg`/`grouped_quantile`/`cat_pair`/`cat_triple`/`numeric_decompose`/`temporal_agg` seeded in `early_b.py:525-531`. No roster is assigned without being seeded first.
