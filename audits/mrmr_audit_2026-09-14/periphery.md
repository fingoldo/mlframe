# periphery — mrmr_audit_2026-09-14

## Scope

| File | LOC |
|---|---|
| `src/mlframe/feature_selection/filters/_mrmr_validate_transform.py` | 725 |
| `src/mlframe/feature_selection/filters/_mrmr_fingerprints.py` | 605 |
| `src/mlframe/feature_selection/filters/_mrmr_explain.py` | 279 |
| `src/mlframe/feature_selection/filters/_mrmr_stability_report.py` | 246 |
| `src/mlframe/training/composite/discovery/_mrmr_base_rank.py` | 87 |

Cross-read for evidence (not audited as scope): `mrmr/_mrmr_class.py:3593-3656` (identity-cache key),
`_mrmr_fit_impl/_fit_impl_core.py:185-270` (`_FIT_CACHE` key), `engineered_recipes/_recipe_dispatch.py:26-90`
(`apply_recipe` signature), `_fe_stability_vote.py:81-92` (`_marginal_mi`),
`training/composite/discovery/_auto_base.py:660-703` (the `mrmr_rank_bases` consumer).

## Findings

### PERIPHERY-1 — `transform()`'s identity fast-path returns the caller's frame without checking it is the SAME columns  [P0]
**Where:** `_mrmr_validate_transform.py:352-362`
**What:** When `recipes` is empty and the number of selected columns equals `X.shape[1]`, `transform` does
`return X` — before any column-name validation. The name-drift guard that raises `RuntimeError`
(`:381-392`) lives further down in the pandas branch and is never reached. The bare `n_features_in_` shape
check at `:338-348` is explicitly SKIPPED for named frames (`_is_named_frame` at `:346-347`), so a pandas /
polars frame reaches the fast path with no identity check at all.
**Why it is wrong / costly:** the fast path only compares a COUNT. Fit on `[a,b,c]` selecting all three, then
transform a frame whose columns are `[a,c,b]` (reordered) returns them in the caller's order, whereas the
normal path (`base_out = X[selected_cols]`, `:393`) returns them in fit order — the downstream model consumes
positionally, so every column is silently mis-mapped. Worse, a frame with a completely DIFFERENT 3-column set
(`[x,y,z]`) is returned unchanged with no error and no log line; the `RuntimeError` designed for exactly this
("an upstream step is mutating the column set BETWEEN fit and transform") cannot fire. This is the
silent-wrong-value case the brief names as P0, and it triggers on the most common configuration of all
(MRMR selected everything, FE off).
**Fix:** before returning `X` unchanged, require column identity, not just width: for a named frame,
`list(X.columns) == [self.feature_names_in_[i] for i in support]`; otherwise fall through to the normal
by-name path. The no-copy benefit is preserved for the genuine identity case (`X[selected_cols]` on an
already-matching frame can also be skipped after the equality check).
**Test:** `test_transform_identity_fastpath_rejects_reordered_and_foreign_columns` — fit on a 3-col frame
where every column is selected and `_engineered_recipes_` is empty; assert `transform(X[['a','c','b']])`
returns columns in fit order `a,b,c` (or raises), and that `transform(X.rename(columns={'a':'x','b':'y','c':'z'}))`
raises `RuntimeError` naming the missing columns. Today both return the input untouched.

### PERIPHERY-2 — `_append_engineered` makes a WHOLE-FRAME deep copy on every `transform()` call  [P1]
**Where:** `_mrmr_validate_transform.py:547-553` (`chained = _X_for_recipes.copy()`)
**What:** The pandas replay path copies the entire input frame (all rows, all columns, deep) purely to own a
scratch frame it can append engineered columns onto. The comment justifies the copy as "take ownership …
never mutate the caller's own frame", and the loop below then correctly uses in-place `chained[r.name] = col`
instead of `.assign()`.
**Why it is wrong / costly:** CLAUDE.md's memory rule is explicit — frames can be 100+ GB and must never be
`.copy()`d to work around an aliasing concern. `chained` is only ever READ by single named-column lookups
(`_extract_column` inside `apply_recipe`) and discarded at loop end (the return is built from `_results`,
`:652`) — the file's own comment at `:609-613` says so. The deep copy is therefore pure waste of one full
frame's RAM and bandwidth per transform call, and transform runs on every predict/val/OOF pass. On a polars
input the copy is made on the zero-copy Arrow pandas VIEW built at `:534`, which materialises the whole frame
that the bridge exists specifically to avoid.
**Fix:** `chained = _X_for_recipes.copy(deep=False)` — a shallow frame shares the caller's blocks, and
adding a NEW column to the shallow copy does not touch the caller's frame (the only mutation performed here);
existing columns are never written. Pin with an aliasing test rather than reverting to a deep copy.
**Test:** `test_append_engineered_does_not_deep_copy_input_frame` — fit with at least one recipe, then call
`transform` on a frame whose column block is identified by `id(df['a']._values.base)`/`np.shares_memory`, and
assert the base column still shares memory after the call AND that the caller's frame gained no new columns.

### PERIPHERY-3 — the cross-target identity cache is keyed on X (and optionally y) but NOT on the selector's parameters  [P1]
**Where:** `_mrmr_fingerprints.py:145-240` (`_mrmr_compute_x_fingerprint`) consumed at
`mrmr/_mrmr_class.py:3605-3650`
**What:** `_x_fp` is `blake2b(repr((cols, n_rows, dtypes_repr, cell_sample)))`, optionally suffixed with a
y-sample fingerprint. No constructor parameter enters the key. On a hit the estimator calls
`self._fit_identity_shortcut(X)` (`:3647`) and returns — `support_` becomes "everything".
**Why it is wrong / costly:** this is the brief's point (b): the fingerprint does not cover every input that
changes the answer. Two MRMR instances with different `max_features` / `n_features_to_select`,
`quantization_nbins`, `nbins_strategy`, `mrmr_relevance_algo`, any `fe_*` flag, or `dcd_enable` share one
cache slot. A permissive config that legitimately returned identity therefore licenses a STRICTER config —
one that would have dropped features — to skip the entire fit and select every column it never scored. Note
that the sibling cache layers got exactly this fix already: `_fit_impl_core.py:200-231` folds
`_hashable_params_signature` into the in-object signature, and `:248,267` folds it into `_FIT_CACHE`. The
identity cache is the one layer left with the asymmetric (weaker) guarantee — the same bug class the file's
own comments record fixing twice elsewhere. Blast radius is gated by `mrmr_skip_when_prior_was_identity`.
**Fix:** fold `_hashable_params_signature(self._pre_fit_ctor_params_snapshot_ or self.get_params(deep=True))`
into `_x_fp` at `_mrmr_class.py:3607`, exactly as the other two layers do.
**Test:** `test_identity_cache_does_not_cross_configs` — with `mrmr_skip_when_prior_was_identity=True`, fit a
permissive MRMR that returns identity on X, then fit a second instance on the SAME X with a config that must
drop features (e.g. `n_features_to_select=1`); assert the second `support_` has length 1, not `n_cols`.

### PERIPHERY-4 — "cheap O(1)" fingerprints materialise a full dense copy of X, twice per fit  [P1]
**Where:** `_mrmr_fingerprints.py:280-330` (`_content_array_signature`, `:300-302` `arr.to_numpy()`) and
`:430-492` (`_full_x_content_hash`, `:448-450` `to_numpy()`, `:472` `np.ascontiguousarray`)
**What:** `_content_array_signature`'s docstring says "Cheap O(1) content-based fingerprint … Samples 10
evenly-spaced positions", and the `_FIT_CACHE` comment at `_fit_impl_core.py:244-245` repeats "cheap O(1)".
The implementation calls `X.to_numpy()` on the WHOLE frame before sampling 1024 cells (`:315-322`).
`_full_x_content_hash` does the same and can add a second copy via `ascontiguousarray`. Both are called on X
in the same fit (`_fit_impl_core.py:199, 249, 257`).
**Why it is wrong / costly:** for a mixed-dtype pandas frame `to_numpy()` upcasts every column into one dense
block (often float64 or object) — a full-frame allocation, which CLAUDE.md forbids, incurred merely to read
1024 cells. On a 100 GB frame the "cheap" discriminator is the dominant allocation of the fit and can OOM
before MRMR's own memory guard at `_mrmr_validate_transform.py:171-190` (which itself only budgets the binned
working set) ever runs. The full-content hash genuinely needs all bytes, but it should stream per column, not
build a dense upcast plane.
**Fix:** sample per column (the fingerprint already has the column list) — read the 1024 positions from each
`X[c]` Series without a frame-wide `to_numpy()`; for `_full_x_content_hash`, update the blake2b incrementally
per column buffer. Correct the docstrings either way: they currently assert a cost the code does not have.
**Test:** `test_content_signature_does_not_materialise_whole_frame` — monkeypatch `pd.DataFrame.to_numpy` to
raise, and assert both `_content_array_signature` and `_full_x_content_hash` still return a usable
(non-`"uncached"`, non-empty) result on a mixed int/float/category frame.

### PERIPHERY-5 — the single-slot X-hash memo has a torn-read race that can return another frame's digest  [P1]
**Where:** `_mrmr_fingerprints.py:427, 467-489`
**What:** the memo is read unlocked (`:468-469`: compare `["id_shape"]`, then read `["hash"]`) and written
under `_MRMR_IDENTITY_FP_LOCK` in the order hash-then-key (`:486-488`). The comment claims the ordering means
"even a torn read … can only see an OLD key paired with whatever hash — a miss that recomputes".
**Why it is wrong / costly:** the claim is false for the interleaving that matters. Reader (frame A) evaluates
`_MRMR_LAST_X_HASH_CACHE["id_shape"] == id_shape_A` → True; writer (frame B) then executes both writes;
reader now executes `return str(_MRMR_LAST_X_HASH_CACHE["hash"])` and gets **B's digest for A**. Publishing
the hash first makes this MORE likely, not less. The returned digest feeds the `_FIT_CACHE` key
(`_fit_impl_core.py:257,267`) and the in-object skip signature (`:199,231`), so the consequence is a wrong
cache identity — either a stale replay of another frame's `support_`, or a spurious miss. MRMR is fitted
under joblib threading in the suite, so concurrent `_full_x_content_hash` calls are reachable.
**Fix:** read both fields under the same lock (one `with` block returning a local tuple), or store the pair in
ONE immutable tuple slot so a single atomic dict read gets a consistent (key, hash) pair.
**Test:** `test_full_x_content_hash_memo_is_not_torn_under_threads` — two threads hashing two distinct frames
in a loop; assert each thread always receives the digest of a fresh single-threaded `blake2b` over its own
frame (run enough iterations to exercise the interleave, with a monkeypatched sleep inside the read to force
it deterministically).

### PERIPHERY-6 — `transform()` never checks dtype drift against fit time; fit-time dtypes are not stored  [P2]
**Where:** `_mrmr_validate_transform.py:305-452`; no dtype capture anywhere in `_validate_inputs`
(`:156-302`)
**What:** transform validates column PRESENCE (`:381`, `:409`) and, for unnamed arrays, WIDTH (`:338-348`).
There is no comparison of per-column dtypes against fit. `_mrmr_compute_x_fingerprint` computes canonical
dtypes for the cache key (`:177-196`) but that value is never persisted as `self._fit_dtypes_` for transform
to check.
**Why it is wrong / costly:** the brief's enumerated drift cases all pass silently. A fit-time `int64` column
arriving as `float64` changes nothing structurally but changes replayed engineered values where a recipe
keys on integer codes; a fit-time `category`/`pl.Enum` arriving as `object`/string changes `factorize` and
`target_encoding` lookup keys, so the replayed column is built from different codes than fit; a numeric
column that arrives ALL-NaN produces an all-NaN engineered column and an all-NaN selected column with no
warning at all — the downstream model consumes it as a live feature. None of these raise, none log above
debug; the result is wrong values, not a crash.
**Fix:** store `self._fit_dtypes_ = {name: canonicalise_dtype(dt)}` at the end of fit (the canonicaliser is
already imported in `_mrmr_fingerprints.py:33`), and at transform compare the SELECTED columns plus every
`recipe.src_names`: raise on a category/object↔numeric kind change, `logger.warning` on a widening within the
numeric kinds, and `logger.warning` when a column that had any finite value at fit arrives all-NaN.
**Test:** `test_transform_raises_on_category_to_object_dtype_drift` and
`test_transform_warns_when_fitted_numeric_column_arrives_all_nan` — fit on a frame with a `category` column
consumed by a `factorize`/`target_encoding` recipe, transform the same rows with that column cast to `object`
(and separately with one numeric column set to all-NaN); assert the raise / the warning, and that the
engineered column is not silently different from the fit-transform output.

### PERIPHERY-7 — extra / unseen columns at transform time are silently accepted  [P2]
**Where:** `_mrmr_validate_transform.py:338-348` and `:381-392`
**What:** the width check is skipped for any frame with names (`_is_named_frame`), and the name check is
one-directional: it only asks whether each SELECTED column is present. A transform frame carrying extra
columns MRMR never saw passes unremarked.
**Why it is wrong / costly:** sklearn's canonical contract is that a named-frame transform validates the full
feature-name set (`_check_feature_names`) and raises/warns on unseen or reordered names. Here an upstream
step that ADDS a column (a new FE stage, a merge that duplicated a key) produces no signal at all, and
combined with PERIPHERY-1 a width-preserving add+drop is completely invisible. The comment at `:340-345` is
right that the by-name error is more actionable — but that error never fires for extras.
**Fix:** after the missing-column check, compare `set(X.columns)` against `set(self.feature_names_in_)` and
`logger.warning` (naming up to 8) on unseen columns; keep `RuntimeError` for missing ones.
**Test:** `test_transform_warns_on_unseen_columns` — fit on `[a,b,c]`, transform on `[a,b,c,d]`, assert a
WARNING naming `d` and that the output is unchanged otherwise.

### PERIPHERY-8 — the "selection-stability" statistic replays RELEVANCE ranking only, but is reported against a redundancy-aware point selection  [P2]
**Where:** `_mrmr_stability_report.py:130-142`, docstring `:26-40` and `:63-73`; legend `:241-245`
**What:** per resample the report computes `rel[c] = _marginal_mi_codes(...)` for every candidate and takes
`np.argpartition(rel, ...)`'s top-`n_selected` as "selected on this resample" (`:136-139`). No redundancy /
mRMR term is applied. The `*` column in the rendered table (`:231`) marks `selected_mask`, which came from
the real fit — a greedy mRMR selection that DID subtract redundancy.
**Why it is wrong / costly:** the two are different estimators, so the report compares a feature against a
yardstick that was never used to pick it. A redundant-but-highly-relevant feature that mRMR deliberately
DROPPED scores ~1.0 selection-frequency (it is always top-k by marginal MI) and is presented as a
high-confidence survivor that is not starred; conversely a genuinely-selected feature that mRMR admitted for
its low redundancy despite middling relevance scores low and is rendered `low` confidence next to its `*`.
The module header states it "recomputes … ranks the candidates, and records which would have been selected",
and the legend says "ranked in the selected top-set" — both read as replaying MRMR's decision. This is a
second implementation of the selection rule that has drifted from the real one; the primitive both sides
should share is the greedy mRMR step in `_mrmr_fit_impl` (relevance minus the configured
`redundancy_aggregator` over the already-picked set), not bare `_marginal_mi`.
**Fix:** either (a) replay the actual greedy loop on the resample using the stored codes (the redundancy term
is computable from the same `cand_codes`), or (b) if the cheap approximation is intended, rename the metric to
`relevance_rank_frequency` and say so in the docstring, header, and the rendered legend — do not call it
selection-frequency, and do not star the mRMR point selection against it.
**Test:** `test_stability_report_statistic_matches_its_documented_definition` — build a fixture with two
perfectly-duplicated strong features (mRMR selects one, drops the other); assert the dropped duplicate is not
reported at a higher selection-frequency than the one that was actually selected.

### PERIPHERY-9 — a NaN relevance makes a degenerate feature rank as "always selected"  [P2]
**Where:** `_mrmr_stability_report.py:134-139`
**What:** `rel` is filled with `_marginal_mi_codes(...)` per candidate and passed to `np.argpartition`
unguarded. `np.argpartition` places NaN at the HIGH end of the partition.
**Why it is wrong / costly:** any candidate whose replayed MI is NaN on a resample (a column that becomes
single-valued in that bootstrap, or any degeneracy returning NaN from `_cmi_from_binned`) is placed inside the
top-`n_selected` on every such resample, so it reports selection-frequency ≈ 1.0 and renders as `HIGH`
confidence. The report's whole purpose is separating genuine signal from chance, and this inverts the verdict
for exactly the degenerate columns it should flag. Unverified whether `_cmi_from_binned` can currently return
NaN on a constant column — reading it (or a targeted call with a constant `x_codes`) would settle it; the
guard is cheap and correct either way.
**Fix:** `rel = np.nan_to_num(rel, nan=-np.inf)` (or mask NaN out of the candidate pool for that resample)
before the `argpartition`, and count a NaN candidate as not-selected.
**Test:** `test_stability_report_nan_relevance_is_not_counted_as_selected` — monkeypatch the marginal-MI
primitive to return NaN for one candidate; assert its reported frequency is 0.0, not 1.0.

### PERIPHERY-10 — the stability replay is a per-candidate Python loop over a per-candidate fancy-index copy  [P2]
**Where:** `_mrmr_stability_report.py:130-139`, and `:204-215` for the recipe loop
**What:** for each of K resamples the code loops `for c in range(n_cand)` and calls
`_marginal_mi_codes(cand_codes[idx, c], y_b)`. `cand_codes[idx, c]` is a fresh gather+allocation per
candidate per resample: `K * n_cand` allocations of length `n_rows`. `_marginal_mi` dispatches into
`_cmi_from_binned` (`_fe_stability_vote.py:89-91`) once per call. Grep for the brief's REJECT gate: there is
no `@njit`, `prange`, `parallel=True`, `cuda.jit`, `cupy`, or `KernelTuningCache` anywhere in
`_mrmr_stability_report.py`, and no bench-attempt-rejected note — so this path is genuinely unoptimised.
**Why it is wrong / costly:** this is the repo's signature pattern (CLAUDE.md "GIL-bound per-resample Python
dispatch loops"): RNG draw + gather + per-candidate Python dispatch around an already-compiled kernel. At the
default `n_boot=50` and a few hundred candidates that is ~10^4 kernel dispatches and ~10^4 full-length
gathers, all serial.
**Fix:** two levels. Cheap: gather `cand_codes[idx]` ONCE per resample (one `(n_rows, n_cand)` take) and
slice columns as views, killing `K*n_cand` gathers. Real: materialise all K index arrays in one vectorised
`rng.integers((K, n_rows))` draw and run the whole bin-count/MI body inside one
`njit(parallel=True)` + `prange` over (resample, candidate). Bench at K∈{50,500}, n_cand∈{50,500},
n_rows∈{10k,1M}; save to `_benchmarks/`; gate acceptance on identical frequencies given the same draw order.
No speedup number is claimed here — none was measured.
**Test:** `test_stability_report_frequencies_unchanged_after_fusion` — pin the reported frequency dict for a
fixed seed against the current implementation before/after the rewrite.

### PERIPHERY-11 — `_content_array_signature`'s final fallback returns an `id()`-derived key  [P2]
**Where:** `_mrmr_fingerprints.py:327-329` (`return ("uncached", id(arr))`)
**What:** every other failure path in the same function returns `("uncached", uuid4().hex)` (`:305, 311,
325`), and both fingerprint functions carry long comments (`:134-142`, `:233-240`) explaining that an
`id()`-keyed fallback is unsafe because CPython recycles addresses. The outermost handler still uses `id()`.
**Why it is wrong / costly:** the returned tuple goes straight into the `_FIT_CACHE` key
(`_fit_impl_core.py:249-250,267`). A frame built after an earlier one was collected very commonly reuses the
address, so two DIFFERENT X can produce the same `("uncached", id)` component — and the rest of the key can
match when the full hashes also fail. It is the exact scenario the module's own comments say must never
happen, left alive in one branch. The prior wave (`audits/mrmr_audit_2026-07-25/stability_misc.md:34`) marked
this file "Clean for determinism"; this branch was not covered by that check.
**Fix:** return `("uncached", uuid4().hex)`, matching the sibling branches.
**Test:** `test_content_array_signature_fallback_never_keys_on_id` — force the outer exception (an object
whose `shape` access raises) and assert two successive calls return different second components.

### PERIPHERY-12 — the two cache layers disagree on `get_params(deep=True)` vs `deep=False`  [P2]
**Where:** `_fit_impl_core.py:227` (`deep=True`) vs `:248` (`deep=False`), both consuming
`_mrmr_fingerprints._hashable_params_signature`
**What:** the in-object skip signature expands nested estimator params; the process-wide `_FIT_CACHE` key does
not.
**Why it is wrong / costly:** in-place mutation of a nested `get_params`-bearing object (a nested estimator or
config passed as a constructor arg) invalidates the in-object skip but NOT the `_FIT_CACHE` entry — so a
clone with a mutated nested config gets a replay computed under the old nested settings. Same asymmetric-
guarantee bug class the `:200-211` comment records fixing for the params-absent case.
**Fix:** use `deep=True` in both, or document why the nested expansion is provably irrelevant to the fitted
result.
**Test:** `test_fit_cache_key_covers_nested_param_mutation` — fit, mutate a nested config object in place,
fit a clone on the same (X, y), assert the second selection reflects the new nested setting.

### PERIPHERY-13 — `explain_selection` reports every provenance row as a "surviving selected feature", and presents in-screen gains as the attribution  [P2]
**Where:** `_mrmr_explain.py:99-139`
**What:** `n_total = len(prov)` and the line rendered is `f"Surviving features: {n_total} selected"`. The
code's own comment two lines below (`:109-111`) states that "screened-out produced columns carry NaN" in
`mrmr_gain` — i.e. `fe_provenance_` contains rows that were NOT selected. When `mrmr_gain` is absent entirely
(`:123-124`) the roster is the unfiltered provenance frame, and `n_eng`/`n_total - n_eng` are likewise counts
over all produced columns.
**Why it is wrong / costly:** the headline number of the user-facing explanation over-reports the selection
size (by however many FE candidates were produced-then-screened), and the "(N engineered, M raw)" split is
wrong with it. The authoritative selection is `support_` / `feature_names_out`, which the section never
consults. Second, the ranked roster is ordered by `mrmr_gain` — the in-screen greedy gain, computed on the
same data the feature was chosen on (winner's-curse biased) — and is labelled "by MI/gain attribution" with a
4-decimal number and no honesty qualifier. Per the brief and this repo's val/test/OOF rule, an in-screen
number presented as the headline attribution is a finding.
**Fix:** filter the roster to rows whose feature is in the actual selection (or whose `mrmr_gain` is non-NaN)
before counting, and cross-check `n_total` against `len(self.support_)`; label the gain column explicitly as
in-screen/optimistically-biased (e.g. `gain(in-screen)=`) in both the roster and the legend.
**Test:** `test_explain_selection_counts_only_selected_features` — fit with FE on a fixture where at least one
produced column is screened out; assert the reported count equals `len(mrmr.get_feature_names_out())` and that
the screened-out name does not appear in the roster.

### PERIPHERY-14 — transform's recipe replay never passes the `col_cache` / `basis_cache` that `apply_recipe` ships for exactly this loop  [P2]
**Where:** `_mrmr_validate_transform.py:601` and `:673`; the parameters at
`engineered_recipes/_recipe_dispatch.py:28-38`
**What:** `apply_recipe` accepts `col_cache` / `basis_cache`, documented as "caller-owned dicts scoped to ONE
`transform()`/`predict()` call, shared across every recipe in that call's replay list", deduping
`_extract_column` for hub source columns and the orth-basis polynomial evaluation for shared operands. Both
call sites in `_append_engineered` pass neither, so both default to `None` = always recompute. The docstring
even notes "every EXISTING caller (none of which construct these dicts yet)".
**Why it is wrong / costly:** the production transform path is the primary intended beneficiary, and it is
the one caller that does not use it. A hub column referenced by many recipes is extracted once per recipe, and
a shared orth operand's basis is re-evaluated once per recipe — pure repeated work on the hot path
(CLAUDE.md's documented "already-optimized primitive, just not wired into the call site" pattern).
**Fix:** allocate `_col_cache: dict = {}` / `_basis_cache: dict = {}` once at the top of `_append_engineered`
and thread them into both `apply_recipe` calls. Bench a fit with many recipes sharing a hub operand
(measure, do not assume, the gain).
**Test:** `test_append_engineered_shares_col_cache_across_recipes` — spy on `_extract_column` and assert a
hub source column is pulled once, not once per recipe, while output values stay bit-identical.

### PERIPHERY-15 — `mrmr_rank_bases` mixes an unnormalised redundancy MI with relevance MI at `beta=1.0`  [P2]
**Where:** `_mrmr_base_rank.py:29-87`, especially `:79-82`; consumer `_auto_base.py:679, 688`
**What:** the score is `rel[i] - beta * mean_p MI(i, p)` with `beta` defaulting to 1.0. `relevance` is
`MI(base, y)` and redundancy is `_mi_pair_bin(col_i, col_j)` — two MIs on different natural scales (feature-
feature MI between two continuous binned columns is routinely much larger than feature-target MI, especially
for a low-cardinality y).
**Why it is wrong / costly:** this is MID with raw, unnormalised terms. When the redundancy term dominates,
the ranking is driven almost entirely by diversity and can push the genuinely most-relevant bases down the
shortlist — the opposite of the stated intent ("trades a LITTLE relevance for diversity", `_auto_base.py:668`).
Classic mRMR implementations either normalise both to [0,1] (symmetric uncertainty) or use the quotient form
(MIQ). Marked P2 rather than P1 because I did not measure the actual scale ratio on real composite candidates;
a bench over the real `_auto_base` candidate pools (print `rel` and `red_sum/n_picked` side by side) would
settle whether the default inverts the ordering in practice.
**Fix:** normalise both terms (SU, or divide each MI by `min(H(a), H(b))`) before combining, or expose and
document the scale sensitivity of `base_ranking_mrmr_beta` with a measured default. Whichever is chosen, note
it in the docstring — it currently presents the raw formula as if the two terms were commensurable.
**Test:** `test_biz_val_mrmr_base_rank_keeps_top_relevance_first_under_scale_mismatch` — relevance in
[0, 0.05] (binary y), redundancy in [0, 2.0] (two continuous columns); assert the highest-relevance
non-duplicate candidate still ranks in the top 2.

### PERIPHERY-16 — the object-dtype inf scan runs a per-element Python callable over every object column  [P3]
**Where:** `_mrmr_validate_transform.py:270-278`
**What:** `np.frompyfunc(lambda v: isinstance(v, (float, np.floating)) and np.isinf(v), 1, 1)(_obj_col_arr)`
invokes a Python lambda once per cell, then `.astype(bool)` materialises a full boolean array per column, with
no early exit (unlike the float path at `:258-262`, which does exit early).
**Why it is wrong / costly:** on a 100 M-row frame with a handful of object columns this is hundreds of
millions of Python-level calls per fit, plus one full bool array per column. Object columns are exactly where
low-cardinality strings live, so this is common, not exotic.
**Fix:** short-circuit on the first hit (iterate a chunked view and `break`), and skip columns whose inferred
pandas type is not float-bearing (`pd.api.types.infer_dtype(col) in {"string","unicode","categorical","bytes"}`
cannot hold a float inf) before scanning at all.
**Test:** `test_validate_inputs_object_inf_scan_short_circuits` — a large object column with `inf` in the
first row; assert the per-element predicate is invoked far fewer than `n_rows` times.

### PERIPHERY-17 — `n_boot` is silently coerced rather than validated, and is unbounded above  [P3]
**Where:** `_mrmr_stability_report.py:54, 123` (`K = max(1, int(n_boot))`)
**What:** `n_boot=0` or a negative value silently becomes 1 and the report renders as if one resample were the
user's request; a non-integer raises deep inside `int()`. There is no upper bound (each resample is a full
`K * n_cand` MI sweep — see PERIPHERY-10).
**Why it is wrong / costly:** a caller asking for 0 resamples gets a frequency table of 0.0/1.0 values that
reads like a real confidence readout. Minor, but it is a confidence report — a silently-degraded K is exactly
the input whose corruption the reader cannot detect.
**Fix:** `raise ValueError` for `n_boot < 1`; keep the default at 50 (sane) and log at `info` when K times
n_cand exceeds a coarse work budget.
**Test:** `test_selection_stability_report_rejects_nonpositive_n_boot`.

### PERIPHERY-18 — y-fingerprint on a non-numeric target fails on every call: a warning per fit and a permanently-dead cache  [P3]
**Where:** `_mrmr_fingerprints.py:132` (`sample.astype(np.float64)`), handler `:134-142`
**What:** for a string / object / non-castable categorical y, `astype(np.float64)` raises; the handler logs a
WARNING and returns `f"yfp_uncacheable_{uuid4().hex}"`.
**Why it is wrong / costly:** the disable-on-failure direction is correct and deliberate, but the trigger is a
perfectly ordinary target type, not an exceptional condition: `mrmr_identity_cache_include_y=True` with a
string-labelled classification target emits a WARNING on every fit and can never produce a cache hit. The
sibling `_mrmr_y_corr_sample` (`:366-369`) already handles this by factorizing non-numeric samples to codes.
**Fix:** reuse the same factorize path in `_mrmr_compute_y_fingerprint_sample` before the `astype`, keeping the
uuid fallback for genuinely unhandleable targets.
**Test:** `test_y_fingerprint_stable_for_string_labels` — the same string-label y hashes to the same value
across two calls, and a different label vector hashes differently.

### PERIPHERY-19 — float sign/NaN payloads in the cell sample are bit-keyed: `-0.0`/`0.0` and distinct NaN payloads hash apart  [P3]
**Where:** `_mrmr_fingerprints.py:222` (`np.asarray(arr[p]).tobytes()`) and `:132` (y sample `tobytes()`)
**What:** the sample is hashed from raw bytes, so `-0.0` and `0.0` (equal under `==`) produce different
fingerprints, as do NaNs with different payloads / signs.
**Why it is wrong / costly:** the failure direction is a cache MISS (an extra full fit), never a wrong hit —
which is why this is P3 and not higher, and the switch to `tobytes()` over the prior rounding is correct for
the collision direction the comment at `:131` and `:221` argues. Worth recording explicitly so the next wave
does not re-derive it: the fingerprints are bit-stable within a process and across processes (blake2b, no
`hash()` of a str, no `id()` except PERIPHERY-11, `repr` of a tuple not a dict — the JSON `sort_keys` rule is
N/A here since nothing is JSON-serialised), and the only non-determinism is the byte-level float edge cases.
**Fix:** none required; if a miss on sign-of-zero ever shows up in practice, normalise with `x + 0.0` on the
sampled buffer before `tobytes()`.
**Test:** `test_x_fingerprint_is_bitstable_and_process_independent` — assert a frame's fingerprint matches a
value computed in a subprocess launched with a different `PYTHONHASHSEED`.

### PERIPHERY-20 — audit/process metadata still embedded in comments, against CLAUDE.md's comment rule  [P3]
**Where:** `_mrmr_validate_transform.py:32-33` ("fix audit row FS-P2-1"), `:46`, `:54`, `:58`, `:69`, `:75-76`,
`:124`, `:126-127`; `_mrmr_fingerprints.py:83`, `:110`, `:250`, `:262`, `:414-426` ("iter627", "iter625",
"iter59"), `:441-444`, `:466`; `_mrmr_explain.py:8-9` ("Layer-? … (b1a1048a)"), `:10-13`, `:165`, `:171`;
`_mrmr_stability_report.py:1` ("backlog W3, 2026-06-11"), `:11-14`, `:114-119` ("(P1): this used to …")
**What:** finding IDs, wave/layer/iteration markers, commit SHAs and date stamps in code comments.
**Why it is wrong / costly:** CLAUDE.md's comment-style rule bans exactly this; that belongs in git history /
the PR description. The prior wave raised this for this cluster as STABILITY_MISC-3 listing
`_mrmr_validate_transform.py:44,52,94,228,318,697` and `_mrmr_explain.py:232` — it was NOT applied; the same
class of marker is still present at the (shifted) lines above.
**Fix:** strip the ID/date/wave/iteration narration; keep the WHY each comment carries.
**Test:** a meta-test already fits this shape — `test_no_audit_metadata_in_comments` over the MRMR package,
grepping `Wave [0-9]|Layer-?[0-9]|iter[0-9]{2,}|[A-Z_]+-[0-9]+ fix|\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\b`.

### PERIPHERY-21 — `_replay_recipe_survival` silently skips recipes whose stored codes do not match y's length  [P3]
**Where:** `_mrmr_stability_report.py:196-203` (`if rs is None: continue`) and `:202-203`
(`if eng.shape[0] != n: continue`)
**What:** both skips are silent — no log at any level — and the skipped recipe is simply absent from
`recipe_survival_frequency`, which the formatter renders as "this recipe was not part of the report" rather
than "we could not evaluate it".
**Why it is wrong / costly:** a length mismatch between stored recipe codes and `y_codes` indicates the replay
state is inconsistent with the fit (a real bug upstream), and it is reported to the user as a shorter table.
Per the brief's rule 4, a handler that substitutes a value and returns without logging above debug is a
finding.
**Fix:** `logger.warning` naming the recipe and the two lengths on the shape-mismatch branch; `logger.debug`
is acceptable for the plain "no state stored" branch, but the formatter should say how many recipes were
skipped.
**Test:** `test_recipe_survival_warns_on_stale_replay_state` — corrupt one recipe's `eng_codes` length in
`_stability_replay_state_`; assert a WARNING naming the recipe.

## Proposed tests (beyond the per-finding ones)

- `test_transform_on_row_subset_replays_frozen_fit_params` — fit on the full frame, then `transform(X.iloc[:100])`
  and `transform(X)`; assert the first 100 rows of both outputs are bit-identical for every engineered column.
  This is the leak/drift invariant the brief prioritises; I found NO refit on the transform path (see
  Verified-clean), but nothing pins it, so a future recipe that re-derives a quantile from the transform frame
  would slip through.
- `test_transform_output_columns_match_get_feature_names_out` — for pandas, polars and ndarray inputs, assert
  the transform output width AND (where named) order equals `get_feature_names_out()`, including the
  zero-column degradation path at `_mrmr_validate_transform.py:647`.
- `test_transform_unresolvable_chained_recipe_emits_zero_not_nan` — pin the documented degradation
  (`:641-648`): a chained recipe whose producer was pruned yields an all-zero column, a throttled WARNING
  fires, and the output width still matches `get_feature_names_out()`.
- `test_polars_and_pandas_transform_are_value_identical` — the same logical frame as pandas and polars through
  the same fitted MRMR; assert equal values and equal column names (the two branches at `:372-396` and
  `:397-419` name engineered columns through `simplified_recipe_names` separately).
- `test_replay_fitted_state_isolates_mutable_fitted_attrs` — mutate a replayed instance's `support_`,
  `ranking_`, and a dict-valued fitted attribute; assert the `_FIT_CACHE` source instance is unchanged
  (pins `_mrmr_fingerprints.py:549-586`).
- `test_explain_selection_never_raises_on_corrupt_artifacts` — set `fe_provenance_` to a non-DataFrame, a
  DataFrame missing `origin`, and a ledger missing `margin`; assert a string is returned each time and a debug
  log fired (pins the never-raise contract at `_mrmr_explain.py:238-262` behaviourally rather than by source
  inspection).
- `test_stability_report_is_reproducible_per_seed` — two calls with the same `random_state` return identical
  frequency dicts; different seeds differ. The seeding is explicit and correct today
  (`_mrmr_stability_report.py:120-121, 156`) but untested.

## Prior-wave findings touching this cluster

| ID (wave) | Status in current source |
|---|---|
| STABILITY_MISC-3 (2026-07-25, comment metadata incl. `_mrmr_validate_transform.py:44,52,94,228,318,697` and `_mrmr_explain.py:232`) | **STILL HOLDS** — not applied; re-reported with current line numbers as PERIPHERY-20. |
| "`_mrmr_fingerprints.py` content-hash determinism … Clean for determinism" (2026-07-25 non-finding, `stability_misc.md:34`) | **Partially contradicted** — the blake2b/`tobytes` claims hold, but the check did not cover the `id(arr)` fallback (PERIPHERY-11), the unlocked memo read (PERIPHERY-5), the full-frame `to_numpy()` cost (PERIPHERY-4), or the params-absent identity key (PERIPHERY-3). |
| USABILITY_B-5 (replay-freeze side effect, noted still open) | Still present by design: `_replay_fitted_state` flips `v.flags.writeable = False` on the SOURCE array (`_mrmr_fingerprints.py:568-570`), so the cached instance's own arrays become read-only as a side effect of a replay. `support_`/`ranking_` are exempted via a writeable copy (`:562-564`). Unchanged from the prior report; not re-scored here. |
| "`_mrmr_stability_report` P1: bootstrap reseeded at 0 instead of `random_state`" (recorded in-code at `:114-119`) | **FIXED and holds** — `:120` resolves via `_effective_random_seed()`. |

## Verified-clean

- **`transform()` does not refit.** Every kind in `apply_recipe` (`engineered_recipes/_recipe_dispatch.py:39-90+`)
  routes to a `_apply_*` replay helper driven by recipe-stored parameters; no y is threaded into the transform
  path (`transform(self, X, y=None)` ignores `y`). A row SUBSET therefore replays frozen fit params — no drift,
  no leak. Untested, hence the proposed test above.
- **Missing selected columns raise, in BOTH named-frame branches** (`:381-392` pandas, `:409-418` polars), and
  the polars branch correctly remaps positional `support` to fit-time names before selecting, so a reordered
  or narrowed polars frame is handled by name rather than position.
- **Empty / degenerate support** is handled: `len(support)==0 and not recipes` returns an empty-column frame
  (`:366-370`); a float empty support array from an old pickle is coerced to `intp` (`:427-429`); an empty
  `support` with recipes falls through and emits only engineered columns.
- **ndarray dtype promotion on the hstack path** (`:718-724`) uses `np.result_type` on both sides, so float
  engineered columns are no longer truncated into an int base — verified present and correct.
- **`_replay_fitted_state` isolation contract** (`_mrmr_fingerprints.py:540-588`): deep-copies everything that
  is not an immutable scalar/ndarray, hands out writeable copies of `support_`/`ranking_`, and excludes the
  per-instance re-entrancy lock and the ctor snapshot. Sound.
- **Fingerprint process-stability**: no `hash()` of a str, no JSON (so the `sort_keys` rule is N/A), no `id()`
  outside PERIPHERY-11; `_hashable_params_signature` sorts its items (`:257`) and content-hashes ndarrays by
  `(tobytes, shape, dtype)`. Digests are blake2b over bytes → identical across processes and PYTHONHASHSEED
  values.
- **`_lazy_chunks`** (`:591-605`): correct O(chunk) streaming, validates `chunk_size >= 1`.
- **`_mrmr_base_rank` greedy mechanics**: the running `red_sum` update (`:76, 85-86`) is a correct incremental
  mean over the picked set; `n_picked` is read before the append so the divisor matches the set summed;
  tie-breaks are deterministic (`-i` → lowest index); `k<=0`, `n==0`, `k>n`, and length/shape mismatches are all
  handled with explicit `ValueError`s (`:48-58`) and covered by
  `tests/training/composite/discovery/test_biz_val_mrmr_base_rank.py:85-105`. The only concern is the scale
  mismatch of PERIPHERY-15.
- **The `_mrmr_base_rank` consumer** (`_auto_base.py:673-697`) uses only the public `mrmr_rank_bases` signature,
  reaches into no private MRMR attribute, is gated behind `base_ranking_criterion == "mrmr"`, guards
  `len(ranked) > 1`, and handles the "nothing to rank" case by leaving `ranked` untouched — so an empty or
  single-candidate pool never reaches the ranker. Clean.
- **Explainability is assembly-only**: `_mrmr_explain.py` reads `fe_provenance_`, `fe_rejection_ledger_` and
  `_fe_recommended_flags_` and recomputes no selection — the "second implementation" risk the brief warns about
  does NOT apply here (it DOES apply to `_mrmr_stability_report.py`, see PERIPHERY-8). The what-if-flip band
  arithmetic (`:204`) matches its documented `-delta < margin < 0` definition exactly.
- **Numerical-stability sweep**: no `sum(x**k)`-minus-a-power-of-the-mean site and no additive-epsilon
  denominator padding anywhere in this cluster. The only arithmetic is `np.corrcoef`
  (`_mrmr_fingerprints.py:389`), integer counting, and formatting.
- **Silent-fallback sweep**: every `except` in the cluster either re-raises, logs at WARNING where the
  substituted value changes the caller's answer (`_mrmr_fingerprints.py:141, 239`;
  `_mrmr_validate_transform.py:511`), or logs at debug where the substitution is neutral — with the two
  exceptions reported as PERIPHERY-21 (silent skips) and the acknowledged debug-only guards at
  `_mrmr_validate_transform.py:282, 301` (which skip a VALIDATION guard, not a value substitution;
  borderline, noted here rather than as a separate finding).
