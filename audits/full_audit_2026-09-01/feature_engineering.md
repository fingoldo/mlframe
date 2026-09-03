# feature_engineering
Files reviewed: 32 read in depth (of 314 in the cluster; all 314 grep-scanned for the target bug classes) | LOC: 9,867 read of 52,158 in cluster

## Summary
The cluster's headline leakage risks are in good shape: the previously-confirmed `knn_aggregate` self-match bug in `spatial.py` is fixed (lines 241-259 now compact out the first distance-0 hit before truncating to k), `cross_sectional_neighbors.py` masks the self-match placeholder correctly, the `transformer/` OOF family (`neighbor_aggregate_features`, `target_quantile`, `y_quintile_baseline_knn`, `local_lift`) builds its index per-fold on train rows only, and broad `except Exception` handlers across the cluster are now almost universally logged. What the audit did find is a different class: silent wrong numbers with no exception. Two were reproduced live. `_numerical_numba.py`'s weighted skew/kurtosis normalise by row COUNT where the same function's weighted mad/std correctly normalise by SUM OF WEIGHTS -- with weights summing to 1 the emitted `wskew`/`wkurt` features are off by a factor of `n` (measured 0.005x at n=200; `wkurt` collapses to the degenerate ~-3.0 constant the file's own comment at line 666 warns about for an earlier bug in the same block). `binned_unique_count.py` mis-aligns its output rows whenever the entity column contains a null, silently handing entity B's count to the NaN row and 0 to B (reproduced). Beyond those: a documented `order` default in `two_step_target_encode.py` that the code does not implement (turning the "leak-free step 1" into row-order encoding on an unsorted frame), a horizon-selector in `multi_window_aggregate.py` whose no-feature baseline is a `DummyClassifier` even for a regression target (making the first candidate horizon unconditionally "pass"), and a small set of additive-epsilon and tie-handling issues.

## Findings

### FEATURE_ENGINEERING-1 [P1] silent-correctness
**File:** src/mlframe/feature_engineering/_numerical_numba.py :726, :731 (guard at :723)
**Summary:** Weighted skew and weighted kurtosis divide by `size` (the row count) instead of `sum_weights`, so both features are wrong by an arbitrary factor whenever the weights do not sum to `n`.
**Failure scenario:** `compute_numaggs(arr, weights=w)` with `w = np.ones(200)/200` (weights normalised to sum to 1 -- the natural convention). Reproduced live: true `wskew = 0.256299`, emitted `wskew = 1.2815e-03` (exactly 1/200 of the truth); true `wkurt = -0.546955`, emitted `wkurt = -2.987735`. The unweighted `skew` in the same call is correct (0.256299), so a caller comparing the two sees a plausible-looking small number, not an obvious failure. `wkurt` degrading toward the constant -3.0 is the exact failure signature the file's own comment at lines 666-667 records for a previous bug in this block.
**Suggested fix:** In `_make_compute_moments_slope_mi`'s epilogue, change line 726 to `factor = sum_weights * weighted_std**3` (line 731's `factor = factor * weighted_std` then correctly yields `sum_weights * weighted_std**4`). This makes the weighted moments consistent with `weighted_std` (line 702: `np.sqrt(weighted_std / sum_weights)`) and `weighted_mad` (line 722: `weighted_mad / sum_weights`), which already normalise by `sum_weights` in the same function.
**Evidence:** Accumulators at line 642 (`weighted_skew += w_d**3 * next_weight`) and line 668 (`weighted_kurt += w_d**4 * next_weight`) are weighted sums, so the correct normaliser is `sum_weights`. Lines 702 and 722 use `sum_weights` for the sibling weighted stats; only 726/731 use `size`. `get_moments_slope_mi_feature_names` (numerical.py:666) confirms `wskew`/`wkurt` are public emitted features. Reproduced by a direct call to `compute_moments_slope_mi`.

**Disposition:** RESOLVED. `factor = sum_weights * weighted_std**3`. `tests/feature_engineering/test_weighted_moments_normalise_by_weight.py`.

### FEATURE_ENGINEERING-2 [P1] silent-correctness
**File:** src/mlframe/feature_engineering/binned_unique_count.py :107-109 (root cause: :69 vs :70)
**Summary:** `entities` comes from `pd.unique` (which keeps NaN as an element) while the codes come from `pd.factorize(..., use_na_sentinel=True)` (which maps NaN to -1), so every entity after the first null in first-seen order is paired with the wrong count.
**Failure scenario:** `binned_unique_count` on a frame with entity column `[A, NaN, B, B, A, B]` and values `[1, 2, 3, 50, 90, 99]`, `n_bins=4`. Reproduced live, output rows: `A -> 2`, `NaN -> 3`, `B -> 0`. Ground truth is `A -> 2`, `B -> 3`, `NaN -> 1`. Entity B, which genuinely visited 3 distinct bins, is reported as having visited zero, and B's count is attributed to the null-id row. No exception, no warning.
**Suggested fix:** Derive the output entity labels from the factorize uniques rather than `pd.unique`: `entity_codes, entity_labels = pd.factorize(df[entity_col], sort=False)` and return `pd.DataFrame({entity_col: entity_labels, ...})` (length `len(entity_labels)`, which excludes the NaN sentinel). If a row for the null entity is wanted, append it explicitly with its own count instead of relying on positional alignment.
**Evidence:** Verified that `pd.unique` on `[A, NaN, B]` returns 3 elements including NaN while `pd.factorize(..., sort=False)[0]` returns `[0, -1, 1]`. Line 103 correctly excludes `entity_codes == -1` from `combined_key` (per the comment at lines 99-102 that fixed an earlier `np.bincount` crash), but the resulting `counts` array is then zipped against the unfiltered `entities` list at line 109. Live repro above.

**Disposition:** RESOLVED. Labels come from `pd.factorize`, so codes and labels agree on NaN. `tests/feature_engineering/test_binned_unique_count_entity_alignment.py`.

### FEATURE_ENGINEERING-3 [P1] leakage / contract-drift
**File:** src/mlframe/feature_engineering/two_step_target_encode.py :85 (docstring claim at :50)
**Summary:** The docstring promises `order` "defaults to `time_col`" for step 1's leak-free encoding, but the code forwards `order=None` straight to `ordered_target_encode`, which then falls back to input row order.
**Failure scenario:** An events frame stored in insertion / entity-major order (the common case for a transactions table: all of card A's events, then all of card B's) with `order=None`. Step 1's causal expanding mean is then computed over row position, not time -- an event's encoding is built from rows that are chronologically in its FUTURE. The function's own step 2 does the right thing at line 105 (`causal_order = order if order is not None else time_vals`), so the two steps silently disagree about what causal means, and the module docstring's "leak-free target-mean-encode" claim (line 5) does not hold for the default call.
**Suggested fix:** Resolve the order once, before step 1: hoist `time_vals = events_df[time_col].to_numpy(dtype=np.float64)` above line 85, set `effective_order = order if order is not None else time_vals`, and pass `effective_order` to both `ordered_target_encode` (line 85) and the step-2 sort (line 105).
**Evidence:** `ordered_target_encode`'s own docstring (training/feature_handling/ordered_target_encoder.py, lines 39-40): "Defaults to the input row order (np.arange(n)) when None." Line 85 passes `order=order` with no defaulting; line 105 in the same function does default to `time_vals`.

**Disposition:** RESOLVED. The causal order is resolved once and passed to both steps. `tests/feature_engineering/test_fe_causal_order_and_task_matched_baseline.py`.

### FEATURE_ENGINEERING-4 [P1] silent-correctness
**File:** src/mlframe/feature_engineering/multi_window_aggregate.py :182-184
**Summary:** The no-feature baseline in the opt-in horizon selector is hard-coded to `DummyClassifier(strategy="prior")` even on the documented regression path, producing a large negative baseline score that makes the first candidate horizon pass the `min_lift` gate unconditionally.
**Failure scenario:** `multi_window_aggregate(..., auto_select=True, target=<continuous>, scoring="r2", estimator=Ridge())`. Reproduced the baseline directly: `cross_val_score(DummyClassifier(strategy='prior'), zeros, continuous_y, cv=5, scoring='r2')` returns `[-11.1, -1.19, -15.2, -9.07, -6.36]`, mean about -8.6 -- it does not raise. `baseline_score` is then -8.6, so for the first horizon `lift = candidate_score - (-8.6)` is about +8.6, far above `min_lift=0.005`, no matter how uninformative that horizon is. It is kept, and the `lifts` dict returned via `return_selection_info=True` reports a meaningless +8.6 lift for it.
**Suggested fix:** Pick the dummy by task, mirroring the estimator: use `DummyRegressor(strategy="mean")` when `sklearn.base.is_classifier(model)` is False, keeping `DummyClassifier(strategy="prior")` for the classification path.
**Evidence:** Line 176 lets the caller pass any `estimator`; the docstring at line 57 explicitly names "the downstream label/regression target"; line 184 hard-codes `DummyClassifier` regardless. Baseline value reproduced above.

**Disposition:** RESOLVED. The dummy is chosen by `is_classifier(model)`, mirroring the estimator. Same test file.

### FEATURE_ENGINEERING-5 [P2] leakage / tie-handling
**File:** src/mlframe/feature_engineering/graph_construction.py :119-122
**Summary:** `shared_attribute_edges(..., timestamp=...)` documents "links only to EARLIER same-group rows (directed past graph) -> leakage-safe" but selects partners by POSITION in a stable time-sort, so rows sharing an identical timestamp are linked as if one were in the other's past.
**Failure scenario:** Day-granularity timestamps (the common case for a transactions / affiliation table). Group G has rows at times `[5, 5, 5]`. After `np.argsort(t[members], kind="stable")` they keep original order; `pos=2` gets `partners = mem[0:2]` -- two contemporaneous rows. Feeding the resulting edges to `graph_features.graph_neighbor_aggregate(values=y)` then aggregates same-day labels into a feature documented as past-only, inflating held-out performance in exactly the way the module's leakage-safety wording promises it does not.
**Suggested fix:** Filter partners on the timestamp VALUE, not the position -- compute `cut = np.searchsorted(t[mem], t[i], side="left")` and use `partners = mem[max(0, cut - max_neighbours) : cut]`. This matches the sibling `knn_graph_edges` in the same file, which already uses the strict value comparison `past = t[cols] < t[rows]` (line 67).
**Evidence:** Lines 120-122 slice `mem[lo:pos]` with `lo` derived from `pos`, not from a timestamp comparison; the comment on line 121 asserts "strictly-earlier". Line 67 of the same file shows the correct strict-value form.

**Disposition:** RESOLVED. Partners are now selected by timestamp VALUE (`np.searchsorted(t_mem, t_mem[pos], side="left")`), matching the sibling `knn_graph_edges`. Measured pre-fix: a group at times `[5, 5, 5]` produced edges `[(1,0), (2,0), (2,1)]` -- three links among rows that are simultaneous -- and `[1, 5, 5]` produced `(2,1)` on top of the two legitimate edges. `tests/feature_engineering/test_past_graph_and_band_ratio_scale.py`.

### FEATURE_ENGINEERING-6 [P2] silent-correctness (additive epsilon in a small denominator)
**File:** src/mlframe/feature_engineering/spectral.py :221
**Summary:** `rolling_hf_lf_ratio` computes `e_hi / (e_lo + 1e-6)` on raw squared-FFT-magnitude band energies, so the epsilon dominates the true denominator for any input whose amplitude is small in absolute units, making the feature a function of the input's SCALE rather than its spectral balance.
**Failure scenario:** Log-returns or any small-unit series with amplitude around 1e-3. Windowed squared-magnitude band energies land around 1e-8 to 1e-7, so `e_lo + 1e-6` is about 1e-6 regardless of `e_lo`, and the emitted ratio collapses to `e_hi * 1e6` -- near-zero for every row, then flattened further by `np.clip(..., 0.0, 100.0)` at line 223. The identical signal multiplied by 1000 (basis points instead of fractions) produces a completely different feature. No NaN, no warning, and the value stays inside `clip_range`, so nothing downstream flags it.
**Suggested fix:** Guard multiplicatively rather than additively: `ratio = np.where(e_lo > 0, e_hi / e_lo, fill_value)`. The existing `np.isfinite` fallback on line 222 then handles the genuine `e_lo == 0` case, and the `fill_value=1.0` default ("balanced") is already the right sentinel. If a floor is still wanted, make it relative (`e_lo + 1e-12 * total_energy`), not absolute.
**Evidence:** `_spec_pow` (lines 226-235) returns `np.abs(np.fft.rfft(wins, axis=1)) ** 2` -- unnormalised squared magnitudes whose units are the square of the input's units. Line 221 adds a fixed 1e-6 to that. The docstring at line 208 documents the formula but not the scale dependence.

**Disposition:** RESOLVED. `np.where(e_lo > 0, e_hi / e_lo, nan)`, with the existing `isfinite` pass supplying `fill_value` for a genuinely empty low band; the stale docstring formula was corrected too. Measured pre-fix relative error on a fixed mixed-tone fixture as the input is rescaled: 0.996 at amplitude 1e-6 (4.59e-6 reported against a true 1.09e-3, 237x too small), 0.703 at 1e-5, 0.023 at 1e-4, 2.4e-4 at 1e-3, negligible from 1e-2 up. The reachable regime is narrower than the finding describes -- at a 128-point window the epsilon only bites below ~1e-4 amplitude, not at 1e-3 -- but within it the feature is a function of scale, not shape. Same test file.

### FEATURE_ENGINEERING-7 [P2] contract-drift
**File:** src/mlframe/feature_engineering/drift_remediation.py :151 (docstring claim at :69-70)
**Summary:** The auto-tune drop-threshold search is documented as "a held-out re-check via `adversarial_auc`", but it re-scores on the same `train_df`/`test_df` frames that produced the importances it is thresholding, so the selected threshold is chosen in-sample.
**Failure scenario:** `remediate_drifting_features(train_df, test_df, auto_tune_drop_threshold=True)`. The importances at line 99 come from `adversarial_auc(train_df[scan_cols], test_df[scan_cols])`; line 151 calls `adversarial_auc(cand_train[cand_cols], cand_test[cand_cols])` on the remediated versions of those same rows. The threshold that minimises this AUC is the one that best de-drifts these rows; on a fresh train/test pair it may drop too many or too few columns, and the reported post-remediation adversarial AUC is optimistic. Nothing in the return value signals this.
**Suggested fix:** Either split the scan rows once (a holdout slice used only for the re-check, unseen by the line-99 importance fit), or correct the docstring at lines 69-70 to say "an in-sample re-check on the same frames" and note the optimism.
**Evidence:** Line 99 and line 151 pass frames derived from the same `train_df`/`test_df` objects; there is no split, subsample, or fold index anywhere between them.

**Disposition:** RESOLVED. New `auto_tune_holdout` (default 0.25) and `auto_tune_seed`: the search fits its OWN importances on the fit rows and scores every candidate on the held-out ones, so neither the flags nor the winning threshold come from the rows they are judged on. The returned remediation still uses the full-data importances, so the non-auto-tune path is unchanged. Same test file.

### FEATURE_ENGINEERING-8 [P2] perf (mechanism: eager whole-frame copy inside a candidate loop)
**File:** src/mlframe/feature_engineering/drift_remediation.py :110-111, called from :149
**Summary:** `_build` unconditionally does `train_df.copy()` plus `test_df.copy()`, and the auto-tune path calls it once per candidate threshold plus once more for the winner -- 12 full-frame materialisations at the default 5 candidates.
**Failure scenario:** `auto_tune_drop_threshold=True` on a train/test pair of any size (this repo's CLAUDE.md notes frames can be 100+ GB). Candidates default to 5 (line 141), so the loop at lines 148-155 produces 5 x 2 = 10 whole-frame copies, plus 2 from the final `_build(best_candidate)` at line 157 -- none of which are needed to SCORE a candidate, since only the flagged columns differ between candidates. On top of the copies, `per_group_rank` is recomputed from scratch for every rank-transformed column on every candidate (lines 127-128).
**Suggested fix:** Hoist the per-column rank computation out of `_build` -- compute `per_group_rank` once per flagged column into a dict before the candidate loop -- and have the candidate loop score against a column-subset view (`train_df[cand_cols]` with the ranked columns swapped in) rather than a copied full frame. Reserve the `.copy()` for the single final `_build(best_candidate)` that actually has to return frames.
**Evidence:** Line 149 `cand_train, cand_test, _ = _build(c)` sits inside `for c in candidates`; `_build` begins with two unconditional `.copy()` calls at lines 110-111. Lines 150-151 then immediately narrow to `cand_train[cand_cols]`, so the copied non-scan columns are never read.

**Disposition:** RESOLVED with FEATURE_ENGINEERING-7. The candidate loop no longer calls `_build`: it assembles only the scanned columns of the holdout rows, column-wise from numpy (never a whole-frame copy), and rank-transforms each flagged column ONCE rather than per candidate. Copies drop from 12 whole frames to the 2 the final `_build` makes; a spy on `pd.DataFrame.copy` pins that. Same test file.

### FEATURE_ENGINEERING-9 [P2] perf (mechanism: whole-array Python-loop recompute to repair a few rows)
**File:** src/mlframe/feature_engineering/multi_window_aggregate.py :229-231
**Summary:** `_cancellation_safe_diff` recomputes the aggregate for EVERY query row via `_direct_window_agg`'s nested Python loop as soon as a single row trips the cancellation heuristic, and does so independently for `sum` and `count` on the `mean` path.
**Failure scenario:** One entity with a long history and a near-empty trailing window makes `risky.any()` True; `_direct_window_agg` then runs its per-entity / per-query double loop (lines 241-253) over all n query rows, calling `getattr(pd.Series(slice), fn)()` per row -- a per-row pandas Series construction. For a `mean` request this happens twice per horizon (lines 119-126), so an H-horizon call can pay 2*H full O(n_queries) Python passes to fix a handful of rows.
**Suggested fix:** Pass the `risky` mask into `_direct_window_agg` and have it iterate only the flagged (entity, query) pairs (`query.loc[risky]` grouped by entity), leaving the fast subtraction for the rest -- which is what the docstring at lines 222-224 already claims ("Rows where the difference is tiny ... are flagged and recomputed directly"). Also replace the per-row `pd.Series(...)` reduction with the corresponding numpy reduction on the slice.
**Evidence:** Line 230 calls `_direct_window_agg(history_df, ...)` with no reference to `risky`; line 231 then discards all but the risky positions. `_direct_window_agg`'s signature (lines 235-237) has no mask parameter.

**Disposition:** RESOLVED. `_direct_window_agg` takes a `rows` boolean mask and walks only the flagged (entity, query) pairs; the per-row `pd.Series(...)` reduction is replaced by a numpy one from a small dispatch table, with the pandas form kept as the fallback for any aggregator not in it. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-10 [P2] silent-correctness (additive epsilon in a small denominator)
**File:** src/mlframe/feature_engineering/anchor.py :206 (identical form at :93, :554, :697)
**Summary:** The EWM anchor slope divides by `den + 1e-12` where `den = Suu - Su*Su/S0` is an exponentially-DECAYED weighted variance, so a short half-life relative to the anchor spacing drives `den` below the epsilon and silently zeroes the slope.
**Failure scenario:** `anchor_ewm_features` with `half_life` small relative to the gap between anchors -- e.g. `half_life=2` rows with anchors 100 rows apart. By the time the second anchor arrives, the first's weight has decayed by `0.5**50`, about 9e-16, so `den` (about 9e-16 times u squared) can sit at or below 1e-12. The reported `ewm_slope` is then a fraction of the true weighted-OLS slope, or about 0, with no NaN and no warning -- it reads as "the process is flat", the opposite of a strong recent move.
**Suggested fix:** Replace the additive pad with a relative guard: `ewm_slope_out[i] = num / den if den > 1e-12 * max(Suu, 1.0) else np.nan` (or leave the output at its NaN initial value). The `n_anch >= 2` check at line 203 already covers the "no data" case; the epsilon is only guarding a numerically small denominator, where a wrong finite number is worse than NaN.
**Evidence:** Lines 184-206 show `den` is built from `Suu`/`Su`/`S0`, all multiplied by `r = 0.5**(1/half_life)` on every row (line 189), so all three shrink geometrically. Line 206 adds a fixed 1e-12 to that decayed quantity. Same pattern at lines 93, 554, 697.

**Disposition:** RESOLVED. Both sites now guard RELATIVELY (`den > 1e-12 * Suu`, and the weighted-scale equivalent in the numpy twin) and leave the NaN-initialised output slot alone rather than emitting a damped number. Measured on a perfectly linear anchor sequence (spacing 100, true slope 1.0): pre-fix `half_life=2` reported exactly 0.0, i.e. a flat process, the opposite of the truth; `half_life=5` and above were already correct and are unchanged. The two unweighted OLS sites (:93, :554) were left on their absolute branch -- their `den` is a sum of squared integer position offsets, so it is either 0 or >= 0.5 and cannot decay. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-11 [P3] silent-correctness
**File:** src/mlframe/feature_engineering/spatial.py :248-258
**Summary:** The non-group-filtered self-match removal drops the first distance-0 neighbour unconditionally, so when `query is not ref` but the reference pool contains a coincident point, a genuine neighbour is discarded and the aggregate silently uses a farther k-th neighbour.
**Failure scenario:** Geocoded rows where two different reference entities share an address (a very common real-world condition). A query at that address has a distance-0 reference hit that is NOT itself; `first_zero` flags it, it is compacted out, and the k-neighbour ring shifts outward by one -- the aggregate is computed over a different, farther set than the k nearest. The docstring at lines 139-141 only promises to skip "the self-match if the same point appears in both pools".
**Suggested fix:** Only compact when a genuine self-match is possible -- accept an explicit `query_is_ref: bool` parameter (or test `query_coords is ref_coords`) and skip the compaction otherwise; or, when row identity is available, mask on `indices == query_row_index` rather than on `distances <= 0.0`.
**Evidence:** Line 248 `is_zero_dist = distances <= 0.0` keys on distance only; there is no comparison of `indices` against the query's own row index -- unlike `cross_sectional_neighbors.py:129`, which uses `neighbor_idx == np.arange(n).reshape(-1, 1)`.

**Disposition:** RESOLVED. New `query_is_ref` parameter, inferred from `query_coords is ref_coords` when not given; the rank-0 removal is skipped when the pools differ. Measured: a query coincident with reference 0 (label 100) and k=2 returned 1.5 pre-fix (the coincident reference dropped, the ring shifted outward) against 50.5 post-fix. `knn_aggregate` has no in-repo callers, so nothing internal depended on the old behaviour. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-12 [P3] diagnosability
**File:** src/mlframe/feature_engineering/spatial.py :290 (and :296)
**Summary:** `_nearest_distance` is documented as "distance to k=1 neighbour after group filtering", but in the group-filtered branch a fully-starved query row (zero valid different-group neighbours) still gets `compact_dist[:, 0]` -- a same-group neighbour's distance -- while its aggregates are correctly NaN.
**Failure scenario:** A panel row whose own group exhausts the entire `q_k = min(n_ref, k*4+1)` candidate window (the case the warning at lines 227-235 already detects and counts). Its median / mean / std come back NaN, correctly signalling "no usable neighbours", but `_nearest_distance` comes back as a finite same-group distance -- a within-group proximity value leaking into a column meant to be group-filtered.
**Suggested fix:** Mask it consistently with the aggregates: `out_aggs["_nearest_distance"] = np.where(compact_mask[:, 0], compact_dist[:, 0], np.nan)` in the group-filtered branch.
**Evidence:** `compact_mask` (line 218) already records per-slot validity and is used to NaN out `labels_arr` at lines 238-240; line 290 reads `compact_dist[:, 0]` without consulting it. The `starved` computation at lines 224-225 proves the all-invalid case is reachable and known.

**Disposition:** RESOLVED. `np.where(compact_mask[:, 0], compact_dist[:, 0], np.nan)`, as suggested. The regression test asserts separately that its fixture genuinely starves the row (six same-group references fill the whole `q_k = min(n_ref, 5)` window before the group-2 one), so the consistency assertion cannot pass on two finite values. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-13 [P3] contract-drift / tie-handling
**File:** src/mlframe/feature_engineering/fuzzy_entity.py :130-133
**Summary:** `value_occurrence_count_in_group` and `days_since_value_last_seen_in_group` are documented as computed "STRICTLY BEFORE this row", but a stable sort on `order` plus `groupby().cumcount()` counts equal-`order` rows that happen to sort earlier.
**Failure scenario:** `time_order` is a date (day granularity). Two rows in the same group with the same value on the same day: the second gets `value_occurrence_count_in_group = 1` (it "has seen" a contemporaneous row) and `days_since_value_last_seen_in_group = 0.0` rather than the NaN a genuine first-observation-at-this-time would get. For an online-scoring novelty signal that is a same-timestamp leak.
**Suggested fix:** After sorting, replace `cumcount()`/`diff()` with a comparison on the `order` VALUE -- within each (group, value) block compute `np.searchsorted(block_order, block_order, side="left")` for the strictly-before count, and take the gap against the last strictly-smaller order value.
**Evidence:** Line 130 `df.sort_values("order", kind="stable")` then line 132 `grp.cumcount()` -- position-based, not value-based. Docstring line 84: "strictly from PRIOR rows only (leak-safe, causal)"; line 105: "STRICTLY BEFORE this row".

**Disposition:** RESOLVED, vectorised rather than per-group. Sorting by (block, order) makes each (group, value) block contiguous and ascending, so a running maximum over run starts gives both the strictly-before count and the last strictly-earlier position with no Python callback -- this module deliberately avoids those (the mode aggregation above it was profiled at 54s/call through one). Measured pre-fix on three rows tied at order 5: counts `[0, 1, 2]` and gaps `[nan, 0.0, 0.0]`, against `[0, 0, 0]` and all-NaN post-fix. The answer also no longer depends on the incoming row order for tied timestamps. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-14 [P3] silent-correctness (missing validation)
**File:** src/mlframe/feature_engineering/graph_features.py :62
**Summary:** `_build_csr` validates `src.min() < 0` but never `dst.min() < 0`, so on the `directed=True` path a negative destination index passes validation and then wraps around under numpy indexing, producing a wrong neighbour aggregate instead of raising.
**Failure scenario:** `graph_neighbor_aggregate(n, edges, values, directed=True)` where an edge row is `[3, -1]` (e.g. a `pd.factorize` sentinel not filtered upstream). The endpoint check passes; `indices` then holds -1, and `_sum_impl` / `_wmean_impl` read `values[-1]` -- the LAST node's value -- silently mixing an unrelated node into node 3's neighbour aggregate. On the `directed=False` path the bug is masked because `src` and `dst` are concatenated symmetrically at line 59, so `src.min()` covers both.
**Suggested fix:** Extend the guard to also test `dst.min() < 0` alongside the three bounds already checked on line 62.
**Evidence:** Line 62 checks exactly three of the four bounds; the `keep = src != dst` self-loop filter at line 60 does not remove negative indices.

**Disposition:** RESOLVED. `dst.min() < 0` added to the guard. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-15 [P3] silent-correctness (additive epsilon in a small denominator)
**File:** src/mlframe/feature_engineering/windowed_shape.py :323 (numpy twin at :467)
**Summary:** The normalised total-variation feature divides by `(wmax - wmin) + 1e-12`, so a window whose genuine range is at or below 1e-12 gets a scale-dependent value instead of the true wiggle ratio.
**Failure scenario:** A large-offset near-constant series (a price column around 1e5 stored in float32, or a sensor pinned at a setpoint), where a window's true range is about 1e-11 and `tv` is about 1e-11. The correct normalised TV is about 1.0 ("maximally wiggly relative to its own range"); the emitted value is `1e-11 / (1e-11 + 1e-12)`, about 0.91, and for a range of 1e-13 it drops to about 0.09. The same signal shifted or rescaled produces a different feature.
**Suggested fix:** Branch explicitly on a zero range instead of padding: `out[r] = tv / (wmax - wmin) if wmax > wmin else 0.0` -- a constant window has `tv == 0` anyway, so 0.0 is the right degenerate value and there is no division at all.
**Evidence:** Lines 322-323 in the njit kernel and line 467 in the numpy fallback both use `(wmax - wmin) + 1e-12`; the docstring at line 297 pins the two forms to each other, so both must change together.

**Disposition:** RESOLVED. Both the njit kernel and the numpy twin branch on `wmax > wmin` and emit 0.0 for a constant window. Measured on a fixed zig-zag whose true normalised TV is 19: pre-fix 17.27 at scale 1e-11 and 1.73 at 1e-13, i.e. down 91%, purely from rescaling the same shape. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-16 [P3] contract-drift (output dtype)
**File:** src/mlframe/feature_engineering/nearest_past_join.py :140-141
**Summary:** In the `fallback_by_chain` path the attached columns are initialised with `pd.NA`, giving them object dtype; the documented "NaN where no eligible past row exists" contract and the numeric dtype of the single-tier path are both lost.
**Failure scenario:** `nearest_past_join(..., fallback_by_chain=[["region"], None])` on a numeric value column. Every attached column comes back as object dtype holding a mix of `pd.NA` and Python floats. A downstream `.to_numpy(dtype=np.float64)`, a polars conversion, or a model fit then either raises or silently coerces; `np.isnan` fails on `pd.NA`. The single-tier path (lines 51-56, `pd.merge_asof`) returns proper float columns for the same inputs, so behaviour changes purely by enabling the fallback chain.
**Suggested fix:** Initialise with `np.nan` and a numeric dtype -- `out[new_name] = np.full(len(out), np.nan, dtype=np.float64)`, or seed the dtype from `right_df[col].dtype` -- and keep the `.notna()` resolution logic unchanged.
**Evidence:** Lines 140-141 assign `pd.NA` into freshly created columns; the docstring at lines 111-112 promises NaN where no eligible past row exists. No cast back to a numeric dtype occurs before the return at line 175.

**Disposition:** RESOLVED. The attached columns are seeded with `np.nan` at the source column's float dtype (or float64 when it is not a float dtype), so `.to_numpy(np.float64)` and `np.isnan` work and the chained path matches the single-tier one. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

### FEATURE_ENGINEERING-17 [P3] diagnosability
**File:** src/mlframe/feature_engineering/financial.py :355-359
**Summary:** OHLCV columns are `fill_null(0.0)` before every TA indicator, silently turning a missing price into a zero price, with no null-count check and no warning.
**Failure scenario:** A ticker with a sporadic missing `low` on one bar. That bar's `low` becomes 0.0, so range-based indicators (ATR, Stochastic, Williams %R, and the `high / low.shift(lag) - 1` ratios built by `add_ohlcv_ratios_rlags` at lines 139-141) see a bar with a 100%-of-price range or divide by zero to inf. The comment at lines 349-354 explicitly documents that the caller must forward-fill upstream, but nothing checks whether they did, so the feature is silently corrupted for exactly the rows a user would most want flagged.
**Suggested fix:** Before building the expressions, count nulls in the five OHLCV columns via `null_count()` and emit a `logger.warning` naming the columns and counts when any are non-zero -- the same "make the fallback diagnosable" treatment already applied at `numerical.py:134` and `numerical.py:429` in this cluster.
**Evidence:** Lines 355-359 apply `.fill_null(0.0)` unconditionally; the comment at lines 349-354 acknowledges the caller-side requirement but no code enforces or reports it.

**Disposition:** RESOLVED, diagnosability only -- the zero-fill itself is forced by the polars limitation documented at the fill site, so the behaviour is unchanged. Null counts are taken BEFORE the fill and a warning names each affected column and its count, so the caller who did not forward-fill upstream finds out. `tests/feature_engineering/test_fe_p2_p3_epsilons_bounds_and_ties.py`.

## Coverage
Read in depth (32 files, 9,867 LOC):
- src/mlframe/feature_engineering/_numerical_stable.py (full)
- src/mlframe/feature_engineering/_numerical_numba.py (lines 458-790, the moments/slope kernel factory)
- src/mlframe/feature_engineering/numerical.py (lines 490-745, compute_numaggs and the feature-name contracts)
- src/mlframe/feature_engineering/_grouped_segments.py (full)
- src/mlframe/feature_engineering/grouped.py (lines 60-240)
- src/mlframe/feature_engineering/anchor.py (lines 100-220)
- src/mlframe/feature_engineering/spatial.py (lines 113-305, plus full function index)
- src/mlframe/feature_engineering/cross_sectional_neighbors.py (full)
- src/mlframe/feature_engineering/two_step_target_encode.py (full)
- src/mlframe/feature_engineering/holiday_locale_target_encoding.py (full)
- src/mlframe/feature_engineering/rolling_target_correlation.py (full)
- src/mlframe/feature_engineering/as_of_aggregate.py (full)
- src/mlframe/feature_engineering/multi_window_aggregate.py (lines 30-258)
- src/mlframe/feature_engineering/nearest_past_join.py (full)
- src/mlframe/feature_engineering/event_proximity_decay.py (full)
- src/mlframe/feature_engineering/binned_unique_count.py (lines 50-113)
- src/mlframe/feature_engineering/fuzzy_entity.py (lines 75-149)
- src/mlframe/feature_engineering/graph_features.py (lines 1-90, 180-290)
- src/mlframe/feature_engineering/graph_construction.py (lines 56-135)
- src/mlframe/feature_engineering/drift_remediation.py (lines 60-158)
- src/mlframe/feature_engineering/spectral.py (lines 195-240)
- src/mlframe/feature_engineering/windowed_shape.py (lines 290-335, 460-470)
- src/mlframe/feature_engineering/recency_aggregation.py (lines 160-290)
- src/mlframe/feature_engineering/recency_density.py (lines 1-60)
- src/mlframe/feature_engineering/financial.py (lines 336-395)
- src/mlframe/feature_engineering/auxiliary_feature_prediction.py (lines 55-135)
- src/mlframe/feature_engineering/transformer/neighbor_aggregate_features.py (full)
- src/mlframe/feature_engineering/transformer/_aggregation.py (lines 200-290)
- src/mlframe/feature_engineering/transformer/local_lift.py (lines 1-80)
- src/mlframe/feature_engineering/transformer/target_quantile.py (lines 1-80)
- src/mlframe/feature_engineering/transformer/y_quintile_baseline_knn.py (lines 1-80)
- src/mlframe/feature_engineering/transformer/trust_score_oof.py (lines 1-80)

Grep-scanned across all 314 .py files in the cluster for: raw-power-sum moment expansions (`s3/n`, `- 3*mean*`, `mean**3`, `raw_moment`, `_moment_sums`, `m3 =`, `m4 =`); additive epsilons (`+ 1e-`, `+ eps`); broad `except Exception` with the following 3 lines, to separate logged from silent; `pl.Categorical` and `min() == max()` polars traps; self-match and fold-discipline markers (`self`, `k+1`, `oof`, `fold`, `leak`, `exclude_self`); strictly-past / leak-safe docstring claims; `fit_transform` and `.fit(`; whole-frame `.copy()` / `.clone()` / `.to_pandas()` / `pl.from_pandas`; rolling / shift / `closed=` window boundaries; and `@njit` / `prange` / `cuda.jit` / `cupy` / `kernel_tuning_cache` before each perf finding.

Live repro used to confirm FEATURE_ENGINEERING-1, -2 and -4: direct calls to `compute_moments_slope_mi`, `binned_unique_count`, and `cross_val_score(DummyClassifier, ...)` on a continuous target.
