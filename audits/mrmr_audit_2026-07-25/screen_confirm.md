# SCREEN_CONFIRM cluster — audit (2026-07-25)

Scope: the greedy screening/confirmation loop and its FE redundancy siblings —
`screen.py`, `pre_screen.py`, `_screen_predictors.py`, `_screen_predictors_gate.py`,
`_screen_predictors_prescreen.py`, `_screen_dcd_swap.py`, `_confirm_predictor.py`,
`_confirm_predictor_context.py`, `_confirm_predictor_engineered.py`,
`_per_fold_majority_accept.py`, the raw-vs-engineered redundancy drop
(`_fe_raw_redundancy_drop.py` / `_anchors.py` / `_helpers.py`), the engineered-vs-engineered
CMI gate + null (`_fe_cmi_redundancy_gate.py` / `_fe_cmi_redundancy_null.py`), batched-MI
(`_fe_batched_mi.py` / `_fe_batched_mi_cmi.py`), SIS prescreen (`_mrmr_sis_screen.py` /
`_mrmr_sis_apply.py`), `_null_importance.py`, `_oracle_scorer_select.py`.

## Prior-audit cross-reference (2026-07-22 → current)

Verified FIXED against current source (not re-reported):
- SCREEN_CONFIRM_A-2 (confirm GPU marginal call had no try/except) — FIXED, `_confirm_predictor.py:463-503` now wraps `mi_direct_gpu` and falls back to CPU `mi_direct`.
- SCREEN_CONFIRM_A-4 (Fleuret pool rebuilt per candidate) — FIXED, `_confirm_predictor.py:596-598` builds one pool per round and caches it on `ctx`.
- SCREEN_CONFIRM_B-4 (prefer_gpu ignores MLFRAME_DISABLE_GPU) — FIXED, `_confirm_predictor.py:444-462,526`.
- FE_REDUNDANCY_SYNERGY-2 (anchor loop over full raw_name_set) — FIXED, `_fe_raw_redundancy_anchors.py:292-293` iterates only `_all_relevant_raws`.
- FE_REDUNDANCY_SYNERGY-3 (`_Y_DENSE_MEMO` unlocked) — FIXED, `_fe_cmi_redundancy_gate.py:193,301-314` now guards with `_Y_DENSE_MEMO_LOCK`.
- ORTH_SCORING_B-3 / X_SECURITY_API_PACKAGING-2 (`_ROWS_CACHE` unlocked) — FIXED, `_oracle_scorer_select.py:96,112-124` guards with `_ROWS_CACHE_LOCK`.
- SCREEN_CONFIRM_A-8 (DCD-init failure silent at verbose=0) — FIXED, `_screen_predictors_gate.py:73-84` logs at debug unconditionally.
- SCREEN_CONFIRM_A-15 (`int(y[0]) if hasattr...` triplicated) — FIXED, factored into `_single_int` (`_screen_predictors_gate.py:19-22`).

Still open (restated below): the `_EVALUATE_CANDIDATES_POOL_ENABLED=False` dead parallel-scoring path (prior A-10).

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| SCREEN_CONFIRM-1 | P2 | bug (correctness) | `_mrmr_sis_screen.py:257-261` | The SIS marginal-MI channel quantile-bins ANY target with `> max(nbins,2)` distinct values whenever the dtype is not string/object/bool — so a **high-cardinality integer NOMINAL classification target** (e.g. `product_category` coded 0..99) is treated as ordinal and collapsed into 10 quantile bins, merging genuinely distinct classes and imposing false ordinal structure on the MI ranking. | `y = np.arange(100).repeat(50)` (integer, kind `'i'`, 100 classes): `kind in "fc"` is False but `np.unique(y).size (100) > max(10,2)` is True, so line 258-260 quantile-bins to 10 codes. Classes 0..9 all map to bin 0, etc. — `_mi_classif_batch` then scores against a corrupted 10-class target. String labels avoid this (they factorize at 255-256); only integer/uint nominal targets with >nbins classes are mis-encoded. Gate is pre-rank-only so it cannot zero out a survivor, but a main-effect feature discriminating a merged class pair is under-ranked. |
| SCREEN_CONFIRM-2 | P2 | bug (correctness) | `_fe_raw_redundancy_drop.py:650-655` | The group-aware leak-exemption cannot distinguish `group_blocked_mi` returning `0.0` because a raw has **zero within-group signal** (a true between-group leak) from `0.0` because **no group cleared `min_rows`** (`_group_mi.py:110-111`, `total_w<=0 → return 0.0`). A dropped NON-leak raw on panel data whose groups are each below `_gmr` is then flagged as a leak and PERMANENTLY exempted from the no-harm Ridge revert (`:699-712`), re-introducing the exact held-out-R² harm the guard exists to prevent. | `group_aware_mi=True` with many small groups (each `< min_rows`, default 20). Any raw in `drop_names` gets `_grp_mi==0.0` → added to `_group_leak_names` → stays dropped even when the Ridge guard measures material harm. The nan-return path (segment misalignment, `_group_mi.py:137-138`) is correctly treated as "cannot decide" (`_grp_mi != _grp_mi`), but the all-groups-too-small `0.0` path is not — it is indistinguishable from a genuine zero-signal leak. |
| SCREEN_CONFIRM-3 | P3 | house-convention | `_confirm_predictor.py:461`; `_mrmr_sis_screen.py:110,113`; `_screen_predictors.py:908` | Stale self-referential `"suppressed in <file>.py:<N>"` debug-log strings whose embedded line number no longer matches the emitting line, so a developer grepping production logs for the swallow site lands on the wrong (or a nonexistent) line. | `_confirm_predictor.py:461` emits `"suppressed in _confirm_predictor.py:528"` while sitting at line 461. `_mrmr_sis_screen.py:110` emits `:109`, `:113` emits `:111`. `_screen_predictors.py:908` emits `:955`. (`_fe_cmi_redundancy_null.py:276` happens to still be accurate.) Same class as FE_REDUNDANCY_SYNERGY-9; the enumerated cluster instances above were missed by that sweep. |
| SCREEN_CONFIRM-4 | P3 | house-convention | `_confirm_predictor.py:76,464,521`; `_confirm_predictor_context.py` (X_EFFICIENCY_ARCHITECTURE-1); `_fe_raw_redundancy_anchors.py:285`; `_fe_cmi_redundancy_gate.py:187`; `_screen_predictors_gate.py:21,74`; `_oracle_scorer_select.py:90,138` | Leftover finding-ID audit metadata embedded in production code comments (`SCREEN_CONFIRM_A-2/A-4/A-8/A-15 fix`, `X_EFFICIENCY_ARCHITECTURE-1 fix`, `FE_REDUNDANCY_SYNERGY-2/-3 fix`, `ORTH_SCORING_B-3/B-7 fix`), which CLAUDE.md's comment-style rule forbids (audit IDs belong in git history, not comments). The 2026-07 metadata cleanup missed these cluster files. | Direct read of each cited line shows a finding-ID token in the comment prose. Cosmetic; no behavior change. |
| SCREEN_CONFIRM-5 | P2 | dead code | `_confirm_predictor.py:64,168-285`; `_evaluation_driver.py` (`evaluate_candidates`/`_evaluate_candidates_inner`) | `_EVALUATE_CANDIDATES_POOL_ENABLED = False` is a hardcoded module constant with no env/param override, so the entire ~120-line parallel `score_candidates` branch (`:168-285`) and the two `evaluate_candidates` functions it dispatches to are unreachable in production; still re-exported and imported (`_confirm_predictor.py:40-48`) as if live. Prior SCREEN_CONFIRM_A-10, verified still open. | grep for `_EVALUATE_CANDIDATES_POOL_ENABLED` shows the only reference is the `and` guard at `:169`; nothing sets it True. The branch runs only from standalone unit tests, never an `MRMR.fit(n_workers>1)`. |

## Non-findings / confirmed-clean angles

- **Raw-redundancy partial-revert bookkeeping** (`_fe_raw_redundancy_drop.py:698-712`): verified correct. On a triggered revert `_final_drop_names` keeps only names in `_group_leak_names`, `_final_drop_idx` maps them via a fresh `cols`→idx map, and the return `[i for i in sel if i not in _final_drop_idx], _final_drop_names` restores non-leak raws while leaving demoted leaks dropped. With `group_aware_mi` off, `_group_leak_names` is empty → `_final_drop_names==[]` → full legacy revert, backward-compatible.
- **`drop_redundant_raw_operands` return-tuple callers** — all three (`_fit_impl_core.py:8155` main sweep, `:8705` never-empty subsumption probe, `:8613` post-retention `_post_drop`) unpack `(kept, dropped)` and guard on `if dropped:`, so an empty/partial dropped set (the new leak-only or full-revert cases) is handled without index errors; `self._raw_redundancy_dropped_` is `|=`-merged with the returned names.
- **DPI-trap consumer filter** (`:312`) and **pseudo-remix exclusion** (`_helpers.py:62-68`): the `len(_eng_signal_parents.get(ei,set()) - {rname}) >= 1 and not _is_pseudo_remix_child(...)` gate correctly restricts the conditioning set to genuine multi-source combinations; empty-consumer path keeps the raw.
- **CMI keep/drop legs** (`_fe_cmi_redundancy_gate.py:627-640`): `passes_floor and (passes_rel or strongly_significant)`; the strong-significance escape requires `passes_floor` first and `floor>0.0`, so no divide-by-noise admit path. Seed tie-break (`_tie_key`, `:470-475`) and per-round ordering (`:553`) are both PYTHONHASHSEED-independent.
- **Permutation-confirmation seeding**: both `_marginal_base_seed` (`_confirm_predictor.py:437`) and `_fleuret_base_seed` (`:587`) fold `hash(X)` + `random_seed`; `hash` over `tuple[int]` is process-stable (no PYTHONHASHSEED effect on ints). Serial small-budget confirm keeps the raw exceedance rate (`:641-649`) deliberately.
- **CMI analytic-null df sign** (`_fe_cmi_redundancy_null.py:190-206`): `_df = k_xyz + k_z - k_xz - k_yz` matches the Miller-Madow bias numerator; `_df>0` and sparse-cell gate are both applied before the analytic return.
- **`group_blocked_mi` MM clamp** (`_group_mi.py:105-106`): mi clamped `>=0` per group, so a genuine weak within-group feature never returns spuriously-negative MI — the leak check's `<= 0.0` test is not fooled by MM over-correction (only by the all-groups-too-small `0.0`, see SCREEN_CONFIRM-2).
- **`per_fold_majority_accept`**: shape/empty guards raise cleanly; Wilson bound opt-in; deterministic.
- **`_oracle_scorer_select.py` pickle**: `__getstate__`/`__setstate__` rebuild the oracle from scalar config (store rebuilt on unpickle) — no live handle pickled.
- **mypy**: prior audit reported this cluster clean; the fixes since (locks, partial-revert) are type-preserving. No new implicit-Optional / return-type drift observed in the read files.

## Proposals (perf / refactor / test — not bugs)

1. **SCREEN_CONFIRM-1 fix**: in `_mrmr_sis_screen.py`, branch the MI-channel y-encoding on target *nature*, not just dtype — factorize an integer target that is classification-like (few distinct values relative to n, or when the caller already knows it is classification) and reserve quantile-binning for a genuinely continuous/regression target (`kind in "fc"` or distinct/n above a ratio). Add a test: a 50-class integer target where a feature perfectly separates two classes that quantile-binning would merge, asserting that feature ranks above noise.
2. **SCREEN_CONFIRM-2 fix**: have the leak check treat `group_blocked_mi == 0.0` as a leak ONLY when at least one group cleared `min_rows` (i.e. `total_w>0`). Cheapest route: add a companion `group_blocked_mi(..., return_eval_count=True)` (or a sentinel) so `0.0`-because-unevaluable returns nan like the misalignment path, then reuse the existing `_grp_mi != _grp_mi` "cannot decide → not exempt" branch. Test: `group_aware_mi=True` with all groups `< min_rows` and a genuinely-lossy non-leak raw whose drop harms held-out Ridge — assert the raw is restored (not leak-exempted).
3. **SCREEN_CONFIRM-3/-4 cleanup pass**: one grep-driven sweep to (a) strip the finding-ID tokens from the enumerated comment lines and (b) fix or remove the stale `"suppressed in <file>.py:<N>"` self-references (prefer dropping the embedded line number entirely — it is guaranteed to drift on the next carve).
4. **SCREEN_CONFIRM-5**: either wire `_EVALUATE_CANDIDATES_POOL_ENABLED` to a real env/param (if the parallel-scoring path is still wanted for some workload shape) or delete it plus the two `evaluate_candidates` functions and their re-exports, since the in-file A/B already found it never wins.
</content>
</invoke>
