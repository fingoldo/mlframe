# Audit 12: MRMR greedy screen core (gap audit, Haiku, read-only, nothing executed)

Scope: the greedy loop (`_screen_predictors_steps.py`, `_confirm_predictor.py`, `_confirm_predictor_helpers.py`, `_confirm_predictor_engineered.py`), and `_screen_predictors.py`, `_screen_predictors_prescreen.py`, `_screen_predictors_gate.py` read in full. The `_screen_round*.py` files hold DCD attach, budget and relax-retry helpers, not the greedy step. No screen module imports `_numba_utils*` directly, so that part has no screen-side call sites.

Tracker rows: AUD23-51 to AUD23-54 (added below).

## Finding 1: timeout accepts an unconfirmed candidate. High. Confirmed by code reading.
- Trigger: `max_runtime_mins` expires inside `score_candidates`, in the last round before the budget runs out.
- `_evaluation_candidate.py:126-127` sets `run_out_of_time`. `_evaluation_driver.py:491-492` breaks the scoring loop with the best-so-far `best_candidate` and `best_gain`, which are raw, unconfirmed scores.
- `_confirm_predictor.py:370-371` returns on `run_out_of_time` without confirming, returning `best_candidate, best_gain` to the caller.
- `_screen_predictors_steps.py:135` writes `st.run_out_of_time` from that return. There is no `if st.run_out_of_time: break` between line 135 and the gate at line 146.
- The gate at 146-152 runs on the unconfirmed candidate. If it passes, lines 153-194 append it to `selected_vars` and `predictors`: the screen keeps a feature that never passed the permutation test.
- Fix to consider: skip the gate and the append when `st.run_out_of_time` is set. Reproduce with a tiny `max_runtime_mins` first.

## Finding 2: `patience_triggered` is sticky across interaction orders. Low. Confirmed.
- `_screen_predictors_steps.py:81-83` takes `patience_triggered` as input and returns it; it is not reset at line 86 when a new order starts.
- `interactions_order=1` trips patience, `interactions_order=2` finishes naturally: `_log_screen_termination` (`_screen_predictors.py:275`) still reports "terminated early via max_consec_unconfirmed".
- The consecutive counter is reset per order (`_screen_predictors_steps.py:109`), so only the reported reason is affected, not the selection.

## Finding 3: abandoned prefetch futures can read mutated state. Medium. Hypothesis.
- `_confirm_predictor_helpers.py:321` creates `_ConfirmPrefetch` per `confirm_in_gain_order` call; futures go to a process-wide `_PREFETCH_POOL` (lines 255-281).
- Every early `break` (347, 392, 411, 420) abandons unconsumed futures, which keep running with the shared `ctx`.
- `_screen_dcd_swap.py:39-42` reassigns `ctx.factors_data`, `ctx.factors_nbins`, `ctx.factors_names`, `ctx.data_copy` after an accepted swap; a still-running abandoned future could evaluate old candidate tuples against the post-swap matrix.
- Not traced whether `confirm_candidate` reads these fields in a way that makes this observable, or whether `cached_confident_MIs` can be written under a stale key.

## Finding 4: child-substitution gain uses a possibly stale partial gain. Low. Hypothesis.
- `_confirm_predictor_engineered.py:361-368` sets the child's gain to `max(expected_gains[idx], partial_gains[idx][0])`. A partial gain was computed against an older `selected_vars`, not a current conditional gain.
- The child is then chosen over its raw parent (`_confirm_predictor.py:414-420`) on that gain.

## Checked and consistent
- Index validity: `_confirmable_engineered_child` returns `(idx, cand, gain)` (`_confirm_predictor_engineered.py:374-375`); the consumer unpacks the same order (`_confirm_predictor.py:463`). `expected_gains` is reallocated per call (`_confirm_predictor.py:336`).
- Stored gain (`_screen_predictors_steps.py:191`) is the confidence-scaled gain (`_confirm_predictor.py:419`), not the MM-corrected gate value discarded at line 146; `compute_selection_gate` re-applies MM to stored gains (`_screen_predictors_gate.py:183-189`), so the floor is consistent.
- Cross-round state (`partial_gains`, `added_candidates`, `failed_candidates`, `nconsec_unconfirmed`) is reset per order (`_screen_predictors_steps.py:106-117`); caches seeded from deterministic keys are reuse by design.
