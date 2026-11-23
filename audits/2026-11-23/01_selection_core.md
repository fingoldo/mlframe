# Audit 01: MRMR selection core (wave 1)

Scope: greedy selection, gain and threshold logic, support assignment, index mapping between cols / original / engineered indices, silent fallbacks, mutable state shared between fits, pickling of fit attributes. Read-only, Haiku model, nothing executed.
Coverage gap: the greedy core (`_screen_predictors*.py`), `_numba_utils*.py`, `evaluation.py`, `_finalise.py` were not read in depth.

Tracker rows: AUD23-37 (confirmed), AUD23-03 (pickle), AUD23-01 (fingerprint; cross-listed from state audit).

## Confirmed by reading the code path (not executed)

1. Engineered-operand re-attach compares the wrong index space. Severity medium-high.
   - `_mrmr_fit_impl/_assign_support.py:746` and `:756-757` (`_collect_engineered_operands`), called from `_readd_redundancy_dropped_engineered` at `:773`.
   - `_post_selection.py:263` sets `selected_vars = original_indices` (raw `feature_names_in_` indices only). Engineered names are removed at `:240-250`.
   - `_collect_engineered_operands` tests `_eb_cols_idx[_enm] not in _eb_sel_set`, a cols-space index tested against a raw-space set. Engineered names are never in that set, so `continue` at `:746-747` fires for every recipe.
   - Scenario: `redundancy_policy="drop"`, selected recipe `div(a,b)` with `a`, `b` dropped as redundant. The re-attach never happens. A recipe passes only by accidental collision with an unrelated raw index, which admits operands spuriously.

2. Operand re-attach failure swallowed at DEBUG. Severity low-medium.
   - `_assign_support.py:786-787`: `except Exception as _eb_exc: logger.debug(...)`. The final support silently omits the operands. Inconsistent with the WARNING used at `:540-541`.

## Hypotheses (not confirmed)

3. Cache-tuple key for `cached_MIs` may miss. Low. `_assign_support.py:318-326`, `:463-473` look up `(ci,)`; not verified that the screen writes that exact key form. If not, relevance lookups return 0.0 and tiebreaks degrade to index order.

4. Fit-cache key coverage. Low. `_fit_cache.py:100-121` keys on `get_params(deep=True)`, X/y hashes, groups. Not checked whether `_hashable_params_signature` drops a selection-changing parameter (e.g. `random_seed`, `factors_to_use`).

5. Pickling. Low. `mrmr/_mrmr_class.py:3325-3350` strips some attributes. Not verified that all fit-time attributes (`_stability_replay_state_`, `_artifacts_`, `_redundancy_emptied_raw_`, `_raw_redundancy_dropped_`, `_fused_subsumed_raws_`) round-trip through `__setstate__`.

## Checked and fine
- `_assign_support.py` index space is consistent from `_post_selection.py:263` onward.
- Never-empty re-attach remaps cols to raw by name (`:614`).
- `_engineered_recipes` and `_predictors_log_` assignments are guarded.
- Only `_FIT_CACHE` is a class-level mutable, intentional and bounded.
