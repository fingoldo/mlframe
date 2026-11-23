# Audit 04: MRMR state, caches and pickling (wave 2)

Scope: fitted attributes and __getstate__/__setstate__, fit-cache keys, module-level caches, thread-safety of shared caches. Read-only, Haiku model, nothing executed.

Tracker rows: AUD23-01 (high), AUD23-03, AUD23-49, AUD23-50.

## 1. HIGH, confirmed by code reading: stale full-X hash can cause a wrong replayed selection
- `_mrmr_fingerprints.py:602-611` and `:628`: the `_full_x_content_hash` memo key is `(id(X), shape, _content_array_signature(arr))`. The strided signature samples about 10 cells (comment at :604). The weakref check confirms object identity, not content.
- Sequence: `sel.fit(X, y)`; edit an unsampled cell in place (`X.iloc[7, 2] = 99`); `sel.fit(X, y)` again.
- The second call gets the old digest. `_fit_cache.py:47` builds the stale signature, `_same_inputs_skip` (`_fit_cache.py:135`) returns True, and the old `support_` is returned. The same digest feeds `_fit_cache.py:111`, so a clone also replays the old fit from `MRMR._FIT_CACHE`.
- Not executed; the 10-cell count is taken from the comment.

## 2. MEDIUM, confirmed mechanism, trigger hypothesis: stale device operand from an id-keyed memo
- `_fe_resident_operands.py:129-160` memoises the content hash on `id(host)` plus shape and dtype. It does not detect in-place mutation of the same array. `:251` uses that hash as the key into `_FE_RESIDENT_OPERANDS`; if a caller mutates a column in place between two `resident_operand` calls, the old device copy is returned. No such caller found.

## 3. MEDIUM, confirmed for pre-attribute pickles: `signature` missing from the setstate roster
- `_mrmr_class.py:3218` sets `self.signature` only in `__init__`; not a ctor param and not in `_mrmr_setstate_defaults.py`.
- `_fit_cache.py:135` reads `self.signature` bare after `skip` is evaluated. Any pickle without it raises AttributeError on the next `fit()` when the skip flag is on (default).
- The same roster also omits `support_`, `feature_names_in_`, `n_features_in_`; their readers' fallbacks not checked.

## 4. LOW, confirmed inconsistency, no wrong result found: mixed param snapshots in cache keys
- `_fit_cache.py:74-75` and `_mrmr_fingerprints.py:558` use `_pre_fit_ctor_params_snapshot_`; `_fit_cache.py:102` (`_fit_cache_key`) uses live `get_params(deep=True)` read inside the window where transient overrides are active. Store and lookup use the same key, so identical refits hit; two configs collide only when effective values match.
- `_finalise.py:550` and `_mrmr_class_fit_helpers.py:782` store post-fit live params as `self.signature`: the only places where live and snapshot disagree.

## 5. MEDIUM, hypothesis: `sample_weight` absent from the identity-shortcut key
- `_mrmr_fingerprints.py:545-560` (`_mrmr_identity_cache_key`) called as `(self, X, y)` at `_mrmr_class_fit_stages.py:228`. The identity check runs at `_mrmr_class.py:3512`, before `_maybe_resample_for_sample_weight` at `:3520`. Two fits with the same X and y but different weights could share an identity result; impact depends on whether identity is weight-sensitive.

## 6. LOW-MEDIUM, confirmed: unbounded module-level cache
- `_mrmr_fingerprints.py:85` (`_MRMR_IDENTITY_FP_CACHE`) has no size bound and no clear/pop/del. A long-lived service grows without limit.

## 7. LOW, thread-safety and performance
- `_fe_resident_operands.py:527-536`: `clear_fe_resident_operands` (called from one fit's teardown, `_step_core.py:80`) empties process-wide caches; concurrent fits lose cached entries. Under lock: performance, not correctness.
- `_fe_resident_operands.py:129` (`_HASH_MEMO`) and `_mrmr_fingerprints.py:512` (single-slot last-X cache) shared across fits: thrash under concurrency. Performance only.
- `_numba_warmup.py:83-85`: `_BACKGROUND_STARTED` checked and set without a lock; two background warmups can start. Low impact.

## Checked, not flagged
- `__getstate__` / `__setstate__` (`_mrmr_class.py:3325-3414`) sound overall; `_replay_fitted_state` (`_mrmr_fingerprints.py:693+`) deep-copies mutable attributes and freezes arrays.
- `_evict_cache_entries_for` runs at fit start (`_fit_impl_core_parts.py:871`) under lock.
- `_numba_warmup.py:36-44` double-checked lock is correct.
