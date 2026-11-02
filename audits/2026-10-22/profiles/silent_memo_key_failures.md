# Memo keys that failed silently (found 2026-11-02)

Two content-keyed memos computed their key with `array.view(np.uint8)` inside a `try` that returned "no memo" on any exception. An object-backed column (a pandas categorical or a string column named in `cols`) has no byte view, so the key raised, the exception was logged at debug level and the call ran in full, every time. Nothing failed and no test looked at the hit rate.

| memo | symptom | measured |
|---|---|---|
| `_orthogonal_univariate_fe._basis_memo_key` (`generate_univariate_basis_features`) | 9 of 12 calls in a 100000 x 100 every-family fit were misses (the `cols` list of the hybrid-orth families names the categorical column `c0`) | the twelve calls took 3.3 s, 1.8 s after the key hashes such a column with `pd.util.hash_pandas_object`; a hit costs 0.03 s |
| `_fe_content_memo._digest` (`content_memo`) | same failure for any decorated call whose frame holds an object column | not on the profile of this fit, fixed for the same reason |

How it was found: wrapping the memoised function and printing, per call, whether its key was already in the store; the "miss" rows with unchanged column hashes pointed at the key, and the key failed on `X[c].to_numpy().dtype == object`.

Rule for memo keys: a key that cannot be built must not be a silent `None`. Either hash every dtype (values for object columns) or log at warning level through `log_throttle` so a run that stops memoising is visible. `tests/feature_selection/filters/test_univariate_basis_memo.py::test_a_categorical_column_among_cols_still_reaches_the_memo` and `test_fe_content_memo.py::test_a_frame_with_a_categorical_or_string_column_is_still_memoised` pin the two keys.

Not changed, same pattern worth a look: `_mrmr_fingerprints` already hashes object columns by `repr`; `_unified_fe_gate._buffer_hash` and `_y_encoding` only receive numeric arrays.
