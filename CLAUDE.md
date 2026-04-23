# mlframe — project conventions

## Memory / RAM constraints (CRITICAL)

**Frames in mlframe can be 100+ GB.** Never copy them to work around a bug.
Copying a prod DataFrame doubles peak RAM, which on a 200 GB+ workload means
OOM — the user observed this in 2026-04-22 prod logs.

Avoid:
- `df.copy()` (pandas) or `df.clone()` (polars) inside hot paths
- `df[cols] = df[cols].astype(...)` when `df` is the caller's frame (pandas
  broadcasts-copies the sub-frame)
- Constructing a fresh `pd.DataFrame(df)` / `pl.DataFrame(df)` to "get a new
  reference"
- Any fit-transform pattern that returns a mutated input

Prefer:
- Work on views (`.iloc`, column selection, slices)
- Mutate-and-restore: `X[col] = new; try: ... finally: del X[col]`
- Use `with` / context managers that revert the mutation on exit
- Lazy eval via polars `lazy()` + `.collect()` at the leaf call
- Pass `inplace` options where sklearn / the transformer supports them

Fuzz-caught example: MRMR.fit needed to temporarily inject a `targ_<id>`
column into X for MI computation. Original code mutated caller's X
in place, leaked the injected column into downstream sklearn steps, and
tripped `validate_data` on the next transform. Fix in
`feature_selection/filters.py:~2895` must inject + remove the column in a
try/finally (never call `X.copy()`).

If you find a bug that genuinely needs a copy, escalate — the user would
rather ship a design change than accept an unconditional copy on a hot
DataFrame path.

## Open work items

### Full MRMR Feature Engineering (FE) support for Polars input

Fix 10 (2026-04-22) made `MRMR.fit(pl.DataFrame, y)` work without any
`.to_pandas()` copy of the main frame — the selector itself runs on a
zero-copy polars path (`to_physical()` for cat codes, `.to_numpy()` on a
column select for numerics).

**FE (feature engineering) inside MRMR is still pandas-only and is
auto-disabled on polars input** (`filters.py:~2890`, gated on
`_is_polars_input and fe_max_steps > 0`). This is a conservative
workaround — the selector quality is preserved, but users who want FE
on polars frames currently get a silent `fe_max_steps=0`.

Affected sites in `feature_selection/filters.py` (pandas-only ops):
- L3184, 3537, 3679 — `X.iloc[:, idx].values`
- L3324 — `X[col] = transformed_vals[:, j]`
- L3623 — `X.iloc[:, original_cols[var]].values` in
  `check_prospective_fe_pairs`

To add full polars FE support later:
- Replace `X.iloc[:, idx].values` with `X[:, idx].to_numpy()` on the
  polars branch (zero-copy for numerics).
- Replace in-place `X[col] = vals` with `X = X.with_columns(pl.Series(...))`
  on the polars branch.
- Add direct tests: `tests/training/test_mrmr_polars_fe.py` plus a
  biz-value test that enables FE (`fe_max_steps > 0`) on a polars frame
  and asserts FE-generated columns appear in `selected_features_`.

Estimated scope: ~80 LOC changes across 4-5 sites + ~100 LOC of tests.
Deferred — current conservative gate keeps the selector usable on
polars while FE remains pandas-only.
