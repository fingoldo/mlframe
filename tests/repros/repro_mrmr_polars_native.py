"""Reproduce/verify MRMR native Polars support (no .to_pandas() copy).

Regression guard for Fix 10 (2026-04-22): MRMR.fit used to call X.to_pandas()
when X was a pl.DataFrame — full materialization, prohibitive on 100+ GB
production frames (see CLAUDE.md). This script asserts:

  1. MRMR.fit(pl.DataFrame, y) completes without error.
  2. MRMR.transform(pl.DataFrame) returns a pl.DataFrame (subset of original).
  3. No pl.DataFrame.to_pandas() is called during fit on the full frame
     (only the cat-col subset for OrdinalEncoder is allowed).
"""
import sys
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe")

import numpy as np
import polars as pl
from mlframe.feature_selection.filters import MRMR

n = 500
rng = np.random.default_rng(42)
pl_df = pl.DataFrame({
    "num1": rng.standard_normal(n).astype(np.float32),
    "num2": rng.standard_normal(n).astype(np.float32),
    "num3": rng.standard_normal(n).astype(np.float32),
    "cat1": pl.Series(["A", "B", "C"] * (n // 3 + 1))[:n].cast(pl.Enum(["A", "B", "C"])),
})
y = rng.integers(0, 2, n)

# Count total pl.DataFrame.to_pandas() calls during fit
orig_to_pandas = pl.DataFrame.to_pandas
call_count = {"n": 0}
def _spy(self, *args, **kwargs):
    call_count["n"] += 1
    return orig_to_pandas(self, *args, **kwargs)
pl.DataFrame.to_pandas = _spy

try:
    sel = MRMR(
        verbose=0, max_runtime_mins=1, n_workers=1,
        quantization_nbins=5, use_simple_mode=True,
        min_nonzero_confidence=0.9, max_consec_unconfirmed=3,
        full_npermutations=3,
    )
    sel.fit(pl_df, y)
    print(f"fit completed. to_pandas() calls during fit: {call_count['n']}")
    # Expected: 1 call for the cat-col subset inside categorize_dataset
    # (bounded by #cat_cols × n_rows, not full frame).
    assert call_count["n"] <= 1, (
        f"Too many .to_pandas() calls: {call_count['n']}. "
        f"Fix 10 regression — MRMR should only convert cat-col subset."
    )

    out = sel.transform(pl_df)
    print(f"transform returned {type(out).__name__} with {len(out.columns) if hasattr(out, 'columns') else '?'} cols")
    assert isinstance(out, pl.DataFrame), (
        f"transform should return pl.DataFrame when input is pl.DataFrame; got {type(out).__name__}"
    )

    print("PASS")
finally:
    pl.DataFrame.to_pandas = orig_to_pandas
