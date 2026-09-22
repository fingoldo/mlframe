"""The booster-dataset cache key must separate frames that differ only in dtype, or whose content it cannot hash.

`compute_signature`'s docstring said it read dtypes; it did not, so a tier transition that recast the same columns
float64 -> float32 (or int -> category) produced an identical key. And when the row-sample hash failed, the key fell
back to columns + shape alone, so two different frames of the same shape shared it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training import _dataset_cache_fingerprint as fp


def _df():
    return pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})


def test_identical_frames_share_a_key():
    assert fp.compute_signature(_df()) == fp.compute_signature(_df())


def test_a_recast_changes_the_key():
    assert fp.compute_signature(_df()) != fp.compute_signature(_df().astype(np.float32))
    assert fp.compute_signature(pl.from_pandas(_df())) != fp.compute_signature(pl.from_pandas(_df().astype(np.float32)))
    assert fp.compute_signature(np.zeros((3, 2))) != fp.compute_signature(np.zeros((3, 2), dtype=np.float32))
    cat = _df().assign(a=_df()["a"].astype(int).astype("category"))
    assert fp.compute_signature(_df().assign(a=_df()["a"].astype(int))) != fp.compute_signature(cat)


def test_an_unhashable_frame_never_matches(monkeypatch):
    monkeypatch.setattr(fp, "_row_sample_hash", lambda X, n: None)
    assert fp.compute_signature(_df()) != fp.compute_signature(_df())
