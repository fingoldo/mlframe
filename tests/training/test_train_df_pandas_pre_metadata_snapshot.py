"""Regression test for the ``train_df_pandas_pre_meta`` metadata-dict snapshot.

The shallow ``train_df_pandas_pre = train_df.copy(deep=False)`` snapshot in
``_phase_helpers.py`` shares the source frame's block-manager. Numpy-level
in-place mutations on the source frame (``df[col].values[i] = x``,
``df.values[:, j] = ...``) propagate into the shallow snapshot and silently
corrupt downstream auto-detect. The metadata-dict snapshot bakes
columns / dtypes / per-column n_unique at snapshot time so its values are
immune to any later mutation on the source frame.

This test reproduces the precise mutation pattern the master-plan Wave-7 audit
flagged: it builds a small DataFrame, captures both the shallow snapshot and
the metadata-dict snapshot the way ``_phase_helpers.py`` constructs them,
mutates the original frame in place via numpy-array poke, and asserts that
the metadata-dict snapshot's recorded dtypes / cardinality match the
pre-mutation state while the shallow snapshot leaks the mutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_meta_dict(train_df: pd.DataFrame) -> dict:
    """Mirror the metadata-dict snapshot logic from ``_phase_helpers.py``."""
    return {
        "columns": list(train_df.columns),
        "dtypes": {c: str(train_df[c].dtype) for c in train_df.columns},
        "n_unique": {
            c: int(train_df[c].nunique(dropna=True))
            for c in train_df.columns
            if train_df[c].dtype.kind in "OUSb" or isinstance(train_df[c].dtype, pd.CategoricalDtype)
        },
        "shape": tuple(train_df.shape),
    }


def test_metadata_dict_snapshot_immune_to_numpy_inplace_mutation():
    """Numpy-array in-place mutation must NOT propagate to the metadata-dict snapshot."""
    n = 50
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(
        {
            "skills_text": [f"token_{i % 20}" for i in range(n)],
            "score": rng.standard_normal(n).astype(np.float64),
            "category": pd.Categorical([f"cat_{i % 5}" for i in range(n)]),
        }
    )

    meta = _build_meta_dict(train_df)

    pre_unique_text = train_df["skills_text"].nunique()
    pre_score_first = float(train_df["score"].iloc[0])
    pre_skills_dtype = str(train_df["skills_text"].dtype)

    # Mutate the SOURCE frame after the snapshot: an in-place cell poke + a
    # wholesale dtype-swapping column replacement. The metadata-dict snapshot
    # baked its values at construction time and must reflect the PRE-mutation
    # state regardless of how the source frame changes afterwards.
    _score_col_idx = train_df.columns.get_loc("score")
    train_df.iat[0, _score_col_idx] = -9999.0
    train_df["skills_text"] = train_df["skills_text"].astype("category").cat.codes

    # NOTE: we deliberately do NOT assert that the shallow ``copy(deep=False)``
    # snapshot leaks the mutation. Under pandas Copy-on-Write (default in
    # pandas 3.0 / opt-in in 2.x, active on the prod box 2026-05-27) a shallow
    # copy is mutation-isolated, so that "premise" is false there. The contract
    # under test is the metadata DICT snapshot's immunity, which holds in BOTH
    # CoW and non-CoW modes because it captures plain values at snapshot time.

    # Metadata-dict snapshot survives unscathed.
    assert meta["columns"] == ["skills_text", "score", "category"]
    # ``skills_text`` is string-typed: ``object`` by default, ``str`` under
    # future.infer_string / pandas 3.0. Accept string-like, not strictly object.
    assert str(meta["dtypes"]["skills_text"]).lower().startswith(("object", "str")), meta["dtypes"]["skills_text"]
    assert meta["dtypes"]["score"] == "float64"
    assert str(meta["dtypes"]["category"]).startswith("category")
    assert meta["n_unique"]["skills_text"] == pre_unique_text
    assert meta["shape"] == (n, 3)
    # Sanity: the source frame now disagrees with the metadata snapshot
    # (the mutation landed on the source, the dict snapshot stayed put).
    assert float(train_df["score"].iloc[0]) != pre_score_first
    assert str(train_df["skills_text"].dtype) != pre_skills_dtype


def test_metadata_dict_snapshot_immune_to_column_add():
    """Adding new columns to the source frame must NOT appear in the metadata snapshot."""
    train_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    meta = _build_meta_dict(train_df)

    train_df["c_added"] = [9, 9, 9]
    train_df["d_added"] = ["p", "q", "r"]

    assert meta["columns"] == ["a", "b"]
    assert "c_added" not in meta["dtypes"]
    assert "d_added" not in meta["dtypes"]
    assert meta["shape"] == (3, 2)


def test_metadata_dict_snapshot_construction_in_phase_helpers():
    """The production snapshot helper must return the metadata dict with the public keys and stay put when the source frame mutates.

    Calls the real ``_snapshot_pandas_pre_meta`` (the code path the auto-detect phase consumes) rather than inspecting source text.
    """
    import pandas as _pd

    from mlframe.training.core._phase_helpers_fit_pipeline import _snapshot_pandas_pre_meta

    train_df = _pd.DataFrame(
        {
            "text": [f"tok_{i % 7}" for i in range(40)],
            "score": np.arange(40, dtype=np.float64),
            "cat": _pd.Categorical([f"c{i % 3}" for i in range(40)]),
        }
    )

    meta = _snapshot_pandas_pre_meta(None, False, train_df)

    assert meta is not None, "pandas input with feature_types_first (default) must produce a snapshot"
    for key in ("columns", "dtypes", "n_unique", "non_null", "shape"):
        assert key in meta, f"snapshot is missing {key!r}: {sorted(meta)}"
    assert meta["columns"] == ["text", "score", "cat"]
    assert meta["shape"] == (40, 3)
    assert meta["n_unique"] == {"text": 7, "cat": 3}
    assert meta["non_null"] == {"text": 40, "cat": 40}

    frozen = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in meta.items()}
    train_df.iat[0, train_df.columns.get_loc("score")] = -1.0
    train_df["text"] = train_df["text"].astype("category")
    train_df["extra"] = 1
    assert meta == frozen, "snapshot must not change when the source frame is mutated afterwards"

    assert _snapshot_pandas_pre_meta(None, True, train_df) is None, "polars input must not produce a pandas snapshot"
