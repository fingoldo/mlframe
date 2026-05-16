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
    train_df = pd.DataFrame({
        "skills_text": [f"token_{i % 20}" for i in range(n)],
        "score": rng.standard_normal(n).astype(np.float64),
        "category": pd.Categorical([f"cat_{i % 5}" for i in range(n)]),
    })

    shallow = train_df.copy(deep=False)
    meta = _build_meta_dict(train_df)

    pre_unique_text = train_df["skills_text"].nunique()
    pre_score_first = float(train_df["score"].iloc[0])

    # In-place numpy poke that escapes the shallow-copy block-manager refcount.
    train_df["score"].values[0] = -9999.0
    # Replace the object-dtype column wholesale to also confirm dtype-swap escape.
    train_df["skills_text"] = train_df["skills_text"].astype("category").cat.codes

    # Shallow snapshot LEAKS the numpy poke (block-manager is shared).
    assert shallow["score"].iloc[0] == -9999.0, (
        "Test premise broken: shallow copy unexpectedly isolated the numpy mutation. "
        "Update test to use a mutation pattern that escapes block-manager refcount."
    )

    # Metadata-dict snapshot survives unscathed.
    assert meta["columns"] == ["skills_text", "score", "category"]
    assert meta["dtypes"]["skills_text"] == "object"
    assert meta["dtypes"]["score"] == "float64"
    assert str(meta["dtypes"]["category"]).startswith("category")
    assert meta["n_unique"]["skills_text"] == pre_unique_text
    assert meta["shape"] == (n, 3)
    # Sanity: the source frame and the shallow snapshot now disagree with the metadata snapshot.
    assert float(train_df["score"].iloc[0]) != pre_score_first
    assert str(train_df["skills_text"].dtype) != "object"


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
    """End-to-end: the snapshot block at ``_phase_helpers.py`` lines ~920-950 must
    populate ``train_df_pandas_pre_meta`` with the expected keys when invoked
    through the same code path the auto-detect phase exercises.

    We probe the module source via ast rather than running the full phase to
    keep the test light. The construction lives inside a deep helper that
    expects a populated TrainingContext; the metadata dict shape is the public
    contract we can lock down here.
    """
    import ast
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("mlframe.training.core._phase_helpers")
    assert spec is not None and spec.origin is not None
    src = Path(spec.origin).read_text(encoding="utf-8")
    tree = ast.parse(src)

    found_meta_assign = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "train_df_pandas_pre_meta":
                    # ensure the RHS is a Dict literal with the expected keys
                    if isinstance(node.value, ast.Dict):
                        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                        if {"columns", "dtypes", "n_unique", "shape"}.issubset(keys):
                            found_meta_assign = True
                            break
            if found_meta_assign:
                break
    assert found_meta_assign, (
        "train_df_pandas_pre_meta dict-literal assignment with keys "
        "{columns, dtypes, n_unique, shape} not found in _phase_helpers.py; "
        "Wave-7 metadata-dict snapshot fix not landed."
    )
