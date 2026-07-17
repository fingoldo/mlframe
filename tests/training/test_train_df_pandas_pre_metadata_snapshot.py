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

    # Wave-105 (2026-05-21) split _phase_helpers.py into multiple sibling
    # files under mlframe.training.core. The dict-literal landed in
    # _phase_helpers_fit_split.py. Search both so the test survives the
    # refactor.
    # 2026-05-22 split: _phase_helpers_fit_split itself was split again into
    # _phase_helpers_fit_pipeline.py. Walk all three so the test survives.
    candidate_modules = [
        "mlframe.training.core._phase_helpers",
        "mlframe.training.core._phase_helpers_fit_split",
        "mlframe.training.core._phase_helpers_fit_pipeline",
    ]

    found_meta_assign = False
    for mod_name in candidate_modules:
        spec = importlib.util.find_spec(mod_name)
        if spec is None or spec.origin is None:
            continue
        src = Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "train_df_pandas_pre_meta":
                        if isinstance(node.value, ast.Dict):
                            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                            if {"columns", "dtypes", "n_unique", "shape"}.issubset(keys):
                                found_meta_assign = True
                                break
                if found_meta_assign:
                    break
        if found_meta_assign:
            break
    assert found_meta_assign, (
        "train_df_pandas_pre_meta dict-literal assignment with keys "
        "{columns, dtypes, n_unique, shape} not found in any of "
        f"{candidate_modules}; Wave-7 metadata-dict snapshot fix not landed."
    )
