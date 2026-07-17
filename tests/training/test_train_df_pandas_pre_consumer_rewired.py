"""Tests for the rewired ``_auto_detect_feature_types`` consumer.

The legacy ``train_df_pandas_pre = train_df.copy(deep=False)`` shallow-copy was kept solely so the
downstream auto-detect phase could keep reading a DataFrame. It shared the source frame's
block-manager and leaked any in-place numpy poke (``df[col].values[i] = x``) into the snapshot,
silently corrupting auto-detect's view of pre-encoding dtypes / cardinality.

This module pins the post-rewire contract:

1. ``_auto_detect_feature_types`` accepts a ``pandas_meta`` dict and routes every read through it
   (column names, dtype strings, n_unique, non-null counts, embedding-shape sniff).
2. Mutating the source train_df after the metadata snapshot is built must NOT change the
   auto-detect output -- the dict is immune by construction.
3. The legacy shallow-copy slot (``train_df_pandas_pre`` on the phase return tuple / TrainingContext)
   is gone; the slot now carries the metadata dict.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def test_auto_detect_uses_metadata_dict_when_supplied():
    """Pass ``pandas_meta`` only (no live frame contents driving the read) and assert auto-detect
    still classifies the columns correctly. The frame is intentionally a stale post-encoding view
    (integer codes); only the metadata dict carries the pre-encoding signal that lets auto-detect
    promote the text column. If the consumer were still reading frame columns the test would fail.
    """
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    n = 800
    rng = np.random.default_rng(7)
    pre_df = pd.DataFrame(
        {
            "skills_text": [f"token_{i % 600}" for i in range(n)],
            "tiny_cat": [f"c_{i % 4}" for i in range(n)],
            "numeric_a": rng.standard_normal(n),
        }
    )
    pre_meta = {
        "columns": list(pre_df.columns),
        "dtypes": {c: str(pre_df[c].dtype) for c in pre_df.columns},
        "n_unique": {
            c: int(pre_df[c].nunique(dropna=True)) for c in pre_df.columns if pre_df[c].dtype.kind in "OUSb" or isinstance(pre_df[c].dtype, pd.CategoricalDtype)
        },
        "non_null": {
            c: int(pre_df[c].notna().sum()) for c in pre_df.columns if pre_df[c].dtype.kind in "OUSb" or isinstance(pre_df[c].dtype, pd.CategoricalDtype)
        },
        "embedding_object_cols": [],
        "shape": tuple(pre_df.shape),
    }
    # Post-encoding stale view: skills_text and tiny_cat are now int codes; if the consumer reads
    # this frame it would treat them all as numeric and skip the promotion.
    post_df = pd.DataFrame(
        {
            "skills_text": rng.integers(0, 600, size=n, dtype=np.int32),
            "tiny_cat": rng.integers(0, 4, size=n, dtype=np.int32),
            "numeric_a": rng.standard_normal(n),
        }
    )

    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=50,
    )

    text_features, embedding_features, drop = _auto_detect_feature_types(
        post_df,
        cfg,
        cat_features=[],
        verbose=False,
        pandas_meta=pre_meta,
    )
    assert "skills_text" in text_features, f"Metadata-dict path failed to promote high-card text column; got text={text_features}"
    assert "tiny_cat" not in text_features, f"Low-card column should NOT be promoted; got text={text_features}"
    assert embedding_features == []
    assert drop == []


def test_auto_detect_mutation_immune_to_train_df_mutation():
    """Mutate the source frame AFTER snapshotting and assert the auto-detect output is unchanged.

    Locks the whole point of the refactor: the dict captures dtype + cardinality at snapshot time,
    so no later in-place poke / column replacement on the source frame can corrupt the result.
    """
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    n = 500
    train_df = pd.DataFrame(
        {
            "skills_text": [f"u_{i % 400}" for i in range(n)],
            "numeric_a": np.arange(n, dtype=np.float64),
        }
    )
    pre_meta = {
        "columns": list(train_df.columns),
        "dtypes": {c: str(train_df[c].dtype) for c in train_df.columns},
        "n_unique": {"skills_text": int(train_df["skills_text"].nunique(dropna=True))},
        "non_null": {"skills_text": int(train_df["skills_text"].notna().sum())},
        "embedding_object_cols": [],
        "shape": tuple(train_df.shape),
    }
    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=50,
    )

    text_pre, _, _ = _auto_detect_feature_types(
        train_df,
        cfg,
        cat_features=[],
        verbose=False,
        pandas_meta=pre_meta,
    )

    # Numpy-array in-place poke (escapes block-manager refcount) + wholesale column dtype swap.
    # pandas 2.3+ marks .to_numpy() and .values views as read-only even
    # after assigning a fresh writable numpy array to a column. Use
    # .iloc[:] = scalar (pandas-native setitem) instead -- the metadata-
    # snapshot consumer doesn't care HOW the value was mutated, only that
    # the source frame changed. iloc[:] correctly bypasses CoW restrictions
    # on a per-column basis.
    train_df.iloc[:, train_df.columns.get_loc("numeric_a")] = -1.0
    train_df["skills_text"] = train_df["skills_text"].astype("category").cat.codes

    text_post, _, _ = _auto_detect_feature_types(
        train_df,
        cfg,
        cat_features=[],
        verbose=False,
        pandas_meta=pre_meta,
    )
    assert text_pre == text_post == ["skills_text"], f"Metadata-dict consumer must be immune to source mutation; pre={text_pre} post={text_post}"


def test_metadata_dict_embedding_object_cols_routed_to_embedding():
    """Object cells holding ndarray / list (sentence-transformer style vectors) must be routed to
    ``embedding_features`` based on the precomputed ``embedding_object_cols`` list -- the frame
    cells must NEVER be probed again at consumer time.
    """
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    n = 200
    train_df = pd.DataFrame(
        {
            "emb_col": [np.zeros(8, dtype=np.float32) for _ in range(n)],
            "skills_text": [f"u_{i % 150}" for i in range(n)],
        }
    )
    pre_meta = {
        "columns": ["emb_col", "skills_text"],
        "dtypes": {"emb_col": "object", "skills_text": "object"},
        "n_unique": {"skills_text": 150},
        "non_null": {"skills_text": n},
        "embedding_object_cols": ["emb_col"],
        "shape": (n, 2),
    }
    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=50,
    )

    # Wipe the live frame's emb_col so that if the consumer ever falls back to probing live cells
    # it would mis-classify; the metadata dict says "embedding" so the consumer must trust it.
    train_df["emb_col"] = [None] * n

    text, embedding, drop = _auto_detect_feature_types(
        train_df,
        cfg,
        cat_features=[],
        verbose=False,
        pandas_meta=pre_meta,
    )
    assert "emb_col" in embedding, f"emb_col should be in embedding_features; got {embedding}"
    assert "emb_col" not in text
    assert "skills_text" in text


def test_legacy_shallow_copy_removed_from_phase_helpers_return_tuple():
    """The phase return tuple must carry ``train_df_pandas_pre_meta`` (dict) and NOT the legacy
    frame-shaped ``train_df_pandas_pre``. The phase is heavy to run end-to-end inside a unit
    test (full pipeline + split + dispatch), so the assertion is behavioural at the
    plumbing layer: the auto-detect phase signature uses the meta kwarg, and the
    TrainingContext slot has been renamed.
    """
    from mlframe.training.core._phase_helpers import _phase_auto_detect_feature_types
    from mlframe.training.core._training_context import TrainingContext

    sig = inspect.signature(_phase_auto_detect_feature_types)
    assert "train_df_pandas_pre_meta" in sig.parameters, "Phase signature must accept train_df_pandas_pre_meta after the rewire."
    assert "train_df_pandas_pre" not in sig.parameters, f"Legacy train_df_pandas_pre kwarg must be gone after the rewire; got params: {list(sig.parameters)}"

    # TrainingContext slot rename: meta dict in, frame slot out.
    ctx_slots = set(getattr(TrainingContext, "__dataclass_fields__", {}).keys())
    assert "train_df_pandas_pre_meta" in ctx_slots, f"TrainingContext must expose train_df_pandas_pre_meta slot; got: {sorted(ctx_slots)}"
    assert "train_df_pandas_pre" not in ctx_slots, (
        f"Legacy train_df_pandas_pre slot must be dropped from TrainingContext; still present in: {sorted(ctx_slots)}"
    )


def test_auto_detect_falls_back_to_frame_when_meta_absent():
    """Back-compat: with ``pandas_meta=None`` the consumer must still work via the live frame.

    This path remains for legacy callers that never threaded the metadata dict; the dict path is
    the preferred one whenever a snapshot was taken.
    """
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    n = 600
    df = pd.DataFrame(
        {
            "skills_text": [f"u_{i % 400}" for i in range(n)],
            "numeric_a": np.arange(n, dtype=np.float64),
        }
    )
    cfg = FeatureTypesConfig(
        auto_detect_feature_types=True,
        use_text_features=True,
        cat_text_cardinality_threshold=50,
    )
    text, embedding, drop = _auto_detect_feature_types(
        df,
        cfg,
        cat_features=[],
        verbose=False,
        pandas_meta=None,
    )
    assert "skills_text" in text
    assert embedding == []
    assert drop == []
