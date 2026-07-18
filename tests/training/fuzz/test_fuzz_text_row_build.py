"""Regression test for the vectorised text-row build in build_frame_for_combo.

Pre-fix the text-column build did
    rows = []
    for _ in range(n):
        idxs = rng.integers(0, len(text_vocab), size=3)
        rows.append(" ".join(text_vocab[j] for j in idxs))

which iter536 OOMed on at n=200k under concurrent profiler memory pressure --
the per-row Python int idx-list + generator-expression `" ".join(text_vocab[j]
for j in idxs)` allocates and immediately discards ~n small Python objects.
The replacement vectorises:
    vocab_arr = np.asarray(text_vocab)
    idxs_arr = rng.integers(0, len(text_vocab), size=(n, 3))
    words = vocab_arr[idxs_arr]
    rows = list(map(" ".join, words))

which is bit-identical (same vocab order + same rng draw sequence -> same
token positions) and ~6.5x faster at n=200k while avoiding the per-row
intermediate list.

This test pins:
  (1) vectorised path produces the exact same row sequence as the old path
      (rng state preserved via reset between the two builds).
  (2) frame builds successfully at n=50k with a text column.
"""

from __future__ import annotations

import numpy as np
import pytest


_TEXT_VOCAB = [
    "python",
    "rust",
    "golang",
    "java",
    "swift",
    "kotlin",
    "backend",
    "frontend",
    "devops",
    "mlops",
    "dataeng",
    "platform",
    "cloud",
    "edge",
    "realtime",
    "batch",
    "stream",
    "vector",
    "search",
    "nlp",
    "vision",
    "audio",
    "robotics",
    "quantum",
]


def _old_rows(n: int, seed: int):
    """Old rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        idxs = rng.integers(0, len(_TEXT_VOCAB), size=3)
        rows.append(" ".join(_TEXT_VOCAB[j] for j in idxs))
    return rows


def _new_rows(n: int, seed: int):
    """New rows."""
    rng = np.random.default_rng(seed)
    vocab_arr = np.asarray(_TEXT_VOCAB)
    idxs_arr = rng.integers(0, len(_TEXT_VOCAB), size=(n, 3))
    words = vocab_arr[idxs_arr]
    return list(map(" ".join, words))


@pytest.mark.parametrize("n,seed", [(100, 1), (10_000, 42), (50_000, 20260528)])
def test_vectorised_text_rows_match_old_per_row_build(n, seed):
    """Vectorised text rows match old per row build."""
    assert _new_rows(n, seed) == _old_rows(n, seed)


def test_build_frame_for_combo_with_text_col_does_not_oom_at_50k():
    """End-to-end smoke: the fuzz harness's frame builder returns a text col
    with the expected shape at n=50k. iter536 OOMed at n=200k; 50k is enough
    to catch the regression without making the test slow."""
    from tests.training._fuzz_combo import build_frame_for_combo, enumerate_combos

    combos = enumerate_combos(target=150, master_seed=20260422)
    # Pick a combo where the builder actually emits a text column. The builder gates emission on the
    # full ``want_text`` condition (_fuzz_combo.build_frame_for_combo): text_col_count>0 AND cb in
    # models AND auto_detect_cats. The auto_detect_cats clause is required because a text column is
    # only routable when auto-detection classifies it as a text_feature (CB consumes it, non-CB
    # models exclude it); with auto_detect_cats=False the raw object column would reach a non-CB
    # numeric pipeline and crash, so emission is suppressed. Match that gate here.
    text_combo = None
    for c in combos:
        if c.text_col_count > 0 and "cb" in c.models and c.auto_detect_cats:
            text_combo = c
            break
    if text_combo is None:
        pytest.skip("no combo with text_col_count>0 + cb + auto_detect_cats in 150-combo pool")

    import dataclasses

    small = dataclasses.replace(text_combo, n_rows=50_000)
    df, _, _ = build_frame_for_combo(small)
    text_cols = [c for c in df.columns if c.startswith("text_")]
    assert text_cols, "expected at least one text_* column"
    col = text_cols[0]
    # Cardinality / shape sanity: every row should be a 3-token space-joined str.
    if hasattr(df, "head"):
        sample = df.head(20)
        col_data = sample[col] if hasattr(sample, "__getitem__") else sample[:, col]
        try:
            iterator = col_data.to_list()
        except AttributeError:
            iterator = list(col_data)
        for s in iterator:
            assert isinstance(s, str)
            assert len(s.split()) == 3
