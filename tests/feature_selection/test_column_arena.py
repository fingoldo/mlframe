"""``ColumnArena`` stores engineered columns once, by name, across chunks and memmap files, and hands them back as views, blocks or a frame equal to the concatenated one."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._column_arena import ColumnArena


def _cols(k=10, n=500, seed=0):
    """``k`` float32 columns named c0..c{k-1}."""
    rng = np.random.default_rng(seed)
    return {f"c{i}": rng.random(n).astype(np.float32) for i in range(k)}


def test_columns_round_trip_across_chunks():
    """Ten columns in chunks of three come back equal, in insertion order, and a view shares the arena memory."""
    cols = _cols()
    with ColumnArena(500, chunk_columns=3) as arena:
        arena.extend(cols)
        assert arena.names == list(cols) and len(arena) == 10
        for name, v in cols.items():
            np.testing.assert_array_equal(arena.column(name), v)
        assert np.shares_memory(arena.column("c4"), arena.column("c4"))


def test_a_repeated_name_keeps_its_first_column_and_a_bad_shape_is_refused():
    """The first owner of a name wins (as the stage merge does); a column of the wrong length raises."""
    with ColumnArena(5) as arena:
        s0 = arena.append("a", np.arange(5, dtype=np.float32))
        assert arena.append("a", np.zeros(5, dtype=np.float32)) == s0
        np.testing.assert_array_equal(arena.column("a"), np.arange(5))
        with pytest.raises(ValueError):
            arena.append("b", np.zeros(4))


def test_values_are_cast_to_the_arena_dtype_in_the_single_copy():
    """A float64 column is stored as float32."""
    with ColumnArena(4) as arena:
        arena.append("x", np.array([1.5, 2.5, 3.5, 4.5]))
        assert arena.column("x").dtype == np.float32


def test_memmap_chunks_hold_the_same_values(tmp_path):
    """With a tiny RAM allowance every chunk after the first is file-backed, and the contents are unchanged."""
    cols = _cols(9, 300)
    arena = ColumnArena(300, chunk_columns=2, memmap_bytes=1, memmap_dir=str(tmp_path))
    arena.extend(cols)
    assert any(isinstance(c, np.memmap) for c in arena._chunks)
    for name, v in cols.items():
        np.testing.assert_array_equal(arena.column(name), v)
    files = list(tmp_path.iterdir())
    assert files
    arena.close()  # a file still mapped may stay on Windows until the temp cleaner runs; only the call must not raise


def test_blocks_respect_the_budget_and_the_frame_equals_concat():
    """Blocks of two columns come out in order; ``as_frame`` equals the frame concatenated from the same columns."""
    cols = _cols(5, 200)
    with ColumnArena(200, chunk_columns=2) as arena:
        arena.extend(cols)
        blocks = list(arena.blocks(budget_bytes=2 * 200 * 4))
        assert [b.shape for _, b in blocks] == [(2, 200), (2, 200), (1, 200)]
        assert all(b.flags.c_contiguous for _, b in blocks)
        pd.testing.assert_frame_equal(arena.as_frame(), pd.DataFrame(cols))


def test_threads_appending_different_columns_do_not_collide():
    """Sixty-four columns appended from eight threads are all present and each equals its source."""
    cols = _cols(64, 400)
    with ColumnArena(400, chunk_columns=5) as arena:
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda kv: arena.append(*kv), cols.items()))
        assert sorted(arena.names) == sorted(cols)
        for name, v in cols.items():
            np.testing.assert_array_equal(arena.column(name), v)


def test_pickling_an_arena_is_refused_with_a_clear_error():
    """The arena owns memmap files and a lock of this process: pickling must fail loudly instead of copying the chunks into RAM."""
    import pickle

    with ColumnArena(500, chunk_columns=3) as arena:
        arena.extend(_cols(k=4))
        with pytest.raises(TypeError, match="process-local"):
            pickle.dumps(arena)
