"""A datetime column in a training frame crashed `data_signature`, and nothing caught it.

Roughly twenty cache-key sites were rewritten from `h.update(a.tobytes())` to
`h.update(np.ascontiguousarray(a).data)` -- a real optimisation, since these are the KEY
computations and the copy is paid on every lookup including the hits. The rewrite is also wrong for
two dtypes: `.data` raises `ValueError: cannot include dtype 'M' in a buffer` on datetime64 and
timedelta64, which have no buffer-protocol format.

`data_signature` therefore raised on any pandas frame containing a datetime or timedelta column. The
dtype guard directly above the call excluded object and string columns -- for a reason that had
nothing to do with buffers -- and had no cause to think of `M`. A datetime column in a training frame
is not an edge case, and the failure is a hard crash in the cache key rather than a wrong answer, so
it would have surfaced as "training is broken" rather than as anything pointing here.

The fix is one leaf module, `_array_buffer.array_buffer`, rather than a corrected idiom at twenty
sites -- the same reasoning `_dtype_canon` was extracted under. Two copies of a cache-key rule do not
fail loudly when one is fixed and the other is not; they start hashing the same frame to two keys and
the cache quietly stops hitting.

THE DIGESTS DO NOT MOVE. `array_buffer` is byte-identical to both `.tobytes()` and `.data` for every
dtype `.data` accepted, which is asserted below rather than argued: the substitution cannot have
changed a key that any existing cache entry was written under.

Written to py-ci-shared/WRITING_TESTS.md habits 1, 2 and 5.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mlframe._array_buffer import array_buffer

REPO_ROOT = Path(__file__).resolve().parents[2]

_DTYPE_CASES = {
    "float64_c": np.arange(12, dtype=np.float64).reshape(3, 4),
    "float64_f": np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
    "float64_strided": np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2],
    "float32": np.arange(6, dtype=np.float32),
    "int64": np.arange(6, dtype=np.int64),
    "int8": np.arange(6, dtype=np.int8),
    "bool": np.array([True, False, True]),
    "unicode": np.array(["ab", "cde"], dtype="U8"),
    "bytes": np.array([b"ab", b"cde"], dtype="S8"),
    "structured": np.zeros(3, dtype=[("a", "i4"), ("b", "f8")]),
    "scalar_0d": np.array(3.5),
    "empty": np.array([], dtype=np.float64),
}

#: The two the rewrite broke. Kept apart because `.data` raises on them, so they cannot take part
#: in the "identical to `.data`" comparison that proves the digests did not move.
_BUFFERLESS_CASES = {
    "datetime64": np.array([1, 2, 3], dtype="datetime64[ns]"),
    "timedelta64": np.array([1, 2, 3], dtype="timedelta64[ns]"),
    "datetime64_2d": np.arange(20, dtype="int64").reshape(5, 4).view("datetime64[ns]"),
    "datetime64_0d": np.array(np.datetime64("2026-01-01")),
    "datetime64_empty": np.array([], dtype="datetime64[ns]"),
}


class TestTheSubstitutionCannotHaveMovedAnyCacheKey:
    """Every digest written by an earlier run has to still be reachable."""

    @pytest.mark.parametrize("name", sorted(_DTYPE_CASES))
    def test_it_is_byte_identical_to_the_data_form_it_replaced(self, name):
        """The proof that ~20 rewritten sites kept their keys: wherever `.data` worked, this is
        the same bytes, so no existing entry became unreachable."""
        arr = _DTYPE_CASES[name]

        assert bytes(array_buffer(arr)) == bytes(np.ascontiguousarray(arr).data)

    @pytest.mark.parametrize("name", sorted({**_DTYPE_CASES, **_BUFFERLESS_CASES}))
    def test_it_is_byte_identical_to_tobytes(self, name):
        """And to the form BEFORE that one, which is what the oldest cache entries were keyed by."""
        arr = {**_DTYPE_CASES, **_BUFFERLESS_CASES}[name]

        assert bytes(array_buffer(arr)) == np.ascontiguousarray(arr).tobytes()

    def test_a_strided_view_is_serialised_in_c_order(self):
        """A column view of a wider frame is not contiguous. Hashing its raw strided memory would
        key the entry on the frame it was sliced from rather than on its own values."""
        frame = np.arange(24, dtype=np.float64).reshape(4, 6)

        assert bytes(array_buffer(frame[:, ::2])) == np.ascontiguousarray(frame[:, ::2]).tobytes()

    def test_it_does_not_copy_a_contiguous_array(self):
        """The reason the rewrite happened at all: these are cache-KEY computations, so a full
        copy of the frame is paid on every lookup, including the hits."""
        arr = np.arange(1000, dtype=np.float64)

        assert np.shares_memory(np.frombuffer(array_buffer(arr), dtype=np.uint8), arr)


class TestTheDtypesTheRewriteBroke:
    """datetime64 and timedelta64 have no buffer-protocol format, so `.data` cannot serve them."""

    @pytest.mark.parametrize("name", sorted(_BUFFERLESS_CASES))
    def test_the_form_it_replaced_raises_on_them(self, name):
        """Not a hypothetical. This is the exact exception `data_signature` produced, and it is
        asserted so a future 'simplification' back to `.data` fails here rather than in training."""
        arr = _BUFFERLESS_CASES[name]

        with pytest.raises(ValueError, match="buffer"):
            memoryview(np.ascontiguousarray(arr).data)

    @pytest.mark.parametrize("name", sorted(_BUFFERLESS_CASES))
    def test_the_helper_handles_them(self, name):
        """The other half: the replacement has to actually work on what the old form refused."""
        arr = _BUFFERLESS_CASES[name]

        assert hashlib.blake2b(array_buffer(arr), digest_size=8).hexdigest()


class TestASignatureSurvivesADatetimeColumn:
    """The reported failure, driven end to end rather than described."""

    @staticmethod
    def _frame(extra_name, extra_values):
        """A two-feature frame plus a target, with *extra_name* carrying the dtype under test."""
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(
            {
                "num": np.arange(100, dtype=np.float64),
                extra_name: extra_values,
                "y": np.arange(100, dtype=np.float64),
            }
        )

    def test_a_pandas_frame_with_a_datetime_column(self):
        """The exact call that raised. A date column is ordinary in a training frame."""
        pd = pytest.importorskip("pandas")
        from mlframe.training.composite.cache import data_signature

        frame = self._frame("when", pd.date_range("2026-01-01", periods=100, freq="D"))

        assert data_signature(frame, "y", ["num", "when"])

    def test_a_pandas_frame_with_a_timedelta_column(self):
        """The sibling dtype, which fails the same way and is easy to forget when fixing 'M'."""
        pd = pytest.importorskip("pandas")
        from mlframe.training.composite.cache import data_signature

        frame = self._frame("dur", pd.to_timedelta(np.arange(100), unit="D"))

        assert data_signature(frame, "y", ["num", "dur"])

    def test_the_datetime_values_reach_the_signature(self):
        """Not merely "it did not raise": a fix that skipped the column would also not raise, and
        would silently key two different frames to the same entry."""
        pd = pytest.importorskip("pandas")
        from mlframe.training.composite.cache import data_signature

        early = self._frame("when", pd.date_range("2026-01-01", periods=100, freq="D"))
        late = self._frame("when", pd.date_range("2030-06-15", periods=100, freq="D"))

        assert data_signature(early, "y", ["num", "when"]) != data_signature(late, "y", ["num", "when"])


class TestNoSiteSpellsTheBrokenFormOutAgain:
    """The helper only helps while it is the thing being called."""

    def test_no_production_module_feeds_a_hash_through_data(self):
        """Twenty copies of a cache-key rule is what made one broken copy invisible. Benchmarks are
        exempt: several deliberately keep an `_old_hash` to measure the new one against."""
        import re

        offenders = []
        pattern = re.compile(r"np\.ascontiguousarray\([^()]*\)\.data")
        for path in sorted((REPO_ROOT / "src").rglob("*.py")):
            if "_benchmarks" in path.as_posix() or path.name == "_array_buffer.py":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
                if line.lstrip().startswith("#") or not pattern.search(line):
                    continue
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")

        assert offenders == [], "these raise on datetime64/timedelta64; use mlframe._array_buffer.array_buffer:\n  " + "\n  ".join(offenders)
