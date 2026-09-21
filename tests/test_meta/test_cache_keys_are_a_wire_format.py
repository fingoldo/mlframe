"""`data_signature` and `hash_array_summary` produce cache keys, and nothing pinned them.

These two hex strings are written to disk by one run and looked up by the next. Change how any byte
reaches either hash and every existing entry becomes unreachable -- not corrupt, not an error, just a
permanent miss that reads as a cold cache. On the frames this package caches that is hours of
recomputation, and the only symptom is that the cache "stopped helping".

`hash_array_summary` carries no version constant; `data_signature` names its encoding epoch but nothing
checks that an edit bumped it. That makes the literals below the only thing standing between a refactor
and a silently dead cache. The key must also not move when a LIBRARY moves: `data_signature` reads only
mlframe's canonical encoding of the data, so a pandas / polars / numpy upgrade that fails these pins is a
bug in that encoding, not a reason to re-capture.
If a change makes them fail, the correct response is a decision, not an edit: either the change was
not meant to move the keys (fix it), or it was (re-capture, and say so in the commit message so the
next person knows their caches went cold on purpose). Editing the literals to match new output
without noticing is the one response that reproduces the failure this file exists to catch.

THE DTYPE COVERAGE IS THE POINT. Both functions route dtypes down branches with different reasons --
object arrays hold pointers rather than values, datetime64 and Duration have no buffer or no string
cast, empty and 0-d arrays skip the row sampling entirely. Three separate crashes were found in
exactly those branches while building this fixture, all in code that was green:

  * `data_signature` raised on a pandas datetime or timedelta column (no buffer-protocol format).
  * `data_signature` raised on a polars Duration column (no String cast).
  * `hash_array_summary` raised on an object array once the buffer read tightened.

A pin over `float64` alone would have found none of them.

Written to py-ci-shared/WRITING_TESTS.md habits 1, 2 and 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.composite.cache import data_signature
from mlframe.utils.disk_cache import hash_array_summary

REPO_ROOT = Path(__file__).resolve().parents[2]


def _frame(extra: dict | None = None) -> pd.DataFrame:
    """The fixed 60-row frame every signature case is built from."""
    cols: dict = {
        "num": np.arange(60, dtype=np.float64),
        "cat": np.arange(60) % 5,
        "y": np.arange(60, dtype=np.float64),
    }
    if extra:
        cols.update(extra)
    return pd.DataFrame(cols)


#: `{case: (frame, feature_cols)}`. Each name is a dtype branch, not a variation on one.
_SIGNATURE_CASES: dict[str, tuple[pd.DataFrame, list[str]]] = {
    "pandas_plain": (_frame(), ["num", "cat"]),
    "pandas_datetime": (_frame({"when": pd.date_range("2026-01-01", periods=60, freq="D")}), ["num", "when"]),
    "pandas_timedelta": (_frame({"dur": pd.to_timedelta(np.arange(60), unit="D")}), ["num", "dur"]),
    "pandas_object": (_frame({"s": [f"v{i % 7}" for i in range(60)]}), ["num", "s"]),
    "pandas_float32": (_frame({"f32": np.arange(60, dtype=np.float32)}), ["num", "f32"]),
    "pandas_bool": (_frame({"b": (np.arange(60) % 2).astype(bool)}), ["num", "b"]),
}

#: Captured under the canonical encoding (``_canonical_hash``). pandas and polars frames of the same data key identically, so
#: one literal per case serves both. The previous literals went cold with no mlframe change when the environment moved:
#: pandas 3 re-typed strings (``object`` -> ``str``) and inferred ``us`` / ``s`` datetime resolutions, and numpy's platform
#: default int width changed the ``cat`` column -- all folded in via ``str(dtype)`` / ``hash_pandas_object`` / raw bytes.
#: See the module docstring before changing one.
_EXPECTED_SIGNATURES = {
    "pandas_bool": "768de72060a9658012e1082cc9ac33b1",  # pragma: allowlist secret
    "pandas_datetime": "57238ddd55e05163458f78ff4d70b335",  # pragma: allowlist secret
    "pandas_float32": "40711b6020311af157829ae4fa4cf60a",  # pragma: allowlist secret
    "pandas_object": "03bcf5e51caf4840df3192149248c9f0",  # pragma: allowlist secret
    "pandas_plain": "d1972c5eff4248feb71f3cc44e2e62a5",  # pragma: allowlist secret
    "pandas_timedelta": "bd4ce72611507cf8e4db5ac265a727dd",  # pragma: allowlist secret
}


def _array_case(name: str) -> np.ndarray:
    """The array for *name*, each from its OWN seeded generator.

    Deliberately not one shared generator consumed in dict order: that couples every digest to the
    POSITION of its case, so adding or removing one silently invalidates all the pins after it. The
    first version of this file did exactly that and two digests came back wrong for a reason that
    had nothing to do with the hasher.
    """
    rng = np.random.default_rng(0)
    builders = {
        "float64_2d": lambda: rng.random((80, 4)),
        "int64": lambda: rng.integers(0, 100, (80, 3)),
        "bool": lambda: rng.random(50) > 0.5,
        "datetime64": lambda: np.array([1, 2, 3, 4, 5], dtype="datetime64[ns]"),
        "unicode": lambda: np.array(["alpha", "beta", "gamma"], dtype="U10"),
        "object": lambda: np.array([{"k": 1}, "s", 3], dtype=object),
        "empty": lambda: np.array([], dtype=np.float64),
        "scalar_0d": lambda: np.array(3.5),
        "large_1d": lambda: rng.random(5000),
    }
    return builders[name]()


_EXPECTED_ARRAY_DIGESTS = {
    "float64_2d": "33a3597439dc1353af696c949a255817",  # pragma: allowlist secret
    "int64": "f19d237d91f87049480dc0b493046dfc",  # pragma: allowlist secret
    "bool": "cb6ace9558b8d83eb07832c90f97a6ed",  # pragma: allowlist secret
    "datetime64": "9a85c15497dc8c2616a7972b466d7b34",  # pragma: allowlist secret
    "unicode": "dad2477c0a68328ab2012085cf74ee7e",  # pragma: allowlist secret
    "object": "e1ed7d836d4b9e2bd52fb79b5bd18757",  # pragma: allowlist secret
    "empty": "35ee2b600e15d7e452ede5aabad966b9",  # pragma: allowlist secret
    "scalar_0d": "21a6d237fc5dcba66f865ff9fc5b340c",  # pragma: allowlist secret
    "large_1d": "a1d20387f4a14c703115c4616c27e45b",  # pragma: allowlist secret
}

_KEY_CHANGED = (
    "This is a cache KEY. Every entry an earlier run wrote is now unreachable -- a permanent miss "
    "that reads as a cold cache, not as an error. Either the change was not meant to move it (fix "
    "the change), or it was (re-capture, and say so in the commit message)."
)


class TestTheSignatureKeysAreUnchanged:
    """Kills any edit that moves a discovery-cache key without anyone noticing."""

    @pytest.mark.parametrize("name", sorted(_SIGNATURE_CASES))
    def test_the_pandas_signature_is_unchanged(self, name):
        """The key a pandas caller writes and later looks up."""
        frame, features = _SIGNATURE_CASES[name]

        assert data_signature(frame, "y", features) == _EXPECTED_SIGNATURES[name], _KEY_CHANGED

    @pytest.mark.parametrize("name", sorted(_SIGNATURE_CASES))
    def test_the_polars_signature_is_unchanged(self, name):
        """Polars takes a different branch throughout (native select for stats, gather for the sample), so
        it is pinned separately. It pins to the SAME literal: the key encodes the data, not the frame library,
        so a polars caller hits the entries a pandas run wrote."""
        frame, features = _SIGNATURE_CASES[name]

        assert data_signature(pl.from_pandas(frame), "y", features) == _EXPECTED_SIGNATURES[name], _KEY_CHANGED


class TestTheArrayDigestsAreUnchanged:
    """The other key: `hash_array_summary`, used for the on-disk artifact cache."""

    @pytest.mark.parametrize("name", sorted(_EXPECTED_ARRAY_DIGESTS))
    def test_the_digest_is_unchanged(self, name):
        """Every entry an earlier run wrote is keyed by this exact string."""
        assert hash_array_summary(_array_case(name)) == _EXPECTED_ARRAY_DIGESTS[name], _KEY_CHANGED

    def test_every_dtype_branch_of_the_hasher_is_covered(self):
        """The branches carry the subtleties; a pin over float64 alone guards the easy one."""
        kinds = {np.asarray(_array_case(name)).dtype.kind for name in _EXPECTED_ARRAY_DIGESTS}

        for kind in ("f", "i", "b", "M", "U", "O"):
            assert kind in kinds, f"no pinned case exercises dtype kind {kind!r}"


class TestTheKeysStillDiscriminate:
    """A pin proves stability. It does not prove the key is worth anything -- these do."""

    def test_a_changed_value_changes_the_signature(self):
        """A key that ignores the data would be stable AND useless."""
        frame, features = _SIGNATURE_CASES["pandas_plain"]
        moved = frame.copy()
        moved.loc[0, "num"] = moved.loc[0, "num"] + 1.0

        assert data_signature(moved, "y", features) != data_signature(frame, "y", features)

    def test_a_changed_dtype_changes_the_signature(self):
        """Same values under a different dtype train a different model."""
        frame, features = _SIGNATURE_CASES["pandas_plain"]
        retyped = frame.astype({"num": np.float32})

        assert data_signature(retyped, "y", features) != data_signature(frame, "y", features)

    def test_two_different_durations_key_differently(self):
        """The polars Duration branch newly goes through Int64 to get a string at all; it has to
        still carry the values rather than collapsing them to one key."""
        base = {"num": np.arange(30, dtype=np.float64), "y": np.arange(30, dtype=np.float64)}
        short = pl.DataFrame({**base, "d": pd.to_timedelta(np.arange(30), unit="D")})
        long = pl.DataFrame({**base, "d": pd.to_timedelta(np.arange(30) * 7, unit="D")})

        assert data_signature(short, "y", ["num", "d"]) != data_signature(long, "y", ["num", "d"])

    def test_an_object_array_keys_on_its_values_not_its_addresses(self):
        """The regression this branch was added for: an object array's buffer is PyObject*
        addresses, which differ in every process, so a key built from it could never hit. Equal
        content built twice must give one key."""
        first = np.array([{"k": 1}, "s", 3], dtype=object)
        second = np.array([{"k": 1}, "s", 3], dtype=object)

        assert hash_array_summary(first) == hash_array_summary(second)
        assert hash_array_summary(first) != hash_array_summary(np.array([{"k": 2}, "s", 3], dtype=object))
