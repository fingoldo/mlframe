"""Unit and determinism tests for name-addressed RNG streams (``mlframe.data.datasets._rng``).

The determinism trio the generator's reproducibility rests on:

1. bit-identity for the same ``(root_seed, path)``;
2. INSERTION INVARIANCE - adding a stream in the middle must not move any other stream's bytes, which is
   precisely what ``SeedSequence.spawn(n)`` fails to give;
3. cross-process invariance under two different ``PYTHONHASHSEED`` values, checked in real subprocesses
   (never via ``importlib.reload``, which cannot change the interpreter's hash salt).
"""

import os
import subprocess
import sys
from pathlib import Path

import mlframe
import numpy as np
import pytest

from mlframe.data.datasets._rng import (
    PATH_SEPARATOR,
    seed_sequence_for,
    stable_name_hash,
    stream_for,
    stream_key,
)

# Directory to put on the child interpreter's PYTHONPATH so the subprocess imports THIS checkout rather
# than whatever editable install happens to be active in the shared environment.
_SRC_ROOT = str(Path(mlframe.__file__).resolve().parents[1])


def _draw(root_seed: int, *path: str) -> np.ndarray:
    """Return a fixed-length draw from the stream addressed by ``(root_seed, *path)``.

    Args:
        root_seed: Root seed of the dataset.
        *path: Name segments addressing the stream.

    Returns:
        Eight uint64 values, enough that an accidental collision is not plausible.
    """
    return stream_for(root_seed, *path).integers(0, 2**63 - 1, size=8, dtype=np.int64)


def test_stable_name_hash_is_in_range_and_pure():
    """The name hash is a pure function into the unsigned 64-bit range."""
    value = stable_name_hash("x3")
    assert 0 <= value < 2**64
    assert value == stable_name_hash("x3")
    assert stable_name_hash("x3") != stable_name_hash("x4")


def test_stable_name_hash_rejects_non_str():
    """A non-string segment is a programming error, not something to coerce."""
    with pytest.raises(TypeError):
        stable_name_hash(3)  # type: ignore[arg-type]


def test_stream_for_is_bit_identical_for_same_address():
    """Same ``(seed, path)`` yields bit-identical draws."""
    np.testing.assert_array_equal(_draw(11, "scenario", "features", "x3"), _draw(11, "scenario", "features", "x3"))


def test_stream_for_separates_seed_and_path():
    """Different seeds, different paths, and different path ORDER all give different streams."""
    base = _draw(11, "features", "x3")
    assert not np.array_equal(base, _draw(12, "features", "x3"))
    assert not np.array_equal(base, _draw(11, "features", "x4"))
    assert not np.array_equal(base, _draw(11, "x3", "features"))


def test_stream_for_insertion_invariance():
    """Inserting a stream in the middle moves nobody else's bytes.

    This is the property positional ``SeedSequence.spawn(n)`` does not have: there, adding a block shifts
    every later index and silently redraws every downstream column.
    """
    names = ["x1", "x2", "x3", "x4"]
    before = {name: _draw(7, "features", name) for name in names}

    # Create an extra stream "in the middle", and re-address the originals in a completely different order.
    inserted_order = ["x4", "x2", "x_inserted", "x1", "x3"]
    after = {name: _draw(7, "features", name) for name in inserted_order}

    for name in names:
        np.testing.assert_array_equal(before[name], after[name], err_msg=f"stream {name!r} moved after an insertion")
    assert not any(np.array_equal(after["x_inserted"], before[name]) for name in names)


def test_stream_for_independent_of_how_many_streams_exist():
    """Addressing one stream after creating a thousand others gives the same bytes as addressing it first."""
    first = _draw(5, "features", "target_column")
    for index in range(1000):
        stream_for(5, "noise", f"col{index}")
    np.testing.assert_array_equal(first, _draw(5, "features", "target_column"))


def test_seed_sequence_for_matches_stream_for():
    """The exposed seed sequence is the one ``stream_for`` uses, not a parallel derivation."""
    sequence = seed_sequence_for(3, "a", "b")
    np.testing.assert_array_equal(
        np.random.default_rng(sequence).integers(0, 2**63 - 1, size=8, dtype=np.int64),
        _draw(3, "a", "b"),
    )


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "7", True])
def test_stream_for_rejects_bad_root_seed(bad_seed):
    """A negative, float, string or bool root seed is refused rather than silently coerced."""
    with pytest.raises((TypeError, ValueError)):
        stream_for(bad_seed, "a")


def test_stream_for_rejects_empty_segment():
    """An empty path segment addresses nothing and is refused."""
    with pytest.raises(ValueError):
        stream_for(1, "features", "")


def test_stream_key_is_readable_and_validated():
    """The provenance key is human-readable and validates its own arguments."""
    assert stream_key(7, "features", "x3") == f"7:features{PATH_SEPARATOR}x3"
    with pytest.raises(ValueError):
        stream_key(-1, "features")


_CHILD_PROGRAM = """
import numpy as np
from mlframe.data.datasets._rng import stable_name_hash, stream_for

print(stable_name_hash("x3"))
print(stream_for(7, "features", "x3").integers(0, 2**63 - 1, size=8, dtype=np.int64).tolist())
"""


def _run_child(hash_seed: str) -> str:
    """Run the stream-derivation program in a fresh interpreter under a given ``PYTHONHASHSEED``.

    A subprocess is the only honest way to test this: ``PYTHONHASHSEED`` is consumed once at interpreter
    start, so reloading the module inside this process would prove nothing.

    Args:
        hash_seed: Value to set ``PYTHONHASHSEED`` to for the child.

    Returns:
        The child's stdout.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = _SRC_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, f"child failed: {completed.stderr}"
    return completed.stdout


def test_streams_are_invariant_across_pythonhashseed():
    """Two interpreters with different hash salts derive byte-identical streams.

    Guards the failure this repository has hit twice: entropy derived from the builtin ``hash`` looks
    perfectly deterministic within one process and differs between them.
    """
    assert _run_child("0") == _run_child("12345")
