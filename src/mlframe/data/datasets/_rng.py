"""Name-addressed RNG streams for the synthetic dataset generator.

Every random draw in this package is taken from a stream addressed by a *name path*
(``stream_for(root_seed, "scenario", "features", "x3")``) rather than by a position in a
spawn sequence. The distinction is not cosmetic:

* ``np.random.SeedSequence(root).spawn(n)`` hands out children by INDEX. Inserting a new
  block of draws in the middle of a generator shifts every later index, so the bytes fed to
  every downstream column change and a scenario that was calibrated yesterday silently
  produces different data today. That failure is invisible: the data still looks fine.
* A name path is order-free and count-free. ``("features", "x3")`` derives the same entropy
  whether it is the first stream created or the thousandth, and whether or not
  ``("features", "x2")`` exists at all. Adding a feature perturbs only that feature's bytes.

Entropy comes from :func:`hashlib.blake2b` of each UTF-8 encoded path segment, never from
Python's builtin :func:`hash`, which is salted by ``PYTHONHASHSEED`` and therefore differs
between processes (this repository has been bitten by exactly that twice - see
``tests/test_rng_determinism_sweep.py`` for the wrapper-subset and fairness-bin regressions).
The precedent generalised here is ``tests/feature_selection/_synth/distributions.py``
(``family_for_operand`` / ``_str_to_int``), which builds a per-operand generator from a
salt-free FNV-1a hash; blake2b is used instead because it is a keyed, standard-library
cryptographic digest with an explicit output width, so the derived integers are stable
across interpreter versions as well as across processes.

Only :func:`numpy.random.default_rng` and :func:`numpy.random.SeedSequence` are used as RNG
entry points, which is what the AST gate in ``tests/test_rng_determinism.py`` permits.

Which parameters belong in a stream path is a design decision, not a convenience: parameters
that change the SHAPE of the data (``n`` rows, ``p`` columns) belong in the path, while
parameters that are swept during calibration (prevalence, signal-to-noise, ceiling targets)
must NOT, or the calibration bisection redraws a fresh dataset at every step and the
recovery-vs-ceiling curve stops being a curve over one dataset.
"""

from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np

#: Width of the blake2b digest used per path segment. 8 bytes keeps every derived value inside
#: the 64-bit words ``SeedSequence`` consumes without an extra reduction step.
_DIGEST_BYTES = 8

#: Separator used only to build the human-readable key in :func:`stream_key`; it never takes
#: part in entropy derivation, so a segment containing it cannot collide with two segments.
PATH_SEPARATOR = "/"


def stable_name_hash(name: str) -> int:
    """Derive a stable unsigned 64-bit integer from ``name``.

    Stable means: identical for the same string in every process, on every platform, under
    every ``PYTHONHASHSEED``, and across interpreter releases. Implemented with blake2b at a
    fixed digest width rather than the builtin :func:`hash`.

    Args:
        name: The path segment to hash. Encoded as UTF-8; may contain any characters.

    Returns:
        An integer in ``[0, 2**64)``.

    Raises:
        TypeError: If ``name`` is not a :class:`str`.
    """
    if not isinstance(name, str):
        raise TypeError(f"stream path segments must be str, got {type(name).__name__}")
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _entropy_for(root_seed: int, path: Tuple[str, ...]) -> List[int]:
    """Build the ``SeedSequence`` entropy list for one named stream.

    The root seed leads so that two different root seeds never share a stream, and the hashed
    segments follow in path order so that ``("a", "b")`` and ``("b", "a")`` differ.

    Args:
        root_seed: Non-negative integer identifying the whole dataset draw.
        path: Already-validated tuple of path segments.

    Returns:
        Entropy list accepted verbatim by :class:`numpy.random.SeedSequence`.

    Raises:
        TypeError: If ``root_seed`` is not an integer (``bool`` is rejected as well: ``True``
            silently meaning seed 1 has never been anyone's intent).
        ValueError: If ``root_seed`` is negative, or a path segment is empty.
    """
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise TypeError(f"root_seed must be an int, got {type(root_seed).__name__}")
    if int(root_seed) < 0:
        raise ValueError(f"root_seed must be non-negative, got {root_seed}")
    entropy = [int(root_seed)]
    for segment in path:
        if not isinstance(segment, str):
            raise TypeError(f"stream path segments must be str, got {type(segment).__name__}")
        if not segment:
            raise ValueError("stream path segments must be non-empty (an empty name addresses nothing)")
        entropy.append(stable_name_hash(segment))
    return entropy


def seed_sequence_for(root_seed: int, *path: str) -> np.random.SeedSequence:
    """Return the :class:`numpy.random.SeedSequence` addressed by ``(root_seed, *path)``.

    Exposed separately from :func:`stream_for` because some consumers need the sequence itself
    (to record its entropy in a manifest, or to hand it to a library that seeds its own
    generator) rather than a live generator.

    Args:
        root_seed: Non-negative integer identifying the whole dataset draw.
        *path: One or more name segments addressing the stream. An empty path is legal and
            addresses the dataset's root stream.

    Returns:
        A seed sequence that is a pure function of ``(root_seed, path)``.
    """
    return np.random.SeedSequence(_entropy_for(root_seed, tuple(path)))


def stream_for(root_seed: int, *path: str) -> np.random.Generator:
    """Return the generator addressed by ``(root_seed, *path)``.

    Deterministic and independent of creation order and of how many other streams exist.

    Args:
        root_seed: Non-negative integer identifying the whole dataset draw.
        *path: One or more name segments addressing the stream, e.g.
            ``stream_for(7, "linear_p50", "features", "x3")``.

    Returns:
        A freshly constructed :class:`numpy.random.Generator`. Each call returns a NEW
        generator positioned at the start of the stream, so callers that want to keep drawing
        must hold on to the object rather than re-addressing it.
    """
    return np.random.default_rng(seed_sequence_for(root_seed, *path))


def stream_key(root_seed: int, *path: str) -> str:
    """Return a human-readable key identifying a stream, for manifests and log lines.

    The key is for provenance only; entropy is derived from the segments themselves, so two
    distinct paths whose segments happen to contain :data:`PATH_SEPARATOR` produce the same
    key but still produce different streams.

    Args:
        root_seed: Non-negative integer identifying the whole dataset draw.
        *path: Name segments addressing the stream.

    Returns:
        ``"<root_seed>:<segment>/<segment>/..."``.
    """
    _entropy_for(root_seed, tuple(path))  # Validate eagerly so a bad key never reaches a manifest.
    return f"{int(root_seed)}:{PATH_SEPARATOR.join(path)}"
