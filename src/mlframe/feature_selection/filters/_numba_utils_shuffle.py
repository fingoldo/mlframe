"""Inline random draws for the Fisher-Yates loops of the permutation nulls.

numba's ``np.random.randint`` (a Mersenne Twister draw with rejection sampling) cost about five times the scoring of the permutations it fed. These helpers are an xorshift64* generator seeded through splitmix64 and a multiply-shift reduction of a
32-bit value to ``[0, bound)`` (the bias is below ``bound / 2**32``). A permutation seeded ``s`` is a pure function of ``s``, so rows drawn on different threads or in different chunks are the same data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def seed_state(seed: int) -> np.uint64:
    """Non-zero xorshift state of ``seed`` (splitmix64 finaliser)."""
    z = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = z ^ (z >> np.uint64(31))
    if x == np.uint64(0):
        x = np.uint64(1)
    return np.uint64(x)


@njit(cache=True, inline="always")
def next_state(x: np.uint64) -> np.uint64:
    """One xorshift64 step."""
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    return x


@njit(cache=True, inline="always")
def bounded(x: np.uint64, bound: int) -> Any:
    """A value in ``[0, bound)`` from the state ``x`` (``bound`` at most 2**32)."""
    r = (x * np.uint64(2685821657736338717)) >> np.uint64(32)
    return np.int64((r * np.uint64(bound)) >> np.uint64(32))


@njit(cache=True)
def shuffle_labels_into(src: np.ndarray, dst: np.ndarray, seed: int) -> None:
    """``dst`` = ``src`` read through the Fisher-Yates shuffle seeded ``seed`` (``src[seeded_permutation(len(src), seed)]``). The one permutation draw of the MDLP split tests: the serial
    per-node test, the per-node null and the batched level test all call it, so they cannot drift to different streams."""
    n = src.shape[0]
    for i in range(n):
        dst[i] = src[i]
    x = seed_state(seed)
    for i in range(n - 1, 0, -1):
        x = next_state(x)
        j = bounded(x, i + 1)
        t = dst[i]
        dst[i] = dst[j]
        dst[j] = t
