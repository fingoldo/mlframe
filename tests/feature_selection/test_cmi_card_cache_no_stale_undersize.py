"""Regression: the CMI cardinality caches must NEVER return a stale-too-small cardinality.

``_cached_card`` / ``joint_cardinalities_cupy``'s ``_YZ_CARD_CACHE`` memoize each operand's histogram WIDTH
(``max(codes)+1``). That width is consumed DIRECTLY as the shared/global joint-histogram extent of the device
kernels (``cmi_joint_entropies`` sizes its tile ``Kx*Ky*Kz`` and indexes it ``(x*Ky+y)*Kz+z``; ``joint_nnz`` /
``joint_counts`` likewise). A cardinality SMALLER than the operand's true ``max+1`` makes a code index PAST the
histogram -> an out-of-bounds ``__shared__`` atomic, i.e. a stray write that lands on the neighbouring
allocation and later surfaces as a misattributed ``cudaErrorIllegalAddress`` (compute-sanitizer:
``Invalid __shared__ atomic ... at cmi_joint_entropies``).

The original key ``(id(host), size, first, last)`` was NOT a safe content fingerprint: after the host array is
GC'd a DIFFERENT operand can reuse the same ``id`` AND match size+endpoints yet have a LARGER max, so the
cache returned the FIRST operand's stale-too-small card for the second -> the OOB. This test pins the
contract with a DETERMINISTIC fingerprint-collision construction (NOT a layout-dependent corruption repro,
which mempool arena rounding masks): two distinct operands engineered to share ``(size, first, last)`` but
differ in their interior max must each get a card >= their own ``max+1``.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as mgc


class _FakeDev:
    """Minimal stand-in for a cupy device-codes array: only ``.size`` and ``.max()`` are used by _cached_card,
    so the contract is exercised without requiring a GPU (CPU-deterministic, runs everywhere)."""

    def __init__(self, host):
        self._h = np.asarray(host)

    @property
    def size(self):
        return self._h.size

    def max(self):
        return int(self._h.max())


def test_cached_card_never_stale_undersize_on_fingerprint_collision(monkeypatch):
    monkeypatch.setattr(mgc, "_CARD_MAX_CACHE", {}, raising=False)

    # Reproduce the GC-id-reuse collision DETERMINISTICALLY: present operand ``buf`` (card 20), then MUTATE the
    # SAME object in place to a different operand (card 100) keeping (size, first, last) identical. The mutated
    # array has the SAME id() AND the same size+endpoints -> it is byte-for-byte the worst case the old
    # ``(id, size, first, last)`` key aliased (and exactly what a recycled id produces). The content-fingerprint
    # key must still return the correct, large-enough card for the mutated content.
    n = 256
    buf = np.zeros(n, dtype=np.int64); buf[0] = 3; buf[-1] = 3; buf[1] = 19    # true card 20
    ca = mgc._cached_card(buf, _FakeDev(buf))
    assert ca == 20, ca

    buf[1] = 99                                                                # mutate in place: true card 100,
    assert buf[0] == 3 and buf[-1] == 3 and buf.size == n                      # SAME (id, size, first, last)
    cb = mgc._cached_card(buf, _FakeDev(buf))

    # The decisive assertion: the card must reflect the CURRENT content's max (100), NOT alias the stale 20 that
    # the weak (id, size, first, last) key returned -- a stale 20 would let a code (up to 99) index past the
    # joint histogram -> the out-of-bounds shared atomic / illegal-address this fix eliminates.
    assert cb == 100, f"stale-undersize cardinality {cb} (expected 100); a code would index past the histogram"
    assert cb > int(buf.max())


def test_cached_card_hits_on_identical_content(monkeypatch):
    """The cache must still HIT on re-presented identical content (the perf contract it exists for): the same
    values via a fresh host object + a device whose .max() would be WRONG if recomputed returns the cached card,
    proving the second call did not recompute (content-keyed hit), and the value is correct."""
    monkeypatch.setattr(mgc, "_CARD_MAX_CACHE", {}, raising=False)
    h = np.array([0, 7, 3, 7, 0], dtype=np.int64)        # true card 8
    first = mgc._cached_card(h, _FakeDev(h))
    assert first == 8
    # Fresh equal-content host (new id) + a device that would report a DIFFERENT max if recomputed: a cache HIT
    # returns the memoized 8, proving the lookup is content-keyed (not recomputed) and stable.
    h2 = np.array([0, 7, 3, 7, 0], dtype=np.int64)
    second = mgc._cached_card(h2, _FakeDev(np.array([0, 1], dtype=np.int64)))
    assert second == 8, f"content-identical operand should hit the cache (got {second})"
