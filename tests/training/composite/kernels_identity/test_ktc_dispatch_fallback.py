"""Unit tests for the composite corr / collinear KTC backend lookup.

``_ktc_dispatch.choose_corr_backend`` / ``choose_collinear_backend`` route the
numba-vs-numpy backend choice through the kernel_tuning_cache.  These tests pin the
two safety contracts that matter regardless of whether a sweep has ever run:

1. **Clean fallback when the cache is unavailable** -- with pyutilz / the cache
   monkeypatched to ``None`` the lookup must return exactly the hardcoded size gate
   (numba iff both dims clear the min thresholds, numpy otherwise).
2. **Env-var force-override wins** -- ``MLFRAME_COMPOSITE_CORR_BACKEND`` /
   ``MLFRAME_COMPOSITE_COLLINEAR_BACKEND`` pin the backend irrespective of size and
   irrespective of the cache, and an unrecognised value is ignored (no silent pin).

Both backends are bit-identical by construction, so this only needs to verify the
routing logic, not numeric equality (that is pinned in the bit-identity sensors).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery import _ktc_dispatch as kd
from mlframe.training.composite.discovery._collinear_numba import (
    _MIN_COLS as COLL_MIN_COLS,
    _MIN_ROWS as COLL_MIN_ROWS,
    near_collinear_keep_mask_fast,
)
from mlframe.training.composite.discovery._corr_numba import (
    _MIN_COLS as CORR_MIN_COLS,
    _MIN_ROWS as CORR_MIN_ROWS,
    safe_abs_corr_all_dispatch,
)

_CORR_KW = dict(min_rows=CORR_MIN_ROWS, min_cols=CORR_MIN_COLS)
_COLL_KW = dict(min_rows=COLL_MIN_ROWS, min_cols=COLL_MIN_COLS)


@pytest.fixture
def no_cache(monkeypatch):
    """Force the cache lookup to report 'no cache' so only the fallback path runs."""
    monkeypatch.setattr(kd, "_get_cache", lambda: None)
    monkeypatch.delenv(kd._CORR_ENV, raising=False)
    monkeypatch.delenv(kd._COLLINEAR_ENV, raising=False)


def test_corr_fallback_is_hardcoded_size_gate(no_cache):
    # Both dims clear the gate -> numba; either below -> numpy. Exactly the old gate.
    assert kd.choose_corr_backend(CORR_MIN_ROWS, CORR_MIN_COLS, **_CORR_KW) == "numba"
    assert kd.choose_corr_backend(CORR_MIN_ROWS - 1, CORR_MIN_COLS, **_CORR_KW) == "numpy"
    assert kd.choose_corr_backend(CORR_MIN_ROWS, CORR_MIN_COLS - 1, **_CORR_KW) == "numpy"
    assert kd.choose_corr_backend(10, 4, **_CORR_KW) == "numpy"


def test_collinear_fallback_is_hardcoded_size_gate(no_cache):
    assert kd.choose_collinear_backend(COLL_MIN_ROWS, COLL_MIN_COLS, **_COLL_KW) == "numba"
    assert kd.choose_collinear_backend(COLL_MIN_ROWS - 1, COLL_MIN_COLS, **_COLL_KW) == "numpy"
    assert kd.choose_collinear_backend(COLL_MIN_ROWS, COLL_MIN_COLS - 1, **_COLL_KW) == "numpy"
    assert kd.choose_collinear_backend(8, 2, **_COLL_KW) == "numpy"


def test_corr_env_override_pins_backend(no_cache, monkeypatch):
    # numba forced even on a tiny input that the size gate would route to numpy.
    monkeypatch.setenv(kd._CORR_ENV, "numba")
    assert kd.choose_corr_backend(10, 4, **_CORR_KW) == "numba"
    # numpy forced even on a huge input that the size gate would route to numba.
    monkeypatch.setenv(kd._CORR_ENV, "numpy")
    assert kd.choose_corr_backend(CORR_MIN_ROWS * 4, CORR_MIN_COLS * 4, **_CORR_KW) == "numpy"
    # Case-insensitive + whitespace tolerant.
    monkeypatch.setenv(kd._CORR_ENV, "  NumPy  ")
    assert kd.choose_corr_backend(CORR_MIN_ROWS, CORR_MIN_COLS, **_CORR_KW) == "numpy"


def test_collinear_env_override_pins_backend(no_cache, monkeypatch):
    monkeypatch.setenv(kd._COLLINEAR_ENV, "numba")
    assert kd.choose_collinear_backend(8, 2, **_COLL_KW) == "numba"
    monkeypatch.setenv(kd._COLLINEAR_ENV, "numpy")
    assert kd.choose_collinear_backend(COLL_MIN_ROWS * 4, COLL_MIN_COLS * 4, **_COLL_KW) == "numpy"


def test_unrecognised_env_value_is_ignored(no_cache, monkeypatch):
    # A typo must NOT silently pin a backend -- it falls through to the size gate.
    monkeypatch.setenv(kd._CORR_ENV, "gpu")
    assert kd.choose_corr_backend(CORR_MIN_ROWS, CORR_MIN_COLS, **_CORR_KW) == "numba"
    assert kd.choose_corr_backend(10, 4, **_CORR_KW) == "numpy"
    monkeypatch.setenv(kd._COLLINEAR_ENV, "")
    assert kd.choose_collinear_backend(8, 2, **_COLL_KW) == "numpy"


def test_cache_exception_falls_back(monkeypatch):
    # A cache that raises on get_or_tune must degrade to the hardcoded gate, not crash.
    class _Boom:
        def lookup(self, *a, **k):
            raise RuntimeError("cache exploded")

    monkeypatch.setattr(kd, "_get_cache", lambda: _Boom())
    monkeypatch.delenv(kd._CORR_ENV, raising=False)
    assert kd.choose_corr_backend(CORR_MIN_ROWS, CORR_MIN_COLS, **_CORR_KW) == "numba"
    assert kd.choose_corr_backend(10, 4, **_CORR_KW) == "numpy"


def test_cache_backend_choice_is_honoured(monkeypatch):
    # When the cache returns a valid backend_choice, the lookup uses it over the gate.
    class _Fake:
        def __init__(self, choice):
            self.choice = choice

        def lookup(self, *a, **k):
            return {"backend_choice": self.choice}

    monkeypatch.setattr(kd, "_get_cache", lambda: _Fake("numpy"))
    monkeypatch.delenv(kd._CORR_ENV, raising=False)
    # Size gate would say numba, but the cache says numpy -> numpy wins.
    assert kd.choose_corr_backend(CORR_MIN_ROWS, CORR_MIN_COLS, **_CORR_KW) == "numpy"

    monkeypatch.setattr(kd, "_get_cache", lambda: _Fake("numba"))
    # Size gate would say numpy, but the cache says numba -> numba wins.
    assert kd.choose_corr_backend(10, 4, **_CORR_KW) == "numba"

    # Garbage backend_choice from the cache is ignored -> falls back to the gate.
    monkeypatch.setattr(kd, "_get_cache", lambda: _Fake("garbage"))
    assert kd.choose_corr_backend(10, 4, **_CORR_KW) == "numpy"


def test_dispatchers_still_bit_identical_through_ktc_path(monkeypatch):
    # End-to-end: with the cache absent the dispatchers must still produce results
    # identical to forcing each backend via the env override (bit-identity invariant
    # holds regardless of which backend the lookup picks).
    monkeypatch.setattr(kd, "_get_cache", lambda: None)
    rng = np.random.default_rng(0)
    n, f = max(CORR_MIN_ROWS, 20_000), max(CORR_MIN_COLS, 64)
    X = rng.standard_normal((n, f))
    y = X[:, 0] * 0.7 + rng.standard_normal(n) * 0.3

    def _ref(yv, Xv):
        Xc = Xv - Xv.mean(0)
        yc = yv - yv.mean()
        denom = np.sqrt((Xc * Xc).sum(0) * float(np.dot(yc, yc)))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.abs((Xc * yc[:, None]).sum(0) / denom)
        out[~np.isfinite(out)] = 0.0
        return out

    monkeypatch.setenv(kd._CORR_ENV, "numba")
    via_numba = safe_abs_corr_all_dispatch(y, X, reference_fn=_ref)
    monkeypatch.setenv(kd._CORR_ENV, "numpy")
    via_numpy = safe_abs_corr_all_dispatch(y, X, reference_fn=_ref)
    np.testing.assert_allclose(via_numba, via_numpy, atol=1e-9, rtol=0)


def test_collinear_dispatch_bit_identical_through_ktc_path(monkeypatch):
    monkeypatch.setattr(kd, "_get_cache", lambda: None)
    rng = np.random.default_rng(1)
    n, b = max(COLL_MIN_ROWS, 300), max(COLL_MIN_COLS, 12)
    M = rng.standard_normal((n, b))
    M[:, 1] = M[:, 0] * 0.999 + rng.standard_normal(n) * 1e-3  # near-collinear pair

    def _ref(fm, *, corr_threshold):
        keep = np.ones(fm.shape[1], dtype=bool)
        kept = []
        for j in range(fm.shape[1]):
            cj = fm[:, j] - fm[:, j].mean()
            drop = False
            for k in kept:
                ck = fm[:, k] - fm[:, k].mean()
                d = np.sqrt(np.dot(cj, cj) * np.dot(ck, ck))
                if d > 0 and abs(np.dot(cj, ck) / d) > corr_threshold:
                    drop = True
                    break
            if drop:
                keep[j] = False
            else:
                kept.append(j)
        return keep

    monkeypatch.setenv(kd._COLLINEAR_ENV, "numba")
    via_numba = near_collinear_keep_mask_fast(M, corr_threshold=0.99, reference_fn=_ref)
    monkeypatch.setenv(kd._COLLINEAR_ENV, "numpy")
    via_numpy = near_collinear_keep_mask_fast(M, corr_threshold=0.99, reference_fn=_ref)
    np.testing.assert_array_equal(via_numba, via_numpy)
