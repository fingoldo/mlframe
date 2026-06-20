"""Wave 9.1 loop-iter-37 regression: ``cached_cond_MIs`` must NOT
store nexisting-exponentiated values keyed without ``nexisting``.

Pre-fix at ``evaluation.py:434``:

    cached_cond_MIs[key] = additional_knowledge   # AFTER exponentiation

The same key ``arr2str(X) + "|" + arr2str(Z)`` was reused across all
``nexisting`` values, but the stored value at line 431 had already
been raised to the power ``nexisting + 1``. A subsequent (X, Z)
lookup with a DIFFERENT ``nexisting`` returned the wrong exponent and
silently biased the Fleuret / CMIM redundancy score by a factor of
``cmi^(Δnexisting)``.

Exposure: default ``only_unknown_interactions=False`` lets candidates
whose components overlap with already-selected vars flow through with
``nexisting > 0``, and ``max_veteranes_interactions_order >= 2`` makes
X a tuple that can partially overlap. Every multi-order MRMR run hit
this silently.

Fix: cache the RAW (pre-exponent) value and apply
``** (nexisting + 1)`` at every read AND on the miss path's local
``additional_knowledge`` AFTER the cache write. The cache is then
nexisting-independent.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def test_cache_raw_value_independent_of_nexisting():
    """Behavioural discriminator for the iter-37 fix: the cache stores the RAW
    conditional MI, so a second lookup of the same ``(X, Z)`` with a larger
    ``nexisting`` re-applies ``** (nexisting + 1)`` to the raw value rather than
    returning the first call's exponent. Scoring the same candidate twice
    against a SHARED cache (nexisting=0 then nexisting=2) must therefore satisfy
    ``score(nexisting=2) == score(nexisting=0) ** 3``. Pre-fix the cache held
    the already-exponentiated value (raw ** 1) and the read path applied no
    further exponent, so the second score equalled the first -> this assert
    fails on pre-fix code and passes post-fix.
    """
    import numba
    from numba.core import types

    from mlframe.feature_selection.filters.evaluation import evaluate_gain

    rng = np.random.default_rng(0)
    n = 600
    # Three binned (low-cardinality) columns: candidate X=col0, target y=col1,
    # already-selected Z=col2. A genuine X-Y dependence that survives
    # conditioning on Z keeps the raw CMI strictly in (0, 1), where ``r ** 3``
    # is well-separated from ``r``.
    col0 = rng.integers(0, 3, size=n)
    col2 = rng.integers(0, 2, size=n)
    noise = rng.integers(0, 2, size=n)
    col1 = (col0 + 2 * col2 + noise) % 4
    factors_data = np.column_stack([col0, col1, col2]).astype(np.int32)
    factors_nbins = np.array([3, 4, 2], dtype=np.int32)

    # ``evaluate_gain`` does ``np.array(selected_vars, ...)`` internally and uses X/y
    # as multiset index sequences, so pass plain Python lists (the types the fit feeds it).
    X = [0]
    y = [1]
    selected_vars = [2]

    def _score(nexisting: int, cache) -> float:
        entropy_cache = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        _, current_gain, _, _ = evaluate_gain(
            current_gain=1.0e9,
            last_checked_k=-1,
            direct_gain=0.0,
            X=X,
            y=y,
            nexisting=nexisting,
            best_gain=-1.0e9,
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars,
            max_veteranes_interactions_order=1,
            extra_knowledge_multipler=-1.0,
            entropy_cache=entropy_cache,
            cached_cond_MIs=cache,
        )
        return current_gain

    shared_cache = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    raw = _score(0, shared_cache)
    exponentiated = _score(2, shared_cache)

    assert 0.0 < raw < 1.0, f"raw CMI must lie in (0,1) for the exponent gap to be detectable; got {raw}"
    assert exponentiated == pytest.approx(raw ** 3, rel=1e-9), (
        "Second lookup with nexisting=2 must re-exponentiate the RAW cached "
        f"value: expected {raw ** 3!r}, got {exponentiated!r}. A non-cubed "
        "result means the cache stored the already-exponentiated value."
    )


def test_mrmr_with_interactions_order_2_runs_clean():
    """End-to-end: enabling 2-way interactions exercises the
    ``nexisting > 0`` path. The iter-37 fix must not regress runtime
    behaviour for the common case.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = 400
    latent_a = rng.standard_normal(n)
    latent_b = rng.standard_normal(n)
    X = pd.DataFrame({
        "a": latent_a,
        "b": latent_b,
        "c": latent_a + 0.1 * rng.standard_normal(n),
        "d": rng.standard_normal(n),
    })
    y = pd.Series((latent_a * latent_b > 0).astype(np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(
            verbose=0,
            max_veteranes_interactions_order=2,
            only_unknown_interactions=False,
        ).fit(X, y)
    # Must complete and return at least one selected feature.
    assert len(sel.support_) >= 1
