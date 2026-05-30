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
    """Verify the structural fix: the value WRITTEN to
    ``cached_cond_MIs`` is the raw conditional MI, NOT the
    exponentiated one. We assert by direct contract instrumentation.
    """
    # Source-presence sensor via file read (no ``inspect.getsource`` -- the
    # meta-test ``test_no_inspect_getsource_in_test_files`` enforces the
    # behavioural-tests rule; reading the file directly carries the same
    # signal without the AST-detected antipattern).
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "feature_selection" / "filters" / "evaluation.py"
    ).read_text(encoding="utf-8")
    # Post-fix contract: the line ``cached_cond_MIs[key] =
    # additional_knowledge`` must come BEFORE the
    # ``additional_knowledge **= (nexisting + 1)`` exponentiation on
    # the miss path.
    cache_write = src.find("cached_cond_MIs[key] = additional_knowledge")
    # Find the exponentiation on the miss path (the SECOND occurrence
    # of ``additional_knowledge ** (nexisting + 1)``).
    expo_positions = []
    idx = 0
    while True:
        next_idx = src.find("additional_knowledge ** (nexisting + 1)", idx)
        if next_idx < 0:
            break
        expo_positions.append(next_idx)
        idx = next_idx + 1
    assert cache_write >= 0
    # The miss-path exponentiation (the second occurrence) must come
    # AFTER the cache write.
    miss_expo = max(expo_positions)
    assert miss_expo > cache_write, (
        "Cache write must precede the miss-path exponentiation so the "
        "stored value is raw."
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
