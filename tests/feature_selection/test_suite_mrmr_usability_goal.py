"""End-to-end /goal pin: train_mlframe_models_suite + usability-aware MRMR reaches the linear floor.

On the heavy-tailed F2 target ``y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3)`` (``f`` hidden), MRMR's
pure-MI selection gives a LINEAR downstream raw ``c``, ``d`` and ``a**2/b`` but NO ``c*d``
interaction, so the suite's linear model sits at ~0.099 test MAE (VERIFIED separately). With
``feature_selection_config.mrmr_usability_aware_lists=True`` MRMR runs its usability second pass and
``transform`` materialises the UNION of all three selection lists, so the engineered interaction is
in the linear model's input and its test MAE drops to ~the ``f/5`` floor (~0.05-0.06).

Runs ONE suite fit per process (the suite caches fitted pipelines by ``id(df)`` + content signature,
so an OFF-then-ON pair in the SAME process would let the OFF run's imputer -- fit on the narrower
pure-MI columns -- be reused against the wider union transform; the existing suite-FE tests isolate
each fit in a subprocess for the same reason). The ~0.099 pure-MI baseline is asserted by the wider
suite-FE recovery suite; here we pin the usability-ON goal.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode


def _case2(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "y": y.astype(float)})


def _linear_test_mae(entries) -> float:
    best = float("inf")
    for e in entries:
        tt = getattr(e, "test_target", None)
        tp = getattr(e, "test_preds", None)
        if tt is None or tp is None:
            continue
        tt = np.asarray(tt, dtype=float).ravel()
        tp = np.asarray(tp, dtype=float).ravel()
        if tt.size and tt.size == tp.size:
            best = min(best, float(np.mean(np.abs(tt - tp))))
    return best


@pytest.mark.slow
@pytest.mark.timeout(600)  # one suite fit (FE) + MRMR's usability CV-MAE pass; see PERF TODO
def test_suite_linear_reaches_floor_with_usability_aware_mrmr():
    from tests.feature_selection._suite_fe_helpers import run_suite

    n = 10_000 if is_fast_mode() else 14_000
    df = _case2(n=n, seed=0)
    usability = dict(
        usability_aware_lists=True,
        usability_greedy_kwargs=dict(shortlist=14, n_folds=3),
        usability_pool_kwargs=dict(max_per_pair=8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entries, _ = run_suite(df, model="linear", use_mrmr=True, random_seed=0, mrmr_kwargs=usability)

    assert entries, "suite returned no fitted linear entries"
    mae = _linear_test_mae(entries)
    assert np.isfinite(mae), "no test predictions found on the fitted linear entries"
    # the irreducible f/5 floor is ~0.05; the union puts the engineered (c,d) interaction in the
    # linear model's input, taking it well below the ~0.099 pure-MI baseline.
    assert mae <= 0.075, (
        f"suite linear test MAE {mae:.4f} did not approach the f/5 floor (~0.05) with the "
        f"usability-aware union (pure-MI baseline is ~0.099)"
    )
