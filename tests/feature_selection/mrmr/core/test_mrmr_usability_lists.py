"""MRMR usability-aware multi-list post-pass -- end-to-end wiring pin.

See ``_usability_lists.py`` + ``_usability_aware_selection.py`` + the design doc. MRMR's pure-MI
``support_`` on the F2 target selects raw ``c`` and ``d`` plus ``a**2/b`` but NOT a ``c*d``
interaction, so a linear model on it sits at ~0.096 test MAE. With ``usability_aware_lists=True``
the fit ALSO produces ``support_linear_`` (a replayable list with a genuine ``(c,d)`` interaction
form); a linear model on the replayed ``transform_usability(X, 'linear')`` space reaches ~the
``f/5`` floor. The pure-MI ``support_`` is byte-identical whether the pass is on or off.
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
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, np.asarray(y, dtype=float)


@pytest.mark.slow
@pytest.mark.timeout(300)  # two MRMR fits (FE) + the CV-MAE usability greedy; see PERF TODO
def test_mrmr_usability_lists_linear_floor_and_byte_identical_support():
    from mlframe.feature_selection.filters import MRMR
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_absolute_error

    n = 12_000 if is_fast_mode() else 18_000
    df, y = _case2(n=n, seed=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.9 * n) :]
    Xtr = df.iloc[tr].reset_index(drop=True)
    Xte = df.iloc[te].reset_index(drop=True)
    ytr, yte = y[tr], y[te]

    light = dict(
        usability_w_linear=0.85,
        usability_greedy_kwargs=dict(shortlist=14, n_folds=3),
        usability_pool_kwargs=dict(max_per_pair=8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs_on = MRMR(verbose=0, random_seed=0, usability_aware_lists=True, **light).fit(X=Xtr, y=pd.Series(ytr, name="y"))
        fs_off = MRMR(verbose=0, random_seed=0).fit(X=Xtr, y=pd.Series(ytr, name="y"))

    # (1) the pass OFF (default) leaves the linear/universal lists unset and aliases the tree list.
    assert fs_off.support_linear_ is None and fs_off.support_universal_ is None
    assert fs_off.support_nonlinear_ is fs_off.support_

    # (2) the pure-MI support_ is BYTE-IDENTICAL whether the usability pass ran or not.
    assert np.array_equal(fs_on.support_, fs_off.support_), "usability pass perturbed support_"
    assert fs_on.support_nonlinear_ is fs_on.support_

    # (3) the linear list is populated and contains a genuine (c,d) interaction form.
    lin = fs_on.support_linear_
    assert lin, "support_linear_ was not populated with usability_aware_lists=True"
    assert any(("c" in cand.src and "d" in cand.src) for cand in lin), f"support_linear_ has no (c,d) interaction form: {[c.name for c in lin]}"

    # (4) a linear model on the REPLAYED usability feature space reaches ~the f/5 floor (<= 0.075),
    #     well below the ~0.096 a pure-MI list (raw c,d, no interaction) gives.
    Ztr = fs_on.transform_usability(Xtr, "linear")
    Zte = fs_on.transform_usability(Xte, "linear")
    assert list(Ztr.columns) == list(Zte.columns) and Ztr.shape[1] == len(lin)
    mdl = make_pipeline(StandardScaler(), LinearRegression()).fit(Ztr.values, ytr)
    mae = mean_absolute_error(yte, mdl.predict(Zte.values))
    assert mae <= 0.075, f"usability linear MAE {mae:.4f} did not approach the f/5 floor (~0.05)"

    # (5) the universal (blend) list is also replayable.
    Zu = fs_on.transform_usability(Xte, "universal")
    assert Zu.shape[0] == Xte.shape[0] and Zu.shape[1] == len(fs_on.support_universal_)
