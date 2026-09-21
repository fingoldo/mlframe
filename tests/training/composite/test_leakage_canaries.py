"""No row's derived value depends on its own target beyond the O(1/n) share a global fit gives it.

A leak is flat in n: the grouped causal base that copied each group's first y into its own row had self-influence 1.0,
and an in-sample target encoding gives every row of a size-c category a 1/(c + a) share of its own y at any n. A global fit
shrinks as 1/n, and a local smoother as 1/(its window), so a bound far below 1 and a size-c category fixture separate them.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery import _base_engineering, _grouped_causal_bases
from mlframe.training.composite.discovery._base_engineering import engineer_temporal_bases
from mlframe.training.composite.discovery._grouped_causal_bases import engineer_grouped_causal_bases
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

from .transforms.test_composite_transforms_registry_contract import _base_for, _call_fit, _call_forward, _call_inverse
from .transforms.test_transform_canonical_dgp import _CANONICAL_DGP, _dgp

# Every public y-derived producer, with the engine that computes its values (wrappers attach the engine's columns).
Y_DERIVED_PRODUCERS = {
    "engineer_grouped_causal_bases": "engineer_grouped_causal_bases",
    "attach_grouped_causal_bases": "engineer_grouped_causal_bases",
    "maybe_add_grouped_causal_bases": "engineer_grouped_causal_bases",
    "engineer_temporal_bases": "engineer_temporal_bases",
    "add_engineered_bases_to_pool": "engineer_temporal_bases",
}
_ROWS = (0, 1, 7, 250, 499)


def _frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Five groups interleaved in the frame, a within-group time column, and a target."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"g": np.arange(n) % 5, "t": rng.permutation(n).astype(np.float64), "y": rng.normal(10.0, 2.0, n)})


def _own_row_influence(produce, df: pd.DataFrame, rows=_ROWS) -> float:
    """Largest change of any produced column at row i when ``y_i`` moves by 1 (NaN-to-NaN counts as no change)."""
    ref = produce(df)
    worst = 0.0
    for i in rows:
        moved = df.copy()
        moved.loc[i, "y"] += 1.0
        out = produce(moved)
        for name, col in ref.items():
            a, b = col[i], out[name][i]
            if np.isnan(a) and np.isnan(b):
                continue
            worst = max(worst, abs(b - a) if np.isfinite(a) and np.isfinite(b) else np.inf)
    return worst


def test_every_public_y_derived_producer_is_registered():
    """A new public function in the causal-base modules must be registered with the engine the canary exercises."""
    public = {n for m in (_grouped_causal_bases, _base_engineering) for n, f in vars(m).items()
              if inspect.isfunction(f) and f.__module__ == m.__name__ and not n.startswith("_")}
    assert public == set(Y_DERIVED_PRODUCERS), sorted(public ^ set(Y_DERIVED_PRODUCERS))


def test_causal_bases_never_read_their_own_row():
    """Every grouped and temporal causal base at row i is unchanged when ``y_i`` moves."""
    grouped = lambda d: engineer_grouped_causal_bases(d, "y", "g", "t", lags=(1, 2), trailing_windows=(1, 3), ops=("lag", "trailing_mean", "expanding_mean"))
    temporal = lambda d: engineer_temporal_bases(d, "y", "t", lags=(1, 2), rolling_windows=(1, 3), ops=("lag", "rolling_mean", "rolling_median", "diff"))
    df = _frame()
    assert _own_row_influence(grouped, df) == 0.0
    assert _own_row_influence(temporal, df) == 0.0


def test_the_canary_sees_an_own_row_copy():
    """The legacy ``group_first`` fill copies a group's first y into that row: self-influence 1.0."""
    legacy = lambda d: engineer_grouped_causal_bases(d, "y", "g", "t", ops=("lag",), first_fill="group_first")
    df = _frame()
    heads = df.sort_values("t").groupby("g").head(1).index
    assert _own_row_influence(legacy, df, heads) == pytest.approx(1.0)


def _encoding_influence(n: int, **fit_kw) -> float:
    """Largest change of row i's encoding ``y_i - T_i`` per unit move of ``y_i``, over size-5 categories."""
    t = TRANSFORMS_REGISTRY["target_encoding_residual"]
    rng = np.random.default_rng(0)
    groups = np.arange(n) // 5
    y = rng.normal(10.0, 2.0, n) + 0.5 * (groups % 7)
    enc = y - t.forward(y, None, t.fit(y, None, groups=groups, **fit_kw), groups=groups)
    worst = 0.0
    for i in (3, n // 2, n - 2):
        y2 = y.copy()
        y2[i] += 1.0
        enc2 = y2 - t.forward(y2, None, t.fit(y2, None, groups=groups, **fit_kw), groups=groups)
        worst = max(worst, abs(enc2[i] - enc[i]))
    return worst


def test_train_target_encoding_excludes_each_rows_own_y():
    """Out-of-fold train encodings move by at most 5/n with the row's own y; the in-sample encoding moves by 1/(5 + 20)."""
    n = 2000
    assert _encoding_influence(n) <= 5.0 / n
    assert _encoding_influence(n, oof_folds=None) == pytest.approx(1.0 / 25.0, rel=0.05), "canary: the in-sample encoding must register"


def _registry_self_influence(name: str, n: int) -> float:
    """Largest move of row i's reconstruction ``inverse(T_i, base_i)`` per unit move of ``y_i``, refit params vs the originals."""
    t = TRANSFORMS_REGISTRY[name]
    y, base, base2 = _dgp(_CANONICAL_DGP.get(name, "saturating"), n, 0.0)
    g = (np.arange(n) * 4 // n).astype(np.int64)
    b = _base_for(name, base, base2) if t.requires_base else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = _call_fit(t, y, b, g)
        T = _call_forward(t, y, b, params, g)
        ref = _call_inverse(t, T, b, params, g)
        delta, worst = 0.1 * float(np.std(y)), 0.0
        for i in (7, n // 2 + 1, n - 7):
            y2 = y.copy()
            y2[i] += delta
            worst = max(worst, abs(_call_inverse(t, T, b, _call_fit(t, y2, b, g), g)[i] - ref[i]) / delta)
    return worst


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_a_transforms_params_hold_no_row_own_target(name: str):
    """Refitting with ``y_i`` moved shifts row i's reconstruction by under 0.1 of the move (an own-row copy shifts it by 1).

    Measured at n=2000: global fits sit at O(1/n) (linear_residual 0.002); the local smoothers (rank / copula / spline) at
    0.04-0.07, flat in n by design, which is why the bound is 0.1 and not 5/n.
    """
    assert _registry_self_influence(name, 2000) < 0.1
