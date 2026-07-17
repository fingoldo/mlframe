"""biz_value: cheap-first prewarp->optuna dispatch for the orthogonal-poly path.

The expensive CMA/Optuna orthogonal-poly search (``fe_smart_polynom_iters``)
only earns its cost on pairs whose signal a trivial library unary/binary feature
CANNOT already capture -- i.e. non-monotone inner distortions
(``c**3-2c`` etc.) where the trivial MI sits well below the pair's joint-MI
ceiling. For a MONOTONE-easy pair (``exp(a)*log(b)``) the trivial feature is
already at the ceiling, so a 1-D polynomial cannot beat it; running the
optimiser there is wasted wall-clock.

``poly_cheap_skip_ratio`` (default 0.97, threaded via
``fe_poly_cheap_skip_ratio``) skips the optimiser for any pair whose trivial
baseline reaches >= that fraction of the joint-MI ceiling. This file pins the
CONTRACT, not raw timings (timings are flaky in CI):

* the HARD non-monotone pair is still recovered with the gate ON (the optimiser
  runs where it is needed),
* the gate ON does NOT bloat the support (the easy pair's redundant poly
  duplicate is dropped -- the cheap feature already covers it),
* downstream predictive quality with the gate ON is NOT worse than with it OFF
  (no signal lost), and
* the gate ON does strictly LESS expensive work (its support carries no MORE
  ``_polynom_`` features than the gate-OFF run) -- the falsifiable proxy for
  "fewer optimiser calls".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR

_LEAN = dict(dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False)


def _make_mixed(seed: int = 0, n: int = 2500):
    """One MONOTONE-easy pair (exp(a)*log(b)) + one NON-monotone-hard pair
    ((c**3-2c)(d**2-d)) + a noise column. Both contribute comparably to y."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.2, 2.0, n)
    b = rng.uniform(1.2, 5.0, n)
    c = rng.uniform(-2.5, 2.5, n)
    d = rng.uniform(-2.5, 2.5, n)
    e = rng.normal(0.0, 1.0, n)
    easy = np.exp(a) * np.log(b)
    hard = (c**3 - 2 * c) * (d**2 - d)
    y = easy / np.std(easy) + hard / np.std(hard) + 0.1 * e
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), pd.Series(y, name="y")


def _fit(ratio: float):
    df, y = _make_mixed()
    MRMR.clear_fit_cache()
    m = MRMR(
        verbose=0,
        random_seed=0,
        fe_smart_polynom_iters=3,
        fe_smart_polynom_optimization_steps=120,
        fe_polynomial_basis="chebyshev",
        fe_optimizer="cma_batch",
        fe_hybrid_orth_enable=False,
        **_LEAN,
    )
    m.fe_poly_cheap_skip_ratio = ratio
    m.fit(df, y)
    return df, y, m


def _ridge_r2(Xt, y):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, KFold

    Xt = np.asarray(Xt)
    if Xt.ndim == 1:
        Xt = Xt.reshape(-1, 1)
    if Xt.shape[1] == 0:
        return float("nan")
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    return float(np.mean(cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), Xt, np.asarray(y, dtype=float), cv=cv, scoring="r2")))


def _n_poly(support):
    return sum(1 for s in support if str(s).startswith("_polynom_"))


def _covers_cd_hard(support):
    """Any poly feature over the hard (c,d) pair -- the optimiser-only signal."""
    return any(str(s).startswith("_polynom_") and ("c" in str(s)) and ("d" in str(s)) for s in support)


def test_cheap_first_recovers_hard_pair_and_does_not_bloat():
    _df, _y, m_off = _fit(1.0)  # gate OFF: optimise every prospective pair (legacy)
    _, _, m_on = _fit(0.97)  # gate ON (default): skip cheaply-saturated pairs
    sup_off = list(m_off.get_feature_names_out())
    sup_on = list(m_on.get_feature_names_out())

    # The HARD non-monotone (c,d) pair needs the optimiser; it must still run +
    # recover with the gate ON.
    assert _covers_cd_hard(sup_on), f"cheap-first gate ON dropped the hard (c,d) poly recovery; support={sup_on}"
    # The gate must do strictly no MORE expensive work: no more _polynom_ features
    # than the gate-OFF run (the easy pair's redundant poly duplicate is skipped).
    assert _n_poly(sup_on) <= _n_poly(sup_off), (
        f"cheap-first gate ON carried MORE poly features than OFF ({_n_poly(sup_on)} > {_n_poly(sup_off)}); it should skip, not add. on={sup_on} off={sup_off}"
    )


def test_cheap_first_preserves_downstream_quality():
    df, y, m_off = _fit(1.0)
    _, _, m_on = _fit(0.97)
    r2_off = _ridge_r2(m_off.transform(df), y)
    r2_on = _ridge_r2(m_on.transform(df), y)
    # Skipping the optimiser only where the cheap feature already saturates the
    # ceiling must NOT cost downstream R^2 (the cheap feature covers that pair).
    assert r2_on >= r2_off - 0.03, f"cheap-first gate ON regressed downstream R^2: off={r2_off:.4f}, on={r2_on:.4f}"
