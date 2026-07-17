"""BIZ-VALUE: the two-sided cross-model-family contract of the MAIN FE-inside-MRMR
path (the univariate hermite / orth basis), proven end-to-end.

The team's documented position is "complementarity, not universal lift": engineered
features make a NON-ADDITIVE signal LINEARLY usable (a big LIFT for a linear model)
while a gradient-boosted tree -- which already models the non-linearity natively --
sees NO HARM (the engineered columns are not junk that dilutes the tree's split
search). ``test_biz_value_mrmr_fe_downstream_delta.py`` pins the LINEAR side of this
on the single-step PAIR-COMPOSITE path (``fe_max_steps=1``) and only spot-checks the
tree direction in prose. This file pins the OTHER half of the contract -- a
``cluster_aggregate``-style explicit NO-HARM-for-trees + LIFT-for-linear two-sided
assertion -- on the DEFAULT FE path: the univariate hermite/orth basis (the
``a__He2`` / ``a__L2`` columns that ``fe_univariate_basis_enable`` generates at the
default depth ``fe_max_steps=2``), which is structurally distinct from the pair
composites the sibling test covers.

The fixtures are EVEN-SYMMETRIC quadratic targets where the raw column is provably
uninformative on the symmetric domain (``y`` depends on ``x**2``, and raw ``x`` has
~zero linear correlation with ``x**2`` when ``x`` is centred): only the recovered
hermite term ``x__He2`` carries the signal linearly. A linear model on the raw-only
selection therefore floors near AUC 0.5, while the FE-augmented selection lifts it
toward 1.0; a tree reaches ~1.0 from the raws ALONE, so the engineered columns must
not drag it down.

Measured (seed 42, n=1500):
  QUAD  ``y=(x1^2>thr)``         LINEAR delta ~ +0.50, TREE delta ~ -0.006
  R2PAIR``y=(x1^2+x2^2>median)`` LINEAR delta ~ +0.49, TREE delta ~ -0.003
Linear-lift floors are set ~50% below the measured deltas; tree no-harm floor is the
contract's -0.02. NEVER xfail / weaken: a linear-lift regression means the FE path
lost downstream value, a tree-harm regression means it started polluting tree inputs
with junk columns -- both are prod bugs to fix, not assertions to relax.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.filters.mrmr import MRMR

sys.path.insert(0, os.path.dirname(__file__))
from tests.feature_selection._biz_val_synth import as_df  # noqa: F401  (kept for parity with sibling biz_val files)

from tests.feature_selection.conftest import is_fast_mode, fast_subset

lgb = pytest.importorskip("lightgbm")


# Every default-ON FE generator OFF -> MRMR selects RAW columns only, via the SAME
# relevance/redundancy machinery, so the delta is attributable purely to the FE step.
_RAW_ONLY = dict(
    fe_max_steps=0,
    # fe_hybrid_orth_enable flipped to default-ON 2026-06-21 (commit d76929d9); the
    # univariate/pair orth-basis stage opens on `_hybrid_on OR _univ_basis_on`, so the
    # raw-only baseline MUST opt out of the hybrid master switch too -- otherwise the
    # baseline silently receives `x__He2`, which (being the only linear carrier of the
    # even-symmetric signal) lifts lr_raw to ~1.0 and erases the FE-vs-raw lift contract.
    fe_hybrid_orth_enable=False,
    fe_univariate_basis_enable=False,
    fe_univariate_fourier_enable=False,
    fe_hinge_enable=False,
    fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False,
    fe_hybrid_orth_pair_enable=False,
    fe_auto_escalation_enable=False,
    fe_pair_prewarp_enable=False,
    fe_rung_schedule_enable=False,
    fe_stability_vote_enable=False,
    cluster_aggregate_enable=False,
    dcd_enable=False,
)
# The MAIN univariate hermite/orth basis path at the DEFAULT depth (fe_max_steps=2);
# the pair-cross orth basis is left on so a pair-quadratic can recover its terms.
# Heavy auxiliary generators OFF to keep each cell < 55s.
_FE_ON = dict(
    fe_univariate_basis_enable=True,
    fe_hybrid_orth_pair_enable=True,
    fe_hinge_enable=False,
    fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False,
    fe_auto_escalation_enable=False,
    fe_pair_prewarp_enable=False,
    fe_rung_schedule_enable=False,
    fe_stability_vote_enable=False,
    cluster_aggregate_enable=False,
    dcd_enable=False,
)


def _quad_target(seed: int = 42, n: int = 1500):
    """``y=(x1^2 > thr)`` -- a SINGLE even-symmetric quadratic. Raw ``x1`` has ~0
    linear correlation with ``x1^2`` on the centred domain, so only the recovered
    univariate hermite term ``x1__He2`` makes the signal linearly usable."""
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(0.0, 1.0, n) for i in range(1, 6)}
    y = (cols["x1"] ** 2 > 1.0).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _pair_quad_target(seed: int = 42, n: int = 1500):
    """``y=(x1^2 + x2^2 > median)`` -- an even-symmetric PAIR quadratic (radial
    threshold). BOTH univariate hermite terms ``x1__He2`` / ``x2__He2`` (or the
    pair-cross ``add(sqr(x1),sqr(x2))``) are the load-bearing linear carriers; raw
    x1/x2 are individually uninformative about the radius on the symmetric domain."""
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(0.0, 1.0, n) for i in range(1, 6)}
    r2 = cols["x1"] ** 2 + cols["x2"] ** 2
    y = (r2 > float(np.median(r2))).astype(np.int64)
    return pd.DataFrame(cols), pd.Series(y, name="y")


_FIXTURES = (
    ("quad_x1sq", _quad_target, 0.25),
    ("pair_quad_x1sq_x2sq", _pair_quad_target, 0.25),
)


def _split(df, y, frac: float = 0.7):
    """Leak-safe shuffle split: FE is fit on train, replayed on test."""
    n = len(df)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    k = int(n * frac)
    tr, te = idx[:k], idx[k:]
    return (df.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True))


def _select_transform(kwargs, df_tr, y_tr, df_te):
    fs = MRMR(verbose=0, random_seed=42, **kwargs)
    fs.fit(df_tr, y_tr)
    Xtr = np.nan_to_num(np.asarray(fs.transform(df_tr), float), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(np.asarray(fs.transform(df_te), float), nan=0.0, posinf=0.0, neginf=0.0)
    return Xtr, Xte, list(fs.get_feature_names_out())


def _logreg_auc(Xtr, Xte, y_tr, y_te):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))
    m.fit(Xtr, y_tr.values)
    return float(roc_auc_score(y_te.values, m.predict_proba(Xte)[:, 1]))


def _lgbm_auc(Xtr, Xte, y_tr, y_te):
    m = lgb.LGBMClassifier(n_estimators=80, random_state=42, verbose=-1)
    m.fit(np.asarray(Xtr), y_tr.values)
    return float(roc_auc_score(y_te.values, m.predict_proba(np.asarray(Xte))[:, 1]))


def _cross_family_deltas(mkdata, n):
    """Return per-family held-out AUC for FE-on vs raw-only selections, plus the
    recovered FE feature names. AUCs: (lr_fe, lr_raw, gb_fe, gb_raw, names_fe)."""
    df, y = mkdata(n=n)
    df_tr, y_tr, df_te, y_te = _split(df, y)
    Xtr_fe, Xte_fe, n_fe = _select_transform(_FE_ON, df_tr, y_tr, df_te)
    Xtr_raw, Xte_raw, n_raw = _select_transform(_RAW_ONLY, df_tr, y_tr, df_te)
    assert Xtr_fe.shape[1] > 0, f"FE-on selection empty -> 0 features downstream; names={n_fe}"
    assert Xtr_raw.shape[1] > 0, f"raw-only selection empty -> 0 features downstream; names={n_raw}"
    lr_fe = _logreg_auc(Xtr_fe, Xte_fe, y_tr, y_te)
    lr_raw = _logreg_auc(Xtr_raw, Xte_raw, y_tr, y_te)
    gb_fe = _lgbm_auc(Xtr_fe, Xte_fe, y_tr, y_te)
    gb_raw = _lgbm_auc(Xtr_raw, Xte_raw, y_tr, y_te)
    return lr_fe, lr_raw, gb_fe, gb_raw, n_fe


def _has_engineered(names):
    return [n for n in names if ("(" in n) or ("__" in n)]


_TREE_NO_HARM_TOL = 0.02  # the contract's tree no-harm floor: auc_fe >= auc_raw - 0.02.


def _assert_two_sided_contract(label, lr_fe, lr_raw, gb_fe, gb_raw, names_fe, lin_floor):
    eng = _has_engineered(names_fe)
    assert eng, f"[{label}] FE-on recovered NO engineered hermite/orth column: {names_fe}"
    lin_delta = lr_fe - lr_raw
    tree_delta = gb_fe - gb_raw
    assert lin_delta >= lin_floor, (
        f"[{label}] LINEAR LIFT lost: the engineered hermite/orth basis no longer makes the "
        f"even-symmetric signal linearly usable. logreg fe={lr_fe:.4f} raw={lr_raw:.4f} "
        f"delta={lin_delta:+.4f} (want >= {lin_floor:+.4f}); FE names={names_fe}."
    )
    assert tree_delta >= -_TREE_NO_HARM_TOL, (
        f"[{label}] TREE NO-HARM violated: the engineered columns polluted the tree inputs and "
        f"diluted its split search. lgbm fe={gb_fe:.4f} raw={gb_raw:.4f} delta={tree_delta:+.4f} "
        f"(want >= -{_TREE_NO_HARM_TOL:.2f}); FE names={names_fe}."
    )


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize("label,mkdata,lin_floor", _FIXTURES)
def test_fe_two_sided_contract_lift_linear_no_harm_tree(label, mkdata, lin_floor):
    """Two-sided cross-model-family contract on the MAIN hermite/orth FE path:
    (i) LINEAR LIFT -- LogisticRegression AUC on the FE-augmented selection beats the
        raw-only selection by a calibrated margin (the hermite term is the only linear
        carrier of the even-symmetric signal), and
    (ii) TREE NO-HARM -- LGBMClassifier AUC on the FE-augmented selection stays within
        0.02 of the raw-only selection (the engineered columns do not dilute the tree's
        native split search). This converts "complementarity, not universal lift" into
        an executable regression sensor."""
    lr_fe, lr_raw, gb_fe, gb_raw, names_fe = _cross_family_deltas(mkdata, n=1500)
    _assert_two_sided_contract(label, lr_fe, lr_raw, gb_fe, gb_raw, names_fe, lin_floor)


@pytest.mark.timeout(120)
def test_fe_two_sided_contract_fast_representative():
    """Fast representative for ``MLFRAME_FAST=1``: runs ONE fixture (``fast_subset``)
    at a reduced ``n`` so the whole cross-family contract path is exercised under the
    fast gate without the full @slow parametrize. Same two-sided assertions, relaxed
    linear floor to absorb the smaller-sample noise."""
    n = 900 if is_fast_mode() else 1500
    for label, mkdata, _lin_floor in fast_subset(_FIXTURES, n=1):
        lr_fe, lr_raw, gb_fe, gb_raw, names_fe = _cross_family_deltas(mkdata, n=n)
        _assert_two_sided_contract(label, lr_fe, lr_raw, gb_fe, gb_raw, names_fe, lin_floor=0.20)
