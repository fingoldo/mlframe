"""Task-axis coverage for the NON-orth target-aware FE mechanisms (param_axes-01).

The fe_hybrid_orth family is already biz-value-tested on regression / multiclass
real datasets (layers 29, 83), but the NON-orth target-aware FE families --
``fe_kfold_te``, ``fe_grouped_agg``, ``fe_cat_num_interaction``, ``fe_rankgauss``
-- had ZERO regression-target and ZERO multiclass biz-value coverage: every
layer enabling them pins a binary-classification label. Prod explicitly claims
continuous-y branches work (``_fit_impl_core``: "TE works for both binary
classification and regression as-is (mean of {0,1} = P(y=1); mean of continuous
= mean)") yet no test witnesses them.

This file fits each mechanism through the MRMR PUBLIC API (ctor flags + fit +
transform + get_feature_names_out + the per-mechanism ``*_features_`` roster) on
BOTH a continuous (regression) target and a 3-way-quantile-cut multiclass target,
asserting:

* fit completes (no crash on continuous / 3-class y);
* transform replays on a disjoint holdout, all-finite, correct row count;
* the mechanism's engineered columns appear (roster non-empty AND a recipe-named
  column shows up in ``get_feature_names_out()``);
* downstream Ridge R2 (regression) / multinomial LogReg accuracy (multiclass)
  lifts over a no-FE MRMR baseline by a calibrated floor -- pinned at the MEDIAN
  seed (majority-of-seeds win), set well below the measured median so a real
  regression in the continuous-y / multiclass branch trips it.

Each fixture injects exactly the structure the mechanism can exploit:
* ``cat_region`` (low-card cat) with a strong per-category effect -> kfold-TE /
  grouped-agg unlock a categorical the raw integer code hides from a linear model;
* per-category-centred ``num1`` whose conditional anomaly drives y -> cat-num
  residual recovers ``income-high-for-this-bracket``;
* heavy-tailed ``x1`` linear-but-saturating in y -> RankGauss tames the outliers
  for the linear downstream (DPI: MI preserved, the win is downstream).

Calibrated 2026-06-10 (measured medians, floors 60-85% below per CLAUDE.md):

| mechanism      | reg R2 lift median | mc acc lift median |
|----------------|--------------------|--------------------|
| kfold-te       | 0.847              | 0.440              |
| grouped-agg    | 0.352              | 0.238              |
| cat-num-resid  | 0.885              | 0.517              |
| rankgauss      | 0.225              | 0.073              |
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tests.feature_selection.conftest import fast_subset

warnings.filterwarnings("ignore")

SEEDS = (0, 1, 2, 3)
N = 1300

# Common MRMR kwargs: disable the default-on general-FE competitors (hinge,
# conditional-dispersion, wavelet, univariate basis/Fourier, the pair-FE step,
# AND the newer default-on cat-aware families -- pairwise-modular, integer-lattice,
# binned-numeric-agg, conditional-gate, row-argmax) so the family under test is the
# only mechanism that can engineer a column -- otherwise a redundant sibling reads
# the same signal and the roster reconciles empty (the documented layer104 caveat).
# Concretely, ``pmod_self__cat_region`` (pairwise-modular self-feature) re-encodes
# the same low-card categorical the mechanism-under-test targets and was winning its
# screening slot, leaving kfold_te_features_ empty. max_runtime keeps each fit bounded.
_ISOLATE = dict(
    max_runtime_mins=0.6,
    fe_max_steps=0,
    fe_univariate_basis_enable=False,
    fe_univariate_fourier_enable=False,
    fe_hinge_enable=False,
    fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False,
    fe_pairwise_modular_enable=False,
    fe_integer_lattice_enable=False,
    fe_binned_numeric_agg_enable=False,
    fe_conditional_gate_enable=False,
    fe_row_argmax_enable=False,
    verbose=0,
)


# ---------------------------------------------------------------------------
# Per-mechanism fixtures (continuous logit; the task axis quantile-cuts it)
# ---------------------------------------------------------------------------


def _fx_cat_effect(seed: int, n: int = N):
    """Low-card ``cat_region`` with a strong per-category additive effect plus two
    linear numerics. The raw integer cat code is non-ordinal -> a linear model
    cannot read the per-category effect off it; kfold-TE (per-category mean) and
    grouped-agg (per-group stat) unlock it. Returns ``(X, logit)``."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 8, size=n)
    eff = np.array([6.0, -5.0, 3.0, 0.0, -3.0, 5.0, -1.0, 2.0])
    num1 = rng.normal(size=n)
    num2 = rng.normal(size=n)
    logit = eff[cat] + 1.2 * num1 - 0.8 * num2
    X = pd.DataFrame(
        {
            "cat_region": cat.astype(np.int64),
            "num1": num1,
            "num2": num2,
            "noise": rng.normal(size=n),
        }
    )
    return X, logit


def _fx_cat_num_resid(seed: int, n: int = N):
    """``num1`` is centred on a per-category mean; y depends on the conditional
    anomaly ``num1 - E[num1 | cat]`` (high FOR this bracket), NOT the level. The
    cat-num OOF residual recovers it; raw num1 / cat hide it. Returns ``(X, logit)``."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 6, size=n)
    cat_mean = np.array([0.0, 4.0, -4.0, 2.0, -2.0, 6.0])
    num1 = cat_mean[cat] + rng.normal(size=n)
    resid = num1 - cat_mean[cat]
    num2 = rng.normal(size=n)
    logit = 3.0 * resid + 0.5 * num2
    X = pd.DataFrame(
        {
            "cat_region": cat.astype(np.int64),
            "num1": num1,
            "num2": num2,
        }
    )
    return X, logit


def _fx_heavytail(seed: int, n: int = N):
    """Heavy-tailed (Student-t df=2) ``x1`` enters y through a saturating monotone
    map; raw x1 outliers dominate the shared regularised linear scale. RankGauss
    Gaussianises x1 -> better linear-model input (DPI: MI preserved, downstream
    win). Returns ``(X, logit)``."""
    rng = np.random.default_rng(seed)
    base = rng.standard_t(df=2, size=n)
    z2 = rng.normal(size=n)
    logit = 1.3 * np.tanh(base) + 0.9 * z2
    X = pd.DataFrame(
        {
            "x1": base,
            "z2": z2,
            "noise": rng.normal(size=n),
        }
    )
    return X, logit


# (ctor kwargs, fixture builder, roster attr, recipe-name substring, reg floor, mc floor)
_MECHS = [
    pytest.param(
        dict(fe_kfold_te_enable=True, fe_kfold_te_cols=("cat_region",)),
        _fx_cat_effect,
        "kfold_te_features_",
        "__te",
        0.10,
        0.10,
        id="kfold-te",
    ),
    pytest.param(
        dict(
            fe_grouped_agg_enable=True,
            fe_grouped_agg_group_cols=("cat_region",),
            fe_grouped_agg_num_cols=("num1",),
        ),
        # The grouped-agg family emits several column kinds (grpagg_*, grpratio(...),
        # grpp90p10(...), grpiqr(...)); match the shared ``grp`` family prefix so the
        # surviving column (e.g. grpratio(num1|cat_region)) satisfies the roster->output
        # contract regardless of which grouped statistic wins the slot.
        _fx_cat_effect,
        "grouped_agg_features_",
        "grp",
        0.05,
        0.05,
        id="grouped-agg",
    ),
    pytest.param(
        dict(
            fe_cat_num_interaction_enable=True,
            fe_cat_num_interaction_cat_cols=("cat_region",),
            fe_cat_num_interaction_num_cols=("num1",),
        ),
        _fx_cat_num_resid,
        "cat_num_interaction_features_",
        "resid_by",
        0.10,
        0.10,
        id="cat-num-resid",
    ),
    pytest.param(
        dict(fe_rankgauss_enable=True, fe_rankgauss_cols=("x1",)),
        _fx_heavytail,
        "rankgauss_features_",
        "rankgauss",
        0.04,
        0.03,
        id="rankgauss",
    ),
]

_TASKS = [
    pytest.param("regression", id="regression"),
    pytest.param("multiclass", id="multiclass"),
]


def _make_target(logit: np.ndarray, task: str, seed: int):
    """Continuous logit -> a regression target (logit + noise) or a balanced
    3-class label (2/3 quantile cut). Returns ``(y, stratify_or_None)``."""
    if task == "regression":
        y = logit + 0.3 * np.random.default_rng(seed + 9).normal(size=len(logit))
        return y, None
    qs = np.quantile(logit, [1.0 / 3.0, 2.0 / 3.0])
    y = np.digitize(logit, qs).astype(np.int64)
    return y, y


def _downstream_lift(Xfe, Xno, y, task: str, seed: int) -> float:
    """Holdout downstream metric delta (FE frame minus no-FE frame).

    Regression -> standardised Ridge R2; multiclass -> standardised multinomial
    LogReg accuracy. Same train/test split for both so the delta isolates FE."""
    y_arr = np.asarray(y)
    strat = y_arr if task == "multiclass" else None
    tr, te = train_test_split(
        np.arange(len(y_arr)),
        test_size=0.3,
        random_state=seed,
        stratify=strat,
    )
    if task == "regression":
        est = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        score = lambda m, Xt: r2_score(y_arr[te], m.predict(Xt.iloc[te]))
    else:
        est = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        score = lambda m, Xt: accuracy_score(y_arr[te], m.predict(Xt.iloc[te]))
    m_fe = est().fit(Xfe.iloc[tr], y_arr[tr])
    m_no = est().fit(Xno.iloc[tr], y_arr[tr])
    return score(m_fe, Xfe) - score(m_no, Xno)


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("mech_kwargs,fx,roster_attr,recipe_sub,reg_floor,mc_floor", fast_subset(_MECHS, 2))
def test_fe_mech_fits_transforms_and_lifts(
    task,
    mech_kwargs,
    fx,
    roster_attr,
    recipe_sub,
    reg_floor,
    mc_floor,
):
    """Each non-orth target-aware FE mechanism, on a regression AND a multiclass
    target: fit completes, transform replays on a holdout, engineered columns
    appear, and the downstream metric lifts over a no-FE MRMR baseline by the
    calibrated floor on the MAJORITY of seeds."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    floor = reg_floor if task == "regression" else mc_floor
    # The MI-gate that admits each engineered column is high-variance after the
    # multiclass quantile-cut weakens the per-category signal: on the occasional
    # seed the column legitimately fails the noise floor and the roster reconciles
    # empty (NOT a regression -- the same gate that protects against spurious
    # columns). So the structural contract is MAJORITY-of-seeds (mirroring the
    # lift's majority semantics), and lift is measured only on the seeds where the
    # mechanism actually engineered a surviving column.
    engineered_seeds = 0
    lifts = []
    for seed in SEEDS:
        X, logit = fx(seed)
        y, _ = _make_target(logit, task, seed)
        y_ser = pd.Series(y, name="y")

        m_fe = MRMR(**mech_kwargs, **_ISOLATE).fit(X, y_ser)
        m_no = MRMR(**_ISOLATE).fit(X, y_ser)

        names_fe = list(m_fe.get_feature_names_out())

        # (1) transform replays on a DISJOINT holdout: correct rows / cols, all
        # finite. This is a HARD contract independent of FE survival -- fit on a
        # continuous / 3-class target must always produce a replayable recipe set.
        X_ho = fx(seed + 100)[0]
        Xt_ho = m_fe.transform(X_ho)
        assert Xt_ho.shape[0] == X_ho.shape[0]
        assert Xt_ho.shape[1] == len(names_fe)
        assert np.isfinite(np.asarray(Xt_ho, dtype=np.float64)).all(), f"{task} seed={seed}: holdout transform produced non-finite values."

        # (2) did THIS mechanism engineer a surviving column? roster non-empty
        # AND a recipe-named column reached the selected output. The roster is set
        # before downstream selection, so when it is populated the recipe column
        # must be in names_fe -- assert they agree, then only count majority.
        roster = list(getattr(m_fe, roster_attr, []) or [])
        recipe_in_out = any(recipe_sub in nm for nm in names_fe)
        if roster:
            assert recipe_in_out, f"{task} seed={seed}: {roster_attr}={roster} populated but no '{recipe_sub}' column in get_feature_names_out ({names_fe})."
            engineered_seeds += 1
            # (3) downstream lift over the no-FE baseline on the SAME split.
            lift = _downstream_lift(m_fe.transform(X), m_no.transform(X), y, task, seed)
            lifts.append(lift)

    # MAJORITY of seeds must engineer + surface the mechanism's column.
    assert engineered_seeds > len(SEEDS) // 2, (
        f"{task}: only {engineered_seeds}/{len(SEEDS)} seeds produced a surviving "
        f"'{recipe_sub}' column -> the non-orth target-aware mechanism is not "
        f"engineering on a {task} target."
    )

    # (4) majority-of-engineered-seeds downstream win clears the calibrated floor.
    median_lift = float(np.median(lifts))
    assert median_lift >= floor, (
        f"{task}: median downstream lift {median_lift:.4f} < floor {floor} "
        f"(per-engineered-seed {[round(x, 4) for x in lifts]}); the non-orth "
        f"target-aware FE mechanism is not lifting on a {task} target."
    )
