"""Default-pipeline null FWER + all-FE-on x production-default redundancy.

Two gaps the per-mechanism FE suite never closes:

(a) ``gaps_fe_masking-01`` -- there is no all-noise end-to-end sensor on the
    SHIPPED-DEFAULT ``MRMR()`` pipeline. ``test_biz_value_mrmr_quality_metrics``
    fits a near-default MRMR on a pure-noise frame but at n=800, asserts only the
    catastrophic ceiling ``n_selected < 10`` (escapes via the fallback path), and
    never inspects ``_engineered_recipes_``. The joint family-wise error rate of
    the multi-family ``fe_max_steps=2`` default -- including whether it MANUFACTURES
    engineered features out of pure noise -- is untested. This file fits plain
    ``MRMR(verbose=0, random_seed=seed)`` (NO other kwargs: the point is the default
    users actually get) on a pure-noise frame with a few low-card int columns to
    wake the cat / count-freq paths, for both a binary-clf and a continuous-reg
    target, and pins a calibrated per-seed ceiling on ``len(get_feature_names_out())``
    AND on ``len(_engineered_recipes_)``. A measured residual FWER -- the default
    reg-target pipeline manufacturing a Haar-wavelet recipe from a pure-noise column
    -- is pinned as a strict-False xfail carrying the measured rate, so a future
    tightening flips it to xpass and a regression that worsens it still fails the
    ceiling.

(b) ``param_axes-05`` -- every all-FE-on kitchen-sink / state-of-the-union contract
    (layers 35, 64, 70, 101) pins ``dcd_enable=False, cluster_aggregate_enable=False,
    build_friend_graph=False``, so the production-default redundancy mechanisms
    (``dcd_enable=True``, ``cluster_aggregate_enable=True`` since 2026-05-30) are NEVER
    validated end-to-end with FE on. This file takes a kitchen-sink FE-on config but
    leaves dcd / cluster_aggregate at their constructor defaults (ON), pins the
    resulting downstream AUC against its own measured floor, AND asserts the
    prod-default selection is within a few hundredths of the isolated-redundancy
    (dcd/cluster OFF) run -- so a future change where DCD pruning or cluster-aggregate
    swallows an engineered column a downstream model depends on is caught.

Calibration (measured once, CPU, store-python 3.14):
  (a) clf null n=2500/p=10 (13 cols), seeds 0..7: max nsel=2, neng=0 every seed.
      reg null same config: max nsel=5 (seed=3), neng=1 on 1/12 seeds (Haar wavelet
      on noise col n1 -- the pinned residual FWER). fast rep n=1200/p=6: nsel=1, neng=0.
  (b) signal+FE n=2000: AUC_default == AUC_isolated to 1e-4 at seeds 42 (0.8835) and
      7 (0.8634); identical feature counts. Floors set ~0.05 below the measured min.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode, fast_subset


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _build_null_frame(n: int, p: int, seed: int):
    """Pure-noise frame: ``p`` standard-normal columns + 3 low-card int columns
    (to wake the categorical / count-frequency FE paths) and two random targets.

    The binary ``y_clf`` is drawn BEFORE the continuous ``y_reg`` so the reg
    target consumes the post-clf RNG state -- this exact draw order is what the
    pinned residual-FWER seed (seed=3, reg) was calibrated on.
    """
    rng = np.random.default_rng(int(seed))
    cols = {f"n{i}": rng.standard_normal(n) for i in range(p)}
    for j in range(3):
        cols[f"c{j}"] = rng.integers(0, 5, n).astype(np.int64)
    X = pd.DataFrame(cols)
    y_clf = pd.Series(rng.integers(0, 2, n).astype(np.int64), name="y")
    y_reg = pd.Series(rng.standard_normal(n), name="y")
    return X, y_clf, y_reg


def _fit_default(X, y, seed: int):
    """Fit MRMR with the SHIPPED defaults only (verbose + seed). No FE / redundancy
    kwargs -- the whole point is the config a user gets from a bare ``MRMR()``."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=int(seed)).fit(X, y)


def _n_selected(sel) -> int:
    return len(list(sel.get_feature_names_out()))


def _n_engineered(sel) -> int:
    return len(getattr(sel, "_engineered_recipes_", []) or [])


def _build_signal_fe_frame(seed: int = 42, n: int = 2000):
    """Signal + FE-amenable fixture for part (b): a linear term, a quadratic term,
    a periodic term, and a skewed low-card categorical whose log-count drives y --
    so the MI-greedy unary/binary, orth-hermite, and count/frequency FE families all
    have genuine work. Four pure-noise columns round it out.
    """
    rng = np.random.default_rng(int(seed))
    x_num1 = rng.standard_normal(n)
    x_quad = rng.standard_normal(n)
    x_periodic = rng.uniform(-1.0, 1.0, size=n)
    n_users = 20
    user_ids = np.array([f"U_{i:02d}" for i in range(n_users)])
    w = np.linspace(1.0, 20.0, n_users)
    w = w / w.sum()
    cat_user = rng.choice(user_ids, size=n, p=w)
    counts = pd.Series(cat_user).value_counts()
    log_cnt = np.log1p(pd.Series(cat_user).map(counts).to_numpy().astype(float))
    log_cnt_c = log_cnt - log_cnt.mean()
    noise = rng.standard_normal((n, 4))
    logit = 0.6 * x_num1 + 1.8 * (x_quad**2 - 1.0) + 2.0 * np.sin(2.0 * np.pi * x_periodic) + 0.8 * log_cnt_c
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(np.int64)
    df = pd.DataFrame(
        {
            "x_num1": x_num1,
            "x_quad": x_quad,
            "x_periodic": x_periodic,
            "cat_user": cat_user,
            "n0": noise[:, 0],
            "n1": noise[:, 1],
            "n2": noise[:, 2],
            "n3": noise[:, 3],
        }
    )
    return df, pd.Series(y, name="y")


def _all_fe_kwargs():
    """Kitchen-sink FE-on config. Deliberately does NOT pin dcd_enable /
    cluster_aggregate_enable / build_friend_graph -- those keep their constructor
    defaults (dcd ON, cluster_aggregate ON, friend_graph OFF) so the all-FE x
    prod-default-redundancy interaction is exercised."""
    return dict(
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=6,
        fe_mi_greedy_include_unary=True,
        fe_mi_greedy_include_binary=True,
        fe_count_encoding_enable=True,
        fe_count_encoding_cols=("cat_user",),
        fe_frequency_encoding_enable=True,
        fe_frequency_encoding_cols=("cat_user",),
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=True,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=6,
    )


def _holdout_auc(sel, X_tr, y_tr, X_ho, y_ho) -> float:
    """Downstream LogisticRegression holdout AUC on the selected (numeric) columns.
    Non-numeric survivors (raw categorical object cols) are dropped -- the same
    no-FE-baseline convention the layer kitchen-sink contracts use."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    a = sel.transform(X_tr).select_dtypes("number")
    b = sel.transform(X_ho).select_dtypes("number")
    if a.shape[1] == 0:
        return float("nan")
    lr = LogisticRegression(max_iter=400).fit(a, y_tr)
    return float(roc_auc_score(y_ho, lr.predict_proba(b)[:, 1]))


# Calibration constants (measured once, see module docstring).
# Per-seed ceiling on selected features from a pure-noise frame: a catastrophic-FWER
# sensor. Measured max across calibration was 5/13 (reg seed=3) and 6/15 (n=4000);
# half the total-column count passes everywhere yet fails the "all noise surfaces"
# regression. Computed per-frame from the real column count.
_NOISE_SELECT_FRACTION_CEIL = 0.5


# ---------------------------------------------------------------------------
# (a) DEFAULT-pipeline null FWER
# ---------------------------------------------------------------------------


def _null_fwer_seeds_slow():
    return [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.slow
@pytest.mark.parametrize("seed", _null_fwer_seeds_slow())
@pytest.mark.parametrize("target", ["clf", "reg"])
def test_default_pipeline_null_selection_is_bounded(seed, target):
    """SHIPPED-DEFAULT MRMR on a pure-noise frame must not surface more than a
    calibrated fraction of the columns -- the joint family-wise false-positive rate
    the per-mechanism noise tests all assume but none provides end-to-end.

    Bounded ceiling (passing sensor): ``nsel <= ceil(0.5 * total_cols)``. A
    regression that admits every noise column trips this. Engineered-recipe
    manufacturing from pure noise is asserted separately (clf strict-zero here;
    the measured reg residual is the xfail below) so the two failure modes stay
    distinguishable.
    """
    n, p = 2500, 10
    X, y_clf, y_reg = _build_null_frame(n, p, seed)
    y = y_clf if target == "clf" else y_reg
    sel = _fit_default(X, y, seed)

    total_cols = X.shape[1]
    ceil_sel = math.ceil(_NOISE_SELECT_FRACTION_CEIL * total_cols)
    nsel = _n_selected(sel)
    assert nsel <= ceil_sel, (
        f"default null FWER: {target} seed={seed} selected {nsel}/{total_cols} "
        f"pure-noise features (ceiling {ceil_sel}); selection should stay bounded "
        f"on an all-noise frame. names={list(sel.get_feature_names_out())}"
    )

    if target == "clf":
        # Calibration: the binary-clf default never manufactures an engineered
        # recipe from pure noise across seeds 0..7. (The reg target's measured
        # residual is pinned in the xfail test below.)
        neng = _n_engineered(sel)
        assert neng == 0, (
            f"default null FWER: clf seed={seed} manufactured {neng} engineered "
            f"recipe(s) from PURE NOISE: "
            f"{[r.name for r in getattr(sel, '_engineered_recipes_', [])]}"
        )


def test_default_pipeline_null_selection_is_bounded_fast():
    """Fast representative of the null-FWER sensor: a single seed at a smaller
    frame so every fast-mode invocation still exercises the default all-noise path.
    Measured: nsel=1, neng=0 at n=1200/p=6."""
    seed = 0
    n, p = (1200, 6) if is_fast_mode() else (2500, 10)
    X, y_clf, _ = _build_null_frame(n, p, seed)
    sel = _fit_default(X, y_clf, seed)

    total_cols = X.shape[1]
    ceil_sel = math.ceil(_NOISE_SELECT_FRACTION_CEIL * total_cols)
    nsel = _n_selected(sel)
    assert nsel <= ceil_sel, f"default null FWER (fast): selected {nsel}/{total_cols} noise features (ceiling {ceil_sel})"
    assert _n_engineered(sel) == 0, "default null FWER (fast): manufactured engineered recipe(s) from pure noise"


@pytest.mark.slow
def test_default_pipeline_null_manufactures_no_engineered_recipe_from_noise():
    """The shipped default should NEVER synthesize an engineered feature from a
    frame that is pure noise -- doing so is a family-wise false positive at the FE
    layer.

    This was previously xfail(strict=False): the reg target at seed=3 manufactured a
    residual recipe from a pure-noise column. The 2026-06-22 binagg redundancy-gate
    tightening (robust mean+2*std permutation-null ceiling, replacing the unstable
    raw max-of-15) collapses that residual to zero, so the contract is now a hard
    assertion. A regression that re-admits a noise recipe trips it directly."""
    n, p, seed = 2500, 10, 3
    X, _y_clf, y_reg = _build_null_frame(n, p, seed)
    sel = _fit_default(X, y_reg, seed)
    neng = _n_engineered(sel)
    recipe_names = [r.name for r in getattr(sel, "_engineered_recipes_", [])]
    assert neng == 0, f"default pipeline manufactured {neng} engineered recipe(s) from pure noise: {recipe_names}"


# ---------------------------------------------------------------------------
# (b) all-FE-on x production-default redundancy
# ---------------------------------------------------------------------------


def _split(df, y, n_tr_frac: float = 0.7):
    n_tr = int(len(df) * n_tr_frac)
    return (df.iloc[:n_tr], y.iloc[:n_tr], df.iloc[n_tr:], y.iloc[n_tr:])


# Measured all-FE-on prod-default holdout AUC: 0.8835 (seed 42), 0.8634 (seed 7).
# Floor ~0.05 below the measured minimum; within-band tolerance vs the isolated
# (dcd/cluster OFF) run -- measured delta was exactly 0.0000 at both seeds.
_ALL_FE_PRODDEFAULT_AUC_FLOOR = 0.81
_ALL_FE_VS_ISOLATED_BAND = 0.03


@pytest.mark.slow
@pytest.mark.parametrize("seed", [42, 7])
def test_all_fe_on_prod_default_redundancy_holds_auc(seed):
    """Kitchen-sink FE-on MRMR with the PRODUCTION-DEFAULT redundancy mechanisms
    (dcd_enable / cluster_aggregate_enable left at their ON constructor defaults)
    must clear a downstream-AUC floor AND stay within a few hundredths of the
    isolated-redundancy (dcd/cluster OFF) run.

    The within-band check is the param_axes-05 sensor: if DCD pruning or
    cluster-aggregate swallows an engineered column the downstream model depends on,
    the prod-default AUC drops below the isolated AUC by more than the band and the
    test fails -- the interaction the all-FE kitchen-sink layers (35/64/70/101) never
    witness because they pin both mechanisms off.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _build_signal_fe_frame(seed=seed, n=2000)
    X_tr, y_tr, X_ho, y_ho = _split(df, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_default = MRMR(verbose=0, random_seed=seed, **_all_fe_kwargs()).fit(X_tr, y_tr)
        sel_isolated = MRMR(
            verbose=0,
            random_seed=seed,
            dcd_enable=False,
            cluster_aggregate_enable=False,
            build_friend_graph=False,
            **_all_fe_kwargs(),
        ).fit(X_tr, y_tr)

    auc_default = _holdout_auc(sel_default, X_tr, y_tr, X_ho, y_ho)
    auc_isolated = _holdout_auc(sel_isolated, X_tr, y_tr, X_ho, y_ho)

    assert auc_default >= _ALL_FE_PRODDEFAULT_AUC_FLOOR, (
        f"all-FE-on prod-default downstream AUC {auc_default:.4f} below floor {_ALL_FE_PRODDEFAULT_AUC_FLOOR} at seed={seed}"
    )
    assert auc_default >= auc_isolated - _ALL_FE_VS_ISOLATED_BAND, (
        f"prod-default redundancy (dcd/cluster ON) DEGRADED the all-FE selection vs "
        f"the isolated config at seed={seed}: AUC_default={auc_default:.4f} < "
        f"AUC_isolated={auc_isolated:.4f} - band({_ALL_FE_VS_ISOLATED_BAND}). A "
        f"redundancy mechanism likely swallowed a useful engineered column."
    )


def test_all_fe_on_prod_default_redundancy_holds_auc_fast():
    """Fast representative of part (b): one seed, smaller n, so fast-mode still
    exercises the all-FE x prod-default-redundancy path end-to-end."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    seed = 42
    n = 1200 if is_fast_mode() else 2000
    df, y = _build_signal_fe_frame(seed=seed, n=n)
    X_tr, y_tr, X_ho, y_ho = _split(df, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_default = MRMR(verbose=0, random_seed=seed, **_all_fe_kwargs()).fit(X_tr, y_tr)
        sel_isolated = MRMR(
            verbose=0,
            random_seed=seed,
            dcd_enable=False,
            cluster_aggregate_enable=False,
            build_friend_graph=False,
            **_all_fe_kwargs(),
        ).fit(X_tr, y_tr)

    auc_default = _holdout_auc(sel_default, X_tr, y_tr, X_ho, y_ho)
    auc_isolated = _holdout_auc(sel_isolated, X_tr, y_tr, X_ho, y_ho)
    # Smaller-n floor is looser; the within-band contract is the real sensor.
    assert auc_default >= 0.70, f"all-FE-on prod-default downstream AUC {auc_default:.4f} unexpectedly low (fast)"
    assert auc_default >= auc_isolated - _ALL_FE_VS_ISOLATED_BAND, (
        f"prod-default redundancy degraded the all-FE selection vs isolated (fast): AUC_default={auc_default:.4f} AUC_isolated={auc_isolated:.4f}"
    )
