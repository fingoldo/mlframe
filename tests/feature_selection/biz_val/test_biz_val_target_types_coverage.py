"""biz_val coverage: FS effectiveness across ALL supported mlframe target types.

Q2 coverage dimension. The existing FS biz_val suite locks in signal recovery /
noise rejection for BINARY and REGRESSION (+ some MULTICLASS). This file extends
the same behavioral floor to the target types that were UNTESTED:

    * MULTICLASS (>2 exclusive classes)
    * MULTILABEL (independent binary outputs, 2D y)
    * MULTI-OUTPUT / MULTI-TARGET REGRESSION (2D continuous y)
    * COUNT / POISSON (y = rng.poisson(lambda(x)))
    * ORDINAL (ordered discrete levels)

mlframe ``TargetTypes`` (src/mlframe/training/configs.py) enumerates REGRESSION,
BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION,
LEARNING_TO_RANK, QUANTILE_REGRESSION, MULTI_TARGET_REGRESSION. COUNT/Poisson and
ORDINAL are not first-class enum members -- they ride on REGRESSION-shaped y -- so
they are exercised here as the data SHAPES a user feeds, validating that the
selector's information-theoretic relevance still recovers the generating columns.

Selector x target-type SUPPORT + EFFECTIVENESS matrix (measured here):

    target type     | MRMR (filter)          | RFECV (wrapper)
    ----------------|------------------------|----------------------------------
    multiclass>2    | recovers {x0,x1}, 0 nz | accepts; argmax keeps all (weak)
    count/Poisson   | recovers {x0,x1}, 0 nz | accepts (PoissonRegressor); weak
    ordinal         | recovers {x0,x1}, 0 nz | accepts (Ridge); recovers {x0,x1}
    multilabel (2D) | recovers {x0,x1}       | recovers {x0,x1} (default multioutput_strategy='union')
    multioutput(2D) | recovers {x0,x1}       | recovers {x0,x1} (default multioutput_strategy='union')

MEASURED floor (5 seeds, no-FE fast preset, n=700, 2 signal + 6 noise):
    every target type recovers BOTH signal columns on 5/5 seeds (sig=2/2);
    single-target types leak 0 noise; 2D targets leak <=1 noise/seed.
Floors below are pinned ~10-15% under those measured values (majority-of-seeds).

Multi-output (2D y) FS via RFECV works out of the box: ``multioutput_strategy``
defaults to ``'union'`` and fits one single-target
RFECV per output column and aggregates the per-column ``support_`` (OR / AND).
Set ``multioutput_strategy=None`` to opt OUT and restore the historical clear
``NotImplementedError`` so a caller who wants the hard failure on 2D y gets an
actionable message rather than a cryptic sklearn crash.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Tiny signal+noise fixtures: 2 true signal cols (x0, x1) + 6 pure-noise.
# Target is built from x0+x1 (or per-label / per-output from x0, x1) so the
# generating set is exactly {0, 1} for every target type.
# ---------------------------------------------------------------------------

N = 700
N_SIGNAL = 2
N_NOISE = 6
SIGNAL = {0, 1}


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _design(seed: int):
    """Build a base (rng, df, x_signal) design with N_SIGNAL signal cols + N_NOISE noise cols."""
    rng = np.random.default_rng(seed)
    x_sig = rng.normal(size=(N, N_SIGNAL))
    x_noise = rng.normal(size=(N, N_NOISE))
    X = np.column_stack([x_sig, x_noise])
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_SIGNAL + N_NOISE)])
    return rng, df, x_sig


def make_target(seed: int, kind: str):
    """Return ``(df, y)`` for a given target ``kind``. Signal lives in x0, x1."""
    rng, df, xs = _design(seed)
    score = xs[:, 0] + xs[:, 1]
    if kind == "multiclass":
        y = pd.Series(np.digitize(score, np.quantile(score, [0.33, 0.66])), name="y")
    elif kind == "count":
        lam = np.exp(0.7 * xs[:, 0] + 0.6 * xs[:, 1])
        y = pd.Series(rng.poisson(lam), name="y")
    elif kind == "ordinal":
        y = pd.Series(np.digitize(score, np.quantile(score, [0.2, 0.4, 0.6, 0.8])), name="y")
    elif kind == "multilabel":
        l0 = (xs[:, 0] + 0.3 * rng.normal(size=N) > 0).astype(int)
        l1 = (xs[:, 1] + 0.3 * rng.normal(size=N) > 0).astype(int)
        y = pd.DataFrame(np.column_stack([l0, l1]), columns=["l0", "l1"])
    elif kind == "multioutput":
        o0 = xs[:, 0] + 0.3 * rng.normal(size=N)
        o1 = xs[:, 1] + 0.3 * rng.normal(size=N)
        y = pd.DataFrame(np.column_stack([o0, o1]), columns=["o0", "o1"])
    elif kind == "ltr":
        # Learning-to-rank target: graded relevance labels (0..4) rising with x0+x1. MRMR's MI
        # estimator treats the relevance grade as the target (it does not consume query groups),
        # so signal recovery is governed by the graded label exactly as for an ordinal target.
        y = pd.Series(np.digitize(score, np.quantile(score, [0.2, 0.4, 0.6, 0.8])).astype(int), name="relevance")
    else:  # pragma: no cover - guard
        raise ValueError(kind)
    return df, y


def _make_mrmr(seed: int):
    """Lightest deterministic no-FE MRMR (mirrors conftest.make_fast_mrmr base).

    FE is OFF so the SELECTED set is RAW columns -- with FE on, MRMR would
    engineer e.g. ``mul(exp(x0),exp(x1))`` and leave ``support_`` (raw) empty
    even though it captured the signal. The no-FE preset is the correct lens for
    a raw signal-recovery / noise-rejection assertion.

    ``strict_groups=False`` opts back into the legacy warn-only group-naive fallback (its default
    flipped to True at finding #20); this helper is used by tests that deliberately exercise
    ``groups=`` on the LtR warn-only contract.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=seed,
        strict_groups=False,
    )


def _selected(selector) -> set:
    """Return the fitted selector's support_ as a set of raw column indices."""
    sup = np.asarray(selector.support_)
    if sup.dtype == bool:
        return set(np.where(sup)[0].tolist())
    return set(int(i) for i in sup.tolist())


# ===========================================================================
# MRMR -- supports every target type (single-target AND 2D). Behavioral floor:
# recover BOTH signal cols, reject noise.
# ===========================================================================

_SINGLE_TARGET = ["multiclass", "count", "ordinal", "ltr"]
_MULTI_TARGET = ["multilabel", "multioutput"]
_ALL_KINDS = _SINGLE_TARGET + _MULTI_TARGET


def _mrmr_recovery(kind: str, seed: int):
    """Fit MRMR on a ``kind`` target and return (recovered signal count, noise count)."""
    df, y = make_target(seed, kind)
    sel = _make_mrmr(seed)
    sel.fit(df, y)
    sup = _selected(sel)
    recovered = len(SIGNAL & sup)
    noise = len([i for i in sup if i >= N_SIGNAL])
    return recovered, noise, sup


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_biz_val_mrmr_recovers_signal_all_target_types(kind):
    """MRMR recovers BOTH generating columns {x0, x1} on the MAJORITY of seeds
    for every supported target type. Measured: 5/5 seeds recover 2/2; floor is
    >=2 recovered on >=4/5 seeds (~10% under measured)."""
    recs = [_mrmr_recovery(kind, s)[0] for s in range(5)]
    n_full = sum(r >= N_SIGNAL for r in recs)
    assert n_full >= 4, f"MRMR on {kind}: expected >=2/2 signal recovered on >=4/5 seeds; per-seed recovered={recs}"


@pytest.mark.parametrize("kind", _SINGLE_TARGET)
def test_biz_val_mrmr_rejects_noise_single_target(kind):
    """On single-target types MRMR leaks ZERO noise columns (measured 0/seed on
    5/5). Floor: total noise across 5 seeds <= 2 (loose, but measured is 0)."""
    total_noise = sum(_mrmr_recovery(kind, s)[1] for s in range(5))
    assert total_noise <= 2, f"MRMR on {kind}: expected ~0 noise leakage across 5 seeds; got {total_noise}"


@pytest.mark.parametrize("kind", _MULTI_TARGET)
def test_biz_val_mrmr_rejects_noise_multi_target_2d(kind):
    """On 2D targets MRMR keeps noise leakage small. Measured: <=1 noise/seed.
    Floor: total noise across 5 seeds <= 5 (avg 1/seed)."""
    total_noise = sum(_mrmr_recovery(kind, s)[1] for s in range(5))
    assert total_noise <= 5, f"MRMR on {kind}: expected <=1 noise/seed across 5 seeds; got {total_noise}"


# ===========================================================================
# Learning-to-rank: graded relevance + query groups. MRMR accepts ``groups`` for
# API compat but does NOT consume them (MI is per-row); the relevance label drives
# selection, so signal recovery must hold WITH groups passed -- and the
# groups-not-consumed contract (warn-only, or raise under strict_groups) must hold.
# ===========================================================================


def _ltr_design_with_query_groups(seed: int, n_queries: int = 40):
    """LtR data: 2 signal + 6 noise cols, graded relevance (0..4) from x0+x1, and a contiguous qid array."""
    rng, df, xs = _design(seed)
    score = xs[:, 0] + xs[:, 1]
    relevance = np.digitize(score, np.quantile(score, [0.2, 0.4, 0.6, 0.8])).astype(int)
    qid = np.sort(rng.integers(0, n_queries, size=N))  # sorted -> contiguous per-query blocks (LtR convention)
    return df, pd.Series(relevance, name="relevance"), qid


def test_biz_val_mrmr_ltr_recovers_signal_with_query_groups():
    """MRMR on an LtR target (graded relevance + query groups) recovers BOTH signal cols and rejects
    noise, with ``groups=qid`` passed. The query groups are accepted-but-ignored, so recovery is driven
    by the relevance grade -- it must match the ordinal floor. Measured: 5/5 seeds recover 2/2, 0 noise."""
    n_full, total_noise = 0, 0
    for s in range(5):
        df, y, qid = _ltr_design_with_query_groups(s)
        sel = _make_mrmr(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # groups-not-consumed warning is exercised separately below
            sel.fit(df, y, groups=qid)
        sup = _selected(sel)
        n_full += int(len(SIGNAL & sup) >= N_SIGNAL)
        total_noise += len([i for i in sup if i >= N_SIGNAL])
        assert getattr(sel, "groups_ignored_", False), "LtR: groups_ignored_ should be set when groups passed"
    assert n_full >= 4, f"MRMR LtR: expected >=2/2 signal on >=4/5 seeds; n_full={n_full}"
    assert total_noise <= 2, f"MRMR LtR: expected ~0 noise across 5 seeds; got {total_noise}"


def test_mrmr_ltr_groups_accepted_but_not_consumed_contract():
    """Contract: passing query ``groups`` warns (groups-not-consumed) by default and raises under
    ``strict_groups=True``. Pins the documented LtR API behaviour so a silent change is caught."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y, qid = _ltr_design_with_query_groups(0)

    sel = _make_mrmr(0)
    with pytest.warns(UserWarning, match="does NOT consume them"):
        sel.fit(df, y, groups=qid)
    assert sel.groups_ignored_ is True

    strict = MRMR(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
        strict_groups=True,
    )
    with pytest.raises(NotImplementedError):
        strict.fit(df, y, groups=qid)


def test_biz_val_mrmr_accepts_2d_y_shape():
    """Contract pin: MRMR.fit accepts a 2D ``y`` (DataFrame) for multilabel /
    multi-output WITHOUT raising, and produces a non-trivial selection. This is
    the capability RFECV lacks (see the GAP xfails below)."""
    for kind in _MULTI_TARGET:
        df, y = make_target(0, kind)
        assert getattr(y, "ndim", 1) == 2
        sel = _make_mrmr(0)
        sel.fit(df, y)  # must not raise
        assert len(_selected(sel)) >= 1


def test_biz_val_mrmr_multioutput_union_aggregates_per_column():
    """Default ``multioutput_strategy='union'`` fits one single-target MRMR per output column and unions the selected raw columns, recovering
    BOTH generating features on a 2D target where each column is driven by a different feature (the merged-target greedy drops the 2nd)."""
    df, y = make_target(0, "multioutput")
    sel = _make_mrmr(0)
    assert sel.multioutput_strategy == "union"
    sel.fit(df, y)
    assert SIGNAL.issubset(_selected(sel)), f"union must recover {SIGNAL}; got {sorted(_selected(sel))}"
    assert set(sel.multioutput_supports_) == {"o0", "o1"}
    assert sel.multioutput_strategy_ == "union"


def test_biz_val_mrmr_multioutput_strategy_none_uses_legacy_merged_target():
    """Opt-out: ``multioutput_strategy=None`` keeps the legacy merged-target path -- no per-column union attributes are set."""
    df, y = make_target(0, "multioutput")
    sel = _make_mrmr(0)
    sel.multioutput_strategy = None
    sel.fit(df, y)
    assert not hasattr(sel, "multioutput_supports_")


def test_biz_val_mrmr_multioutput_strategy_validation():
    """An invalid ``multioutput_strategy`` is rejected at fit with a clear error naming the param."""
    df, y = make_target(0, "multioutput")
    sel = _make_mrmr(0)
    sel.multioutput_strategy = "bogus"
    with pytest.raises(ValueError, match="multioutput_strategy"):
        sel.fit(df, y)


def _multioutput_target_from_cols(seed: int, sig_a: int, sig_b: int):
    """A 2D 'union' target whose two columns are driven by features ``sig_a``/``sig_b``.
    The X design is identical to ``make_target`` (8 columns) so two such targets share the
    same ``(strategy, n_features)`` while differing only in WHICH features carry the signal."""
    rng, df, _ = _design(seed)
    xa = df.iloc[:, sig_a].to_numpy()
    xb = df.iloc[:, sig_b].to_numpy()
    o0 = xa + 0.3 * rng.normal(size=N)
    o1 = xb + 0.3 * rng.normal(size=N)
    y = pd.DataFrame(np.column_stack([o0, o1]), columns=["o0", "o1"])
    return df, y


def test_biz_val_mrmr_multioutput_refit_does_not_replay_stale_support():
    """Two different-CONTENT multioutput fits on the SAME instance that share an identical
    ``(strategy, n_features)`` must each recover THEIR OWN signal columns -- the second fit must
    not in-object-skip-replay the first fit's ``support_``. The signature for this path is ``None``
    by construction, so the single-target skip check can never spuriously match it."""
    sel = _make_mrmr(0)
    df1, y1 = _multioutput_target_from_cols(0, sig_a=0, sig_b=1)
    sel.fit(df1, y1)
    sup1 = _selected(sel)
    assert {0, 1}.issubset(sup1), f"first fit must recover {{0,1}}; got {sorted(sup1)}"
    assert sel.signature is None

    df2, y2 = _multioutput_target_from_cols(1, sig_a=2, sig_b=3)
    sel.fit(df2, y2)
    sup2 = _selected(sel)
    assert {2, 3}.issubset(sup2), f"second fit (signal in cols 2,3) must recover {{2,3}}, not stale-replay the first fit's {{0,1}}; got {sorted(sup2)}"


# ===========================================================================
# RFECV -- SINGLE-target target types it DOES handle.
# ===========================================================================


def _make_rfecv(estimator):
    """Build a lightweight RFECV wrapping the given estimator for target-type coverage tests."""
    from mlframe.feature_selection.wrappers import RFECV

    return RFECV(
        estimator=estimator,
        cv=3,
        max_refits=4,
        random_state=0,
        leakage_corr_threshold=None,
        n_features_selection_rule="argmax",
    )


@pytest.mark.slow
def test_biz_val_rfecv_multiclass_accepts_and_keeps_signal():
    """RFECV accepts a multiclass (>2) target via a LogisticRegression estimator
    and KEEPS both signal columns in its support_ (it may also keep noise --
    argmax on a tiny n is permissive, so we assert signal-retention, not noise
    rejection, here)."""
    from sklearn.linear_model import LogisticRegression

    df, y = make_target(0, "multiclass")
    sel = _make_rfecv(LogisticRegression(max_iter=200, random_state=0))
    sel.fit(df, y)
    sup = set(np.where(np.asarray(sel.support_))[0].tolist())
    assert SIGNAL.issubset(sup), f"RFECV multiclass must retain {SIGNAL}; got {sorted(sup)}"


@pytest.mark.slow
def test_biz_val_rfecv_ordinal_as_regression_recovers_signal():
    """ORDINAL target treated as continuous (Ridge): RFECV recovers exactly the
    signal set {x0, x1} (measured) -- ordinal levels carry the x0+x1 ordering
    linearly, so RFE prunes all 6 noise columns."""
    from sklearn.linear_model import Ridge

    _rng, df, xs = _design(0)
    score = xs[:, 0] + xs[:, 1]
    y = pd.Series(np.digitize(score, np.quantile(score, [0.2, 0.4, 0.6, 0.8])).astype(float))
    sel = _make_rfecv(Ridge())
    sel.fit(df, y)
    sup = set(np.where(np.asarray(sel.support_))[0].tolist())
    assert SIGNAL.issubset(sup), f"RFECV ordinal-as-reg must retain {SIGNAL}; got {sorted(sup)}"


@pytest.mark.slow
def test_biz_val_rfecv_count_poisson_accepts():
    """COUNT/Poisson target via PoissonRegressor: RFECV runs and retains the
    signal. Noise rejection is WEAK on a count target at this n (argmax keeps the
    full set), so this pins acceptance + signal-retention only -- the weakness is
    documented in the matrix, not asserted away."""
    from sklearn.linear_model import PoissonRegressor

    rng, df, xs = _design(0)
    lam = np.exp(0.7 * xs[:, 0] + 0.6 * xs[:, 1])
    y = pd.Series(rng.poisson(lam))
    sel = _make_rfecv(PoissonRegressor(max_iter=300))
    sel.fit(df, y)
    sup = set(np.where(np.asarray(sel.support_))[0].tolist())
    assert SIGNAL.issubset(sup), f"RFECV count/Poisson must retain {SIGNAL}; got {sorted(sup)}"


# ===========================================================================
# RFECV -- 2D y (multilabel / multi-output) via opt-in multioutput_strategy.
# Capability closed: fit one single-target RFECV per output column and union
# the per-column support_. DEFAULT 'union' just works; None opts out to the clear
# NotImplementedError -- both pinned in the last two tests below.
# ===========================================================================


@pytest.mark.slow
def test_biz_val_rfecv_multilabel_recovers_signal():
    """multioutput_strategy='union' on a 2D multilabel y recovers BOTH generating
    columns {x0, x1}: one single-target RFECV per label, support_ OR-aggregated."""
    from sklearn.linear_model import LogisticRegression

    df, y = make_target(0, "multilabel")
    sel = _make_rfecv(LogisticRegression(max_iter=200, random_state=0))
    sel.multioutput_strategy = "union"
    sel.fit(df, y)
    sup = set(np.where(np.asarray(sel.support_))[0].tolist())
    assert SIGNAL.issubset(sup), f"RFECV multilabel union must retain {SIGNAL}; got {sorted(sup)}"
    assert set(sel.multioutput_supports_) == {"l0", "l1"}


@pytest.mark.slow
def test_biz_val_rfecv_multioutput_recovers_signal():
    """multioutput_strategy='union' on a 2D multi-target regression y recovers
    BOTH generating columns {x0, x1} via per-target RFECV + OR-aggregation."""
    from sklearn.linear_model import Ridge

    df, y = make_target(0, "multioutput")
    sel = _make_rfecv(Ridge())
    sel.multioutput_strategy = "union"
    sel.fit(df, y)
    sup = set(np.where(np.asarray(sel.support_))[0].tolist())
    assert SIGNAL.issubset(sup), f"RFECV multioutput union must retain {SIGNAL}; got {sorted(sup)}"
    assert set(sel.multioutput_supports_) == {"o0", "o1"}


@pytest.mark.slow
def test_biz_val_rfecv_multioutput_intersect_is_subset_of_union():
    """'intersect' (AND) keeps only features selected for EVERY output, so it is
    a subset of the 'union' (OR) selection on the same fixture -- precision vs
    recall. Pins both aggregation modes against silent regression."""
    from sklearn.linear_model import Ridge

    df, y = make_target(0, "multioutput")
    u = _make_rfecv(Ridge())
    u.multioutput_strategy = "union"
    u.fit(df, y)
    i = _make_rfecv(Ridge())
    i.multioutput_strategy = "intersect"
    i.fit(df, y)
    sup_u = set(np.where(np.asarray(u.support_))[0].tolist())
    sup_i = set(np.where(np.asarray(i.support_))[0].tolist())
    assert sup_i.issubset(sup_u), f"intersect {sorted(sup_i)} must subset union {sorted(sup_u)}"


def test_biz_val_rfecv_multioutput_strategy_validation():
    """An invalid multioutput_strategy is rejected at construction with a clear
    ValueError, not deferred to a cryptic fit-time crash."""
    from sklearn.linear_model import Ridge
    from mlframe.feature_selection.wrappers import RFECV

    with pytest.raises(ValueError, match="multioutput_strategy"):
        RFECV(estimator=Ridge(), cv=3, multioutput_strategy="bogus")


def test_biz_val_rfecv_2d_y_default_handles_multioutput():
    """Pin the FRIENDLY DEFAULT: a bare RFECV.fit on 2D y just works -- multioutput_strategy
    defaults to 'union', so it fits one single-target RFECV per output column and unions the
    support_, recovering the generating signal without the caller opting in."""
    from sklearn.linear_model import LogisticRegression

    df, y = make_target(0, "multilabel")
    sel = _make_rfecv(LogisticRegression(max_iter=100))  # no strategy set -> ctor default 'union'
    assert sel.multioutput_strategy == "union"
    sel.fit(df, y)
    sup = set(np.where(np.asarray(sel.support_))[0].tolist())
    assert SIGNAL.issubset(sup), f"default-union RFECV on 2D y must retain {SIGNAL}; got {sorted(sup)}"


def test_biz_val_rfecv_2d_y_none_opts_out_to_raise():
    """Opt-OUT contract: multioutput_strategy=None explicitly restores the historical clear
    NotImplementedError on 2D y (for a caller who wants the hard failure)."""
    from sklearn.linear_model import LogisticRegression

    df, y = make_target(0, "multilabel")
    sel = _make_rfecv(LogisticRegression(max_iter=100))
    sel.multioutput_strategy = None
    with pytest.raises(NotImplementedError, match="multi-output"):
        sel.fit(df, y)
