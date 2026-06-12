"""Consolidated from test_biz_value_mrmr_layer16.py.

Layer 16 biz_value MRMR contracts: MULTICLASS + ORDINAL classification.

WHY THIS LAYER
--------------
Layers 1-14 covered binary y and Layer 15 covered continuous regression.
Multiclass classification (n_classes > 2) and ordinal classification
(ordered levels, often domain-coded as strings) are the third major
production target shape we had ZERO coverage for:

* sentiment / star ratings (5-star reviews on every e-commerce site)
* severity / triage levels (P0-P4 in incident management; low/med/high
  in healthcare prioritisation)
* credit ratings (A / B / C / D / F or AAA / AA / .. / D in finance)
* customer-tier / segment classifiers (bronze / silver / gold / platinum)

A silent quality drop here -- e.g. MRMR collapsing all minority classes
into one majority class via int16 downcast, or treating string-coded
ordinal labels as a million-card categorical and screening the y
column itself -- would let a feature-selection front-end ship with a
hidden ceiling on every ratings / severity / segmentation model in
production.

DATA DESIGN
-----------
Two sub-scenarios:

A. MULTICLASS NOMINAL (5 classes): y in {0, 1, 2, 3, 4}, drawn from
   a 5-component Gaussian mixture in (x1, x2). x1 and x2 are
   class-discriminative; 5 noise columns are sampled independently
   from y. We additionally test a 70/20/10 imbalanced 3-class
   variant where MRMR must still surface the signals despite the
   tail-class scarcity.

B. ORDINAL (5 levels): y is ordered low < med_low < med < med_high <
   high, derived from a monotonic linear combination of x1 + x2.
   Tested in three encodings:
     * numeric-coded (int) -- the canonical sklearn path
     * string-coded (object) -- what pandas.read_csv yields when the
       column already carries domain labels
     * ordered pd.Categorical -- the recommended pandas idiom

CONTRACTS PINNED
----------------
1. Multiclass nominal: both signal columns (x1, x2) appear in
   ``support_`` on every seed; no ``noise_*`` column appears.
2. Multiclass downstream usefulness: a LogisticRegression on the MRMR
   support beats macro-F1 >= 0.70 on holdout -- the "did MRMR pick
   semantically useful columns" anchor.
3. Imbalanced multiclass (70 / 20 / 10): the minority-class signal
   still gets selected (i.e. MRMR does NOT silently collapse to the
   majority-vs-rest binarisation).
4. Per-class minority recall via the MRMR support is not collapsed to
   zero (>= 0.40 on the smallest class) -- detects the silent
   "minority class lost" failure mode end-to-end.
5. Ordinal numeric-coded y works and recovers x1, x2.
6. Ordinal string-coded y EITHER works (recovering x1, x2) OR raises
   an actionable ValueError naming the offending dtype. Silent
   crash / silent wrong support_ is forbidden.
7. Ordinal pd.Categorical (ordered=True) works and recovers x1, x2.
8. Ordinal float-coded y works (production frames often hold integer
   levels as float32 after a generic ETL cast).

DEFAULT-CONFIG SURFACE
----------------------
DCD ON, relative-gain stop ON, Miller-Madow ON, MDLP nbins_strategy ON
(Wave 9 flip 2026-05-30). Layer 16 respects those defaults; only
``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned to keep
wall-time bounded. They do NOT interact with multiclass/ordinal
target handling.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TOTAL = 2_500
N_NOISE = 5
N_HOLDOUT = 500
SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_multiclass_5way_data(seed: int):
    """5-class Gaussian mixture in (x1, x2); 5 i.i.d. noise columns.

    Cluster centers are far enough apart (radius ~3) that x1 and x2
    individually carry strong class-discriminative MI; a multiclass-
    aware MI estimator must surface them and drop the noise.
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-3.0, 0.0], [3.0, 0.0], [0.0, -3.0], [0.0, 3.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    y_arr = rng.integers(0, 5, N_TOTAL)
    x1 = centers[y_arr, 0] + rng.standard_normal(N_TOTAL) * 0.7
    x2 = centers[y_arr, 1] + rng.standard_normal(N_TOTAL) * 0.7
    cols = {"x1": x1, "x2": x2}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_5class")
    return X, y


def _build_imbalanced_3class_data(seed: int):
    """70 / 20 / 10 imbalanced 3-class; tests minority-signal survival.

    Classes are well-separated geometrically (no overlap region), so
    the minority class (10%) leaves a clean signature in x1, x2 even
    though it represents only ~10% of rows. A class-blind importance
    scorer that effectively measures majority-vs-rest would still
    surface x1, x2; the contract here is the WEAKER "support_ is not
    empty + downstream minority-class recall is non-zero".
    """
    rng = np.random.default_rng(seed)
    probs = np.array([0.70, 0.20, 0.10])
    y_arr = rng.choice([0, 1, 2], size=N_TOTAL, p=probs)
    centers = np.array([[0.0, 0.0], [2.8, 0.0], [0.0, 2.8]], dtype=np.float64)
    x1 = centers[y_arr, 0] + rng.standard_normal(N_TOTAL) * 0.55
    x2 = centers[y_arr, 1] + rng.standard_normal(N_TOTAL) * 0.55
    cols = {"x1": x1, "x2": x2}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_imb3")
    return X, y


def _build_ordinal_data(seed: int, encoding: str = "int"):
    """5-level ordinal y derived from monotonic z = 1.5*x1 + 0.8*x2.

    Levels are pd.qcut quintiles of z, so the ordering of (level, mean
    x1) is strictly monotonic by construction. We expose three encodings:

      * ``int``  -- the canonical sklearn path (np.int64 codes 0..4).
      * ``str``  -- domain string labels in unordered semantic order
        (low / med_low / med / med_high / high). MRMR has no way to
        recover the intended ordering from the string codes alone; we
        only test that it does not silently mis-bin them.
      * ``ordered_cat`` -- pd.Categorical(ordered=True) with the
        domain labels in correct order.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    z = 1.5 * x1 + 0.8 * x2 + 0.25 * rng.standard_normal(N_TOTAL)
    codes = pd.qcut(z, q=5, labels=False).astype(np.int64)
    cols = {"x1": x1, "x2": x2}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    X = pd.DataFrame(cols)
    labels = np.array(["low", "med_low", "med", "med_high", "high"])
    if encoding == "int":
        y = pd.Series(codes, name="y_ord_int")
    elif encoding == "str":
        y = pd.Series(labels[codes], name="y_ord_str")
    elif encoding == "ordered_cat":
        cat = pd.Categorical.from_codes(codes, categories=labels, ordered=True)
        y = pd.Series(cat, name="y_ord_cat")
    elif encoding == "float":
        y = pd.Series(codes.astype(np.float32), name="y_ord_float")
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    return X, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    """Default-config MRMR -- Wave 9 production surface.

    Only ``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned
    to keep wall-time bounded; they don't interact with the multiclass /
    ordinal target handling path tested here.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_quiet(sel, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.fit(X, y)


# ---------------------------------------------------------------------------
# Contract 1: Multiclass nominal -- both signals recovered, no noise leak
# ---------------------------------------------------------------------------


class TestMulticlassNominalSignalRecovery:
    """5-class Gaussian mixture: x1 and x2 individually carry strong
    class-discriminative MI. MRMR must surface BOTH on every seed and
    leak no noise column.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_both_signals_in_support_5class(self, seed):
        X, y = _build_multiclass_5way_data(seed)
        assert y.nunique() == 5, f"test bug: y has {y.nunique()} classes"
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names, (
            f"signal x1 missing from 5-class support_; "
            f"seed={seed}, support={names}"
        )
        assert "x2" in names, (
            f"signal x2 missing from 5-class support_; "
            f"seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_no_noise_leak_5class(self, seed):
        X, y = _build_multiclass_5way_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"noise column(s) {leaked} leaked into 5-class support_; "
            f"seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 2: Multiclass downstream usefulness (macro-F1)
# ---------------------------------------------------------------------------


class TestMulticlassDownstreamAnchor:
    """The "did MRMR pick semantically useful columns" sanity gate:
    a LogisticRegression on the MRMR support must beat macro-F1 >= 0.70
    on a 5-class holdout. A noise-only / collapsed support would yield
    macro-F1 ~= 0.20 (random over 5 classes).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_macro_f1_on_support(self, seed):
        X, y = _build_multiclass_5way_data(seed)
        X_tr, X_te = X.iloc[:-N_HOLDOUT].copy(), X.iloc[-N_HOLDOUT:].copy()
        y_tr, y_te = y.iloc[:-N_HOLDOUT], y.iloc[-N_HOLDOUT:]
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X_tr, y_tr)
        Xs_tr = sel.transform(X_tr)
        Xs_te = sel.transform(X_te)
        model = LogisticRegression(max_iter=500).fit(Xs_tr, y_tr)
        macro_f1 = f1_score(y_te, model.predict(Xs_te), average="macro")
        assert macro_f1 >= 0.70, (
            f"LogReg on MRMR 5-class support has macro-F1={macro_f1:.3f} "
            f"on holdout; expected >= 0.70 for a 5-Gaussian-mixture "
            f"design. Selected: {list(sel.get_feature_names_out())}, "
            f"seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 3: Imbalanced 70 / 20 / 10 -- minority signal preserved
# ---------------------------------------------------------------------------


class TestImbalancedMulticlassMinorityRecall:
    """The class-imbalance failure mode is silent: a MI estimator that
    effectively scores ``majority vs everything else`` would still
    surface x1, x2 (they discriminate majority from minority), but
    the per-class minority recall on the downstream classifier would
    collapse to ~0 because the bin edges chosen by MDLP collapse the
    minority class into the majority's bin.

    The contract therefore has two halves:
      a) Both signals appear in support_ on every seed.
      b) Downstream LogReg minority-class recall >= 0.40 on the
         10%-prevalent class -- i.e. MRMR's binning did NOT erase
         the minority signature.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_signals_in_support_imbalanced(self, seed):
        X, y = _build_imbalanced_3class_data(seed)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names and "x2" in names, (
            f"imbalanced 3-class: signals missing from support_; "
            f"seed={seed}, support={names}, class counts="
            f"{np.bincount(y.to_numpy())}"
        )
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"imbalanced 3-class: noise leaked into support_ "
            f"({leaked}); seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_minority_class_recall_preserved(self, seed):
        X, y = _build_imbalanced_3class_data(seed)
        X_tr, X_te = X.iloc[:-N_HOLDOUT].copy(), X.iloc[-N_HOLDOUT:].copy()
        y_tr, y_te = y.iloc[:-N_HOLDOUT], y.iloc[-N_HOLDOUT:]
        # Sanity: holdout must contain at least a few minority samples
        # for the recall number to be meaningful.
        if int((y_te == 2).sum()) < 10:
            pytest.skip(
                f"holdout has <10 minority rows on seed={seed}; "
                f"recall estimate would be unstable"
            )
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X_tr, y_tr)
        Xs_tr = sel.transform(X_tr)
        Xs_te = sel.transform(X_te)
        model = LogisticRegression(max_iter=500).fit(Xs_tr, y_tr)
        preds = model.predict(Xs_te)
        per_class_recall = recall_score(y_te, preds, average=None, labels=[0, 1, 2])
        minority_recall = per_class_recall[2]
        assert minority_recall >= 0.40, (
            f"Minority-class (10%) recall collapsed to "
            f"{minority_recall:.3f} on MRMR support -- imbalanced "
            f"binning likely erased the minority signature. "
            f"per-class recall={per_class_recall}, "
            f"selected={list(sel.get_feature_names_out())}, seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 4: Ordinal int-coded works
# ---------------------------------------------------------------------------


class TestOrdinalIntCoded:
    """Numeric (int) ordinal: the canonical sklearn-style encoding.
    Both signals must be recovered; gain order should reflect that x1
    (coef=1.5) is stronger than x2 (coef=0.8).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_int_ordinal_recovers_signals(self, seed):
        X, y = _build_ordinal_data(seed, encoding="int")
        assert y.dtype.kind in "iu", f"int ordinal y dtype={y.dtype}"
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names, (
            f"int ordinal: x1 (stronger signal) missing from "
            f"support_; seed={seed}, support={names}"
        )
        assert "x2" in names, (
            f"int ordinal: x2 missing from support_; "
            f"seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_int_ordinal_x1_picked_first(self, seed):
        """x1 has a 1.5x coefficient and x2 has 0.8x -- the stronger
        signal should be selected first by MRMR's greedy step.
        """
        X, y = _build_ordinal_data(seed, encoding="int")
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        assert len(names) >= 1
        assert names[0] == "x1", (
            f"int ordinal: stronger signal x1 (coef=1.5) not picked "
            f"first; got {names[0]} on seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_int_ordinal_no_noise(self, seed):
        X, y = _build_ordinal_data(seed, encoding="int")
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"int ordinal: noise {leaked} in support_; "
            f"seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 5: Ordinal string-coded -- works or raises actionable error
# ---------------------------------------------------------------------------


class TestOrdinalStringCoded:
    """Strict contract: MRMR.fit on a string-coded ordinal y must EITHER
    succeed AND recover x1+x2, OR raise an actionable ``ValueError``/
    ``TypeError`` whose message names the offending dtype. A silent
    crash (e.g. ``np.isnan`` on object array) or a silent wrong
    support_ (e.g. y collapsed to a single bin) would be a real bug.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_string_ordinal_either_works_or_raises_clean(self, seed):
        X, y = _build_ordinal_data(seed, encoding="str")
        assert y.dtype.kind in ("O", "U") or str(y.dtype) == "str", (
            f"test bug: expected object/string dtype, got {y.dtype}"
        )
        sel = _make_mrmr(random_seed=seed)
        try:
            _fit_quiet(sel, X.copy(), y)
        except (ValueError, TypeError) as exc:
            msg = str(exc).lower()
            # Must be actionable: mention dtype, object, str, label, or encode.
            assert any(kw in msg for kw in (
                "dtype", "object", "string", "str", "label",
                "encode", "categor", "non-numeric",
            )), (
                f"string ordinal: raised {type(exc).__name__} but "
                f"message {exc!r} is not actionable; expected to name "
                f"dtype/object/string/label/encode/categorical/"
                f"non-numeric. seed={seed}"
            )
            return  # Acceptable failure mode.
        # If fit succeeded, the support must contain BOTH signals -- a
        # successful fit with x1 missing would mean MRMR silently
        # mis-coded the string y and degraded relevance.
        names = list(sel.get_feature_names_out())
        assert "x1" in names and "x2" in names, (
            f"string ordinal: fit succeeded but support_ missing "
            f"signals: {names} on seed={seed}. Silent encoding "
            f"degradation is a real bug -- either reject string y "
            f"explicitly or label-encode it transparently."
        )


# ---------------------------------------------------------------------------
# Contract 6: Ordered pd.Categorical works
# ---------------------------------------------------------------------------


class TestOrdinalOrderedCategorical:
    """``pd.Categorical(ordered=True)`` is the recommended pandas idiom
    for ordinal targets. MRMR must accept it (pandas already exposes
    integer codes via .cat.codes) and recover x1+x2.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ordered_cat_recovers_signals(self, seed):
        X, y = _build_ordinal_data(seed, encoding="ordered_cat")
        assert isinstance(y.dtype, pd.CategoricalDtype)
        assert y.cat.ordered, "test bug: cat must be ordered"
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names and "x2" in names, (
            f"ordered cat ordinal: signals missing from support_; "
            f"seed={seed}, support={names}"
        )
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"ordered cat ordinal: noise leaked ({leaked}); "
            f"seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 7: Float-coded ordinal (production ETL silently casts int to float)
# ---------------------------------------------------------------------------


class TestOrdinalFloatCoded:
    """Production frames frequently arrive with integer ordinal levels
    cast to float32 because of a generic ETL ``df.astype(np.float32)``
    memory-budget pass. MRMR must not silently degrade: x1 and x2
    must still be recovered, and parity with int-coded y should hold.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_float_ordinal_recovers_signals(self, seed):
        X, y = _build_ordinal_data(seed, encoding="float")
        assert y.dtype.kind == "f"
        assert y.nunique() == 5, (
            f"test bug: float-coded ordinal lost levels (nunique="
            f"{y.nunique()})"
        )
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names and "x2" in names, (
            f"float ordinal: signals missing from support_; "
            f"seed={seed}, support={names}"
        )
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, (
            f"float ordinal: noise leaked ({leaked}); "
            f"seed={seed}, support={names}"
        )

    @pytest.mark.parametrize("seed", (7, 42))
    def test_float_ordinal_matches_int_ordinal_support(self, seed):
        """Same underlying levels, only the dtype differs. The chosen
        signals should be the same set; the order is allowed to flip
        because MDLP bin edges can differ marginally on float vs int.
        """
        X_int, y_int = _build_ordinal_data(seed, encoding="int")
        X_float, y_float = _build_ordinal_data(seed, encoding="float")
        # Both data builders use the same seed so X is bit-identical.
        sel_int = _make_mrmr(random_seed=seed)
        _fit_quiet(sel_int, X_int.copy(), y_int)
        sel_float = _make_mrmr(random_seed=seed)
        _fit_quiet(sel_float, X_float.copy(), y_float)
        sup_int = set(sel_int.get_feature_names_out())
        sup_float = set(sel_float.get_feature_names_out())
        # The signal pair must be in both; the noise/other-feature
        # tail may differ marginally between int- and float-binned y.
        assert {"x1", "x2"}.issubset(sup_int), (
            f"int-ordinal lost signals: {sup_int}, seed={seed}"
        )
        assert {"x1", "x2"}.issubset(sup_float), (
            f"float-ordinal lost signals: {sup_float}, seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 8: get_feature_names_out / transform shape sanity
# ---------------------------------------------------------------------------


class TestTransformShapeOnMulticlass:
    """End-to-end sklearn-contract sanity on multiclass y: transform
    output column count == len(support_); transform of a held-out
    frame has the same column count. A silent off-by-one would break
    every downstream Pipeline using MRMR with multiclass y.
    """

    def test_transform_columns_match_support(self):
        X, y = _build_multiclass_5way_data(seed=1)
        sel = _make_mrmr(random_seed=1)
        _fit_quiet(sel, X.copy(), y)
        out = sel.transform(X.copy())
        assert out.shape[0] == X.shape[0]
        assert out.shape[1] == sel.n_features_
        assert out.shape[1] == len(sel.get_feature_names_out())

    def test_transform_holdout_shape(self):
        X, y = _build_multiclass_5way_data(seed=1)
        X_tr, X_te = X.iloc[:-N_HOLDOUT].copy(), X.iloc[-N_HOLDOUT:].copy()
        y_tr = y.iloc[:-N_HOLDOUT]
        sel = _make_mrmr(random_seed=1)
        _fit_quiet(sel, X_tr, y_tr)
        out_te = sel.transform(X_te)
        assert out_te.shape == (N_HOLDOUT, sel.n_features_)
