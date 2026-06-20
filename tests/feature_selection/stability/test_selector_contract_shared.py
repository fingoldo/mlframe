"""Capability-flagged shared contract over ALL registered selectors + the
production-default GroupAware wrap + the public HybridSelector.

This is the lifted home for the cross-selector sklearn-protocol + robustness
contracts that previously existed as MRMR-only "Wave 9.1" regression files
(get_support, set_output-adjacent, transform n_features_in_ validation,
pickle round-trip, sample_weight semantics, NaN/Inf policy, column-order
invariance, misaligned pandas index, duplicate-name rejection, global-RNG
hygiene). Every selector is enrolled through the single
``_selector_factories.SELECTOR_SPECS`` source; capability flags drive whether a
given contract is a hard assertion or an explicit xfail (no silent skips).

Heavy real-selector specs (BorutaShap / ShapProxiedFS / HybridSelector /
GroupAware) are ``slow``-marked and fitted ONCE per spec via a module-level
cache so the ~10-12 s shadow/model fits are not paid per test.
"""
from __future__ import annotations

import pickle
import random
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from tests.feature_selection._selector_factories import (
    SELECTOR_SPECS,
    selected_mask,
    selected_names,
    spec_params,
)


# --- shared data (built once) ----------------------------------------------


def _binary_frame(n=400, p=10, seed=0):
    # Strong, well-separated signal so the SELECTED SET is stable across fits --
    # the column-order / index-alignment invariance contracts test reproducibility
    # of the selection, not robustness to a noisy near-tie (a flip on noisy data
    # is a power artefact, not an invariance violation).
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=p, n_informative=5,
                               n_redundant=0, n_classes=2, random_state=seed,
                               shuffle=False, class_sep=2.5)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


_BINARY_X, _BINARY_Y = _binary_frame()


def _fit(selector, X, y, **fit_params):
    # Reset the global numpy/random RNG to a fixed state before every fit so the
    # contract's own apples-to-apples comparisons (column-order, refit) are not
    # perturbed by global-RNG pollution leaked from a prior selector's fit. Any
    # selector that LEAKS global RNG is caught separately by TestGlobalRngHygiene;
    # any selector that DEPENDS on global RNG is made deterministic here, so a
    # column-order flip that survives this reset is a genuine order-sensitivity.
    np.random.seed(0)
    random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return selector.fit(X, y, **fit_params)


# Module-level fitted cache: each heavy selector is fitted ONCE on the canonical
# binary frame. Tests that only READ fitted state share this; tests needing a
# fresh/unfitted instance call ``spec.make()`` directly.
_FIT_CACHE: dict[str, object] = {}


def fitted_binary(spec):
    if spec.name not in _FIT_CACHE:
        # Fit on a COPY so a (hypothetical future) frame-mutation regression in one
        # selector cannot corrupt the shared module frame and cascade into unrelated
        # specs' assertions. Frame non-mutation itself is pinned by
        # TestUniversalContract.test_fit_does_not_mutate_input_frame.
        _FIT_CACHE[spec.name] = _fit(spec.make("binary"), _BINARY_X.copy(), _BINARY_Y)
    return _FIT_CACHE[spec.name]


_SPECS = spec_params()


# ===========================================================================
# Universal invariants (read-only against the cached fit)
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestUniversalContract:
    def test_n_features_in_matches_input(self, spec):
        sel = fitted_binary(spec)
        # Selectors may drop degenerate columns at fit entry, so allow <=.
        assert 1 <= int(getattr(sel, "n_features_in_", -1)) <= _BINARY_X.shape[1]

    def test_feature_names_in_recorded(self, spec):
        sel = fitted_binary(spec)
        names = list(getattr(sel, "feature_names_in_", []))
        assert names, f"{spec.name}: feature_names_in_ not recorded"
        assert set(names) <= set(_BINARY_X.columns)

    def test_support_mask_well_formed(self, spec):
        sel = fitted_binary(spec)
        mask = selected_mask(sel)
        assert mask.dtype == bool
        assert mask.shape == (int(sel.n_features_in_),)
        assert int(mask.sum()) >= 1, f"{spec.name}: selected zero features"

    def test_transform_row_count_and_width(self, spec):
        sel = fitted_binary(spec)
        Xt = sel.transform(_BINARY_X)
        assert Xt.shape[0] == _BINARY_X.shape[0]
        n_eng = len(getattr(sel, "_engineered_recipes_", []))
        assert Xt.shape[1] == int(selected_mask(sel).sum()) + n_eng

    def test_not_fitted_before_fit_raises(self, spec):
        sel = spec.make("binary")
        with pytest.raises((NotFittedError, AttributeError, ValueError)):
            sel.transform(_BINARY_X)

    def test_fit_does_not_mutate_input_frame(self, spec):
        # A selector must never add/drop/reorder columns on the caller's input
        # DataFrame (CLAUDE.md RAM rule: a leaked engineered column on a 100 GB
        # frame is both a memory leak and a silent schema corruption). MRMR's
        # hinge / cat-cross FE used to materialise legs into the caller's X in
        # place (``X[leg] = ...``) -- this pins that fit leaves X identical.
        X = _BINARY_X.copy()
        cols_before = list(X.columns)
        _fit(spec.make("binary"), X, _BINARY_Y)
        assert list(X.columns) == cols_before, (
            f"{spec.name}: fit mutated the caller's DataFrame columns "
            f"(added {[c for c in X.columns if c not in cols_before]}, "
            f"dropped {[c for c in cols_before if c not in list(X.columns)]})")

    def test_get_feature_names_out_capability(self, spec):
        sel = fitted_binary(spec)
        has = callable(getattr(sel, "get_feature_names_out", None))
        if spec.has_gfno:
            assert has, f"{spec.name}: has_gfno=True but get_feature_names_out missing"
            names = sel.get_feature_names_out()
            assert len(names) == sel.transform(_BINARY_X).shape[1]
        elif not has:
            pytest.xfail(f"{spec.name}: no get_feature_names_out (declared sklearn-parity gap)")

    def test_get_support_capability(self, spec):
        sel = fitted_binary(spec)
        gs = getattr(sel, "get_support", None)
        if spec.has_get_support:
            assert callable(gs), f"{spec.name}: has_get_support=True but get_support missing"
            mask = np.asarray(gs())
            idx = np.asarray(gs(indices=True))
            assert np.array_equal(np.where(np.asarray(_as_bool(mask, sel)))[0], idx)
        elif not callable(gs):
            pytest.xfail(f"{spec.name}: no get_support (declared sklearn-parity gap)")


def _as_bool(mask, sel):
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask
    out = np.zeros(int(sel.n_features_in_), dtype=bool)
    out[mask.astype(int)] = True
    return out


# ===========================================================================
# Registry tripwire: every production-registered selector has a contract spec
# ===========================================================================


def test_every_registered_selector_has_contract_spec():
    """A 5th registry entry without a SelectorSpec here fails CI -- registration
    implies contract coverage (kills the two-files-can-drift gap)."""
    from mlframe.feature_selection import registry
    missing = set(registry.available()) - set(SELECTOR_SPECS)
    assert not missing, (
        f"registry selectors without a contract spec in _selector_factories.SELECTOR_SPECS: "
        f"{sorted(missing)} -- add a SelectorSpec so registration implies contract coverage"
    )


# ===========================================================================
# B2: pickle round-trip -- HARD fail on regression (capability-gated, no skip)
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestPicklePersistence:
    def test_pickle_roundtrip_preserves_transform(self, spec):
        sel = fitted_binary(spec)
        if not spec.pickle_safe:
            with pytest.raises(Exception):
                pickle.dumps(sel)
            return
        blob = pickle.dumps(sel)  # NO try/except: a pickle regression must FAIL
        restored = pickle.loads(blob)
        before = np.asarray(sel.transform(_BINARY_X))
        after = np.asarray(restored.transform(_BINARY_X))
        assert before.shape == after.shape
        np.testing.assert_allclose(np.asarray(before, dtype=float),
                                   np.asarray(after, dtype=float), rtol=0, atol=0)


# ===========================================================================
# B2: transform-time n_features_in_ validation (wrong width raises)
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestTransformWidthValidation:
    def test_wrong_ndarray_width_raises(self, spec):
        sel = fitted_binary(spec)
        if not spec.accepts_ndarray:
            pytest.xfail(f"{spec.name}: DataFrame-only (declared)")
        if not spec.validates_transform_width:
            pytest.xfail(f"{spec.name}: no transform-time width validation "
                         "(silent positional indexing -- prod guard backlog)")
        nfin = int(sel.n_features_in_)
        bad = np.random.default_rng(0).standard_normal((20, nfin + 2))
        with pytest.raises((ValueError, KeyError, IndexError)):
            sel.transform(bad)

    def test_extra_dataframe_columns_realign_by_name(self, spec):
        sel = fitted_binary(spec)
        Xplus = _BINARY_X.copy()
        Xplus.insert(0, "ID_PREPENDED", np.arange(len(Xplus)))  # ETL prepends an id col
        out = sel.transform(Xplus)  # name-based realign must ignore the extra col
        n_eng = len(getattr(sel, "_engineered_recipes_", []))
        assert out.shape[1] == int(selected_mask(sel).sum()) + n_eng


# ===========================================================================
# B2: fit-time column-order invariance of the SELECTED SET [DEDUP-F]
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestColumnOrderInvariance:
    def test_reversed_columns_select_same_names(self, spec):
        if not spec.column_order_invariant:
            pytest.xfail(f"{spec.name}: selection depends on input column order "
                         "(positional tie-break / shadow ordering -- reproducibility gap)")
        rev = list(_BINARY_X.columns)[::-1]
        s1 = _fit(spec.make("binary"), _BINARY_X, _BINARY_Y)
        s2 = _fit(spec.make("binary"), _BINARY_X[rev], _BINARY_Y)
        names1, names2 = set(selected_names(s1)), set(selected_names(s2))
        if spec.determinism >= 1.0:
            assert names1 == names2, (
                f"{spec.name}: column reorder changed selection {names1} vs {names2}")
        else:
            inter = len(names1 & names2)
            union = len(names1 | names2) or 1
            assert inter / union >= spec.determinism


# ===========================================================================
# B2: misaligned pandas index (positional semantics) [DEDUP-F]
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestIndexAlignment:
    def test_nondefault_index_recovers_signal_without_crash(self, spec):
        # A non-default / shuffled pandas index (X rows + y Series carrying the
        # SAME shuffled index, values aligned) is the classic CV-split / df.sample
        # production shape. The contract is: fit must NOT crash on a non-RangeIndex
        # and must still recover the informative block. We do NOT assert bit-identity
        # to the unshuffled fit -- selectors with positional CV folds (RFECV uses
        # StratifiedKFold(shuffle=False), and engineers data-dependent thresholds)
        # legitimately depend on row order; that is reproducibility-under-row-order,
        # not index handling. Column-order invariance (TestColumnOrderInvariance)
        # is the bit-identity contract; this one guards silent index MISalignment.
        rng = np.random.default_rng(1)
        perm = rng.permutation(len(_BINARY_X))
        Xs = _BINARY_X.iloc[perm].copy()
        ys = pd.Series(_BINARY_Y[perm], index=Xs.index)  # values aligned to Xs rows
        sel = _fit(spec.make("binary"), Xs, ys)
        names = set(selected_names(sel))
        informative = {f"f{i}" for i in range(5)}
        # If the selector silently aligned y by pandas index instead of positionally,
        # the labels would be scrambled and signal recovery collapses to ~noise.
        assert names & informative, (
            f"{spec.name}: non-default-index fit recovered no informative feature "
            f"(selected {sorted(names)}) -- possible silent index misalignment")


# ===========================================================================
# B2: sample_weight semantics (capability-gated loud failure) [DEDUP-I]
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestSampleWeight:
    def test_uniform_weight_equals_unweighted(self, spec):
        if not spec.supports_sample_weight:
            # A weight-incapable selector must reject the kwarg LOUDLY -- pin it so
            # the training-suite marker stamping on such a selector is a visible
            # question, never a silent no-op.
            sel = spec.make("binary")
            with pytest.raises(TypeError):
                sel.fit(_BINARY_X, _BINARY_Y, sample_weight=np.ones(len(_BINARY_Y)))
            return
        s_none = fitted_binary(spec)
        s_w = _fit_with_weight(spec, np.ones(len(_BINARY_Y)))
        if spec.determinism >= 1.0:
            assert set(selected_names(s_none)) == set(selected_names(s_w))


def _fit_with_weight(spec, w):
    return _fit(spec.make("binary"), _BINARY_X, _BINARY_Y, sample_weight=w)


# ===========================================================================
# B2: duplicate column-name rejection
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestDuplicateColumnNames:
    def test_duplicate_names_rejected(self, spec):
        Xdup = pd.concat([_BINARY_X, _BINARY_X[["f0"]]], axis=1)  # two cols named f0
        sel = spec.make("binary")
        if not spec.rejects_duplicate_names:
            pytest.xfail(f"{spec.name}: no duplicate-column-name guard at fit entry "
                         "(silent positional pick -- prod guard backlog)")
        with pytest.raises((ValueError, KeyError, AssertionError)):
            _fit(sel, Xdup, _BINARY_Y)


# ===========================================================================
# B2: fit() must not mutate the global numpy RNG (seed passed) [shared_lift-16]
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestGlobalRngHygiene:
    def test_fit_does_not_touch_global_numpy_rng(self, spec):
        # Measure the SELECTOR's effect on the global RNG -- call fit directly,
        # NOT via the harness ``_fit`` (which reseeds for comparison hygiene).
        np.random.seed(12345)
        before = np.random.get_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.make("binary").fit(_BINARY_X, _BINARY_Y)
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2] == after[2]


# ===========================================================================
# B2: NaN-in-X policy (only assert where the spec declares a known policy)
# ===========================================================================


@pytest.mark.parametrize("spec", _SPECS)
class TestNanInXPolicy:
    def test_nan_in_X_declared_policy(self, spec):
        if spec.nan_in_X_policy == "unknown":
            pytest.skip(f"{spec.name}: NaN-in-X policy not yet pinned (measure-then-pin backlog)")
        X = _BINARY_X.copy()
        rng = np.random.default_rng(0)
        nan_rows = rng.choice(len(X), size=int(0.05 * len(X)), replace=False)
        X.iloc[nan_rows, 0] = np.nan
        X.iloc[nan_rows, 1] = np.nan
        sel = spec.make("binary")
        if spec.nan_in_X_policy == "raises":
            with pytest.raises((ValueError, TypeError)):
                _fit(sel, X, _BINARY_Y)
        else:  # tolerates
            fitted = _fit(sel, X, _BINARY_Y)
            assert int(selected_mask(fitted).sum()) >= 1
