"""Unit tests for the richer OOF cross-target meta-stackers (ridge / lasso / elasticnet / gbm).

Covers: shapes, off-path bit-identity (no _meta_model -> legacy linear blend byte-for-byte), leakage-freeness (meta fit
sees only the OOF matrix the caller passes), ridge non-negative constraint, component-dropout graceful degradation,
sample_weight validation, the uniform-mean fallback on degenerate inputs, and pickle round-trip.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


class _Col:
    """Component model that emits the k-th column of its input matrix."""

    def __init__(self, col: int) -> None:
        self.col = col

    def predict(self, X):
        return np.asarray(X)[:, self.col]


def _toy(n=300, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = X @ np.array([0.6, 0.3, 0.1]) + 0.05 * rng.normal(size=n)
    return X, y


def _models(k):
    return [_Col(i) for i in range(k)], ["c%d" % i for i in range(k)]


# ---------------------------------------------------------------- shapes / happy path


@pytest.mark.parametrize("stacker", ["ridge", "lasso", "elasticnet", "gbm"])
def test_meta_stack_predict_shape(stacker):
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    X, y = _toy()
    models, names = _models(X.shape[1])
    ens = build_meta_stack_ensemble(E, models, names, X, y, stacker=stacker)
    assert getattr(ens, "_meta_model", None) is not None
    assert ens.strategy == "meta_%s" % stacker
    out = ens.predict(X)
    assert out.shape == (X.shape[0],)
    assert np.all(np.isfinite(out))


def test_nnls_via_dispatcher_equals_direct():
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    X, y = _toy()
    models, names = _models(X.shape[1])
    a = build_meta_stack_ensemble(E, models, names, X, y, stacker="nnls")
    b = E.from_nnls_stack(models, names, X, y)
    np.testing.assert_allclose(a.weights, b.weights)
    # NNLS path attaches NO meta-model: predict stays the legacy linear blend.
    assert getattr(a, "_meta_model", None) is None


# ---------------------------------------------------------------- off-path bit identity


def test_default_path_bit_identical_when_no_meta_model():
    """An ensemble with no _meta_model attached must predict byte-for-byte the legacy linear blend (the meta-stack branch
    is a pure no-op when _meta_model is absent)."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble as E

    X, y = _toy()
    models, names = _models(X.shape[1])
    nn = E.from_nnls_stack(models, names, X, y)
    assert getattr(nn, "_meta_model", None) is None
    # Reference linear blend computed exactly as the legacy predict path does it.
    cols = np.column_stack([m.predict(X) for m in models])
    ref = (cols * nn.weights[None, :]).sum(axis=1)
    np.testing.assert_array_equal(nn.predict(X), ref)


# ---------------------------------------------------------------- leakage-freeness


def test_meta_fit_uses_only_passed_oof_matrix():
    """The meta-model is fit on the OOF matrix argument ONLY -- it never re-reads the component models or any holdout. We
    prove this by fitting on a DELIBERATELY swapped OOF matrix: the meta-model's coefficients must reflect the swapped
    columns, not the models' natural column order."""
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    _X, y = _toy(k=3)
    models, names = _models(3)
    # OOF matrix where only column 0 carries the signal; columns 1,2 are pure noise.
    rng = np.random.default_rng(7)
    oof = np.column_stack([y + 0.01 * rng.normal(size=len(y)), rng.normal(size=len(y)), rng.normal(size=len(y))])
    ens = build_meta_stack_ensemble(E, models, names, oof, y, stacker="ridge")
    coef = np.asarray(ens._meta_model.coef_)
    # The ridge meta sees ONLY the OOF matrix: col 0 (signal) must dominate.
    assert abs(coef[0]) > abs(coef[1]) and abs(coef[0]) > abs(coef[2])


# ---------------------------------------------------------------- ridge non-negative


def test_ridge_non_negative_constraint():
    from mlframe.training.composite.ensemble import fit_ridge_meta_stacker

    rng = np.random.default_rng(3)
    n = 400
    a = rng.normal(size=n)
    # col1 is anti-correlated with y -> unconstrained ridge would give it a negative coef.
    y = a + 0.05 * rng.normal(size=n)
    oof = np.column_stack([a, -a + 0.1 * rng.normal(size=n)])
    unconstrained = fit_ridge_meta_stacker(oof, y, 2, non_negative=False)
    constrained = fit_ridge_meta_stacker(oof, y, 2, non_negative=True)
    assert (np.asarray(unconstrained.coef_) < -1e-6).any(), "unconstrained ridge should produce a negative coef here"
    assert np.all(np.asarray(constrained.coef_) >= -1e-9), "non_negative ridge must have all coefs >= 0"


# ---------------------------------------------------------------- dropout degradation


def test_component_dropout_graceful_degradation():
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    class _Boom:
        def predict(self, X):
            raise RuntimeError("component down")

    X, y = _toy(k=3)
    models, names = _models(3)
    ens = build_meta_stack_ensemble(E, models, names, X, y, stacker="gbm")
    # Swap one component for a failing one; predict must still return finite output via mean-fill.
    ens.component_models[1] = _Boom()
    out = ens.predict(X)
    assert out.shape == (X.shape[0],)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- validation / fallbacks


def test_sample_weight_validation():
    from mlframe.training.composite.ensemble import fit_ridge_meta_stacker

    X, y = _toy()
    with pytest.raises(ValueError, match="sample_weight length"):
        fit_ridge_meta_stacker(X, y, X.shape[1], sample_weight=np.ones(len(y) - 1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(len(y))
        sw[0] = -1.0
        fit_ridge_meta_stacker(X, y, X.shape[1], sample_weight=sw)


def test_bad_stacker_name_raises():
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    X, y = _toy()
    models, names = _models(X.shape[1])
    with pytest.raises(ValueError, match="unknown stacker"):
        build_meta_stack_ensemble(E, models, names, X, y, stacker="nope")


def test_degenerate_too_few_rows_falls_back_to_mean():
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    X, y = _toy(n=5, k=3)
    models, names = _models(3)
    ens = build_meta_stack_ensemble(E, models, names, X, y, stacker="gbm")
    # Too few OOF rows for a GBM stacker -> falls back to uniform-weight mean (no meta-model).
    assert ens.strategy == "mean"
    assert getattr(ens, "_meta_model", None) is None


def test_matrix_shape_mismatch_raises():
    from mlframe.training.composite.ensemble import fit_gbm_meta_stacker

    X, y = _toy(k=3)
    with pytest.raises(ValueError, match="expected"):
        fit_gbm_meta_stacker(X, y, 4)  # claim 4 components but matrix has 3 cols


# ---------------------------------------------------------------- pickle round-trip


@pytest.mark.parametrize("stacker", ["ridge", "lasso", "elasticnet", "gbm"])
def test_meta_stack_pickle_roundtrip(stacker):
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E,
        build_meta_stack_ensemble,
    )

    X, y = _toy()
    models, names = _models(X.shape[1])
    ens = build_meta_stack_ensemble(E, models, names, X, y, stacker=stacker)
    before = ens.predict(X)
    restored = pickle.loads(pickle.dumps(ens))
    after = restored.predict(X)
    np.testing.assert_allclose(after, before, rtol=1e-10)
    assert getattr(restored, "_meta_model", None) is not None
