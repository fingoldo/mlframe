"""Adversarial fuzz + metamorphic test suite for ``CompositeTargetEstimator``.

Two complementary QA layers over the composite-target wrapper:

Fuzz (``TestFuzzFitPredictNeverCrashes``)
    Across many random seeds, sample a random (transform from the registry,
    base column, n in [20, 2000], feature count, NaN fraction, outlier
    injection, dtype) config and assert ``fit -> predict`` never crashes and
    yields finite ``y`` inside the train envelope (or routed to the fallback
    constant, which is itself inside the envelope). Feature-NaN cells use a
    NaN-tolerant inner (HistGradientBoostingRegressor): the wrapper deliberately
    does NOT impute feature NaNs -- that is the inner's / strategy-layer's job
    (see CLAUDE.md "Frame-type conversions are caller responsibility"), so a
    NaN-feature + NaN-intolerant-inner crash is an inner limitation, not a
    wrapper bug, and would be a false positive.

Metamorphic (``TestMetamorphicInvariants``)
    Relations that must hold between two related fits / predicts regardless of
    the concrete data:
      (1) shifting ``y`` by a constant ``c`` shifts additive-residual
          predictions by ``~c``;
      (2) duplicating rows leaves per-row predictions unchanged;
      (3) a monotone-increasing base perturbation does not flip the sign of a
          ``linear_residual`` prediction delta where ``alpha > 0``;
      (4) clone + refit on the same data is bit-identical for a deterministic
          inner;
      (5) permuting the NON-base feature columns leaves predictions unchanged
          for a base-only inner (the wrapper passes the frame through; the
          invariant is exercised with an order-agnostic inner so a column-order-
          sensitive inner like sklearn ``LinearRegression`` does not mask it).

Runtime is bounded: n capped at 2000, iteration counts kept small, inners are
cheap (``max_iter=15`` HGBR / closed-form linear / constant). The suite found
NO production bug -- the wrapper's domain-fallback + T-clip + y-envelope-clip
machinery already absorbs every adversarial config exercised here.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin, clone

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform, list_transforms

@pytest.fixture(autouse=True)
def _quiet_composite_logger():
    """Quiet the wrapper's INFO/WARNING domain-drop / T-clip chatter so the fuzz
    does not flood the captured log -- snapshot + RESTORE the level so caplog
    warning-assertion tests in other modules are not polluted by a global mute.
    """
    lg = logging.getLogger("mlframe.training.composite")
    prev = lg.level
    lg.setLevel(logging.CRITICAL)
    yield
    lg.setLevel(prev)


sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
HistGradientBoostingRegressor = sklearn_ensemble.HistGradientBoostingRegressor

# Transforms whose base is a K-column matrix (need ``base_columns``).
_MULTI_BASE = {
    "linear_residual_multi",
    "geometric_mean_residual",
    "pairwise_interaction_residual",
}
# Transforms that consume a group / category column (need ``group_column``).
_GROUPED = {"linear_residual_grouped", "target_encoding_residual"}
# Transforms that require strictly-positive y and/or base on every row.
_POSITIVE = {
    "logratio",
    "ratio",
    "geometric_mean_residual",
    "reciprocal_residual",
}
# Additive-residual cores: inverse is ``y = T + (base-only term)``, so a
# constant y-shift carries through to the prediction one-for-one.
_ADDITIVE_SHIFT = ["diff", "linear_residual", "additive_residual", "median_residual"]


# ----------------------------------------------------------------------
# Inners
# ----------------------------------------------------------------------
class _BaseOnlyInner(BaseEstimator, RegressorMixin):
    """Deterministic, column-order-agnostic inner: predicts the constant
    mean of the T-scale target, ignoring every feature.

    Used by the metamorphic tests that must isolate the WRAPPER's
    base-only arithmetic from any feature-order sensitivity of a real
    inner (sklearn ``LinearRegression`` enforces fit-time column order and
    would mask the feature-permutation invariant)."""

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self._mean_t, dtype=np.float64)


# ----------------------------------------------------------------------
# Data factory
# ----------------------------------------------------------------------
def _make_config_frame(
    n: int,
    seed: int,
    transform: str,
    nan_frac: float,
    inject_outliers: bool,
    n_feat: int,
    dtype: str,
):
    """Build (X, y, wrapper_kwargs) for a random fuzz config.

    Bases are drawn strictly-positive for ``_POSITIVE`` transforms (else most
    rows would be domain-dropped and the fit would be trivial / empty). y is a
    linear function of base + features plus noise, forced positive for the
    positive-domain transforms. Feature NaNs are injected only into the plain
    feature columns (never the base / group plumbing columns) so the base
    extraction + domain check stay well-defined.
    """
    rng = np.random.default_rng(seed)
    t = get_transform(transform)
    positive = transform in _POSITIVE
    n_base = 2 if transform in _MULTI_BASE else 1

    cols: dict[str, np.ndarray] = {}
    bases = []
    for j in range(n_base):
        b = rng.uniform(0.5, 5.0, n) if positive else rng.normal(2.0, 1.5, n)
        cols[f"b{j}"] = b
        bases.append(b)
    for j in range(n_feat):
        cols[f"f{j}"] = rng.normal(0.0, 1.0, n)
    if transform in _GROUPED:
        cols["grp"] = rng.integers(0, 5, n).astype(str)

    y = (
        1.3 * bases[0]
        + 0.5 * sum(cols[f"f{j}"] for j in range(n_feat))
        + rng.normal(0.0, 0.3, n)
    )
    if positive:
        y = np.abs(y) + 0.5
    if inject_outliers:
        idx = rng.choice(n, max(1, n // 50), replace=False)
        y[idx] = y[idx] * 50.0

    X = pd.DataFrame(cols)
    if nan_frac > 0.0:
        for j in range(n_feat):
            mask = rng.random(n) < nan_frac
            X.loc[mask, f"f{j}"] = np.nan
    if dtype == "float32":
        for c in X.columns:
            if X[c].dtype.kind == "f":
                X[c] = X[c].astype(np.float32)

    kwargs: dict = {}
    if transform in _MULTI_BASE:
        kwargs["base_columns"] = [f"b{j}" for j in range(n_base)]
    elif t.requires_base:
        kwargs["base_column"] = "b0"
    if transform in _GROUPED:
        kwargs["group_column"] = "grp"
    return X, y, kwargs


# ----------------------------------------------------------------------
# Fuzz
# ----------------------------------------------------------------------
class TestFuzzFitPredictNeverCrashes:
    """fit -> predict over random configs never raises and yields finite y in envelope."""

    @pytest.mark.parametrize("batch", range(6))
    def test_fuzz_batch(self, batch: int) -> None:
        # Each batch is an independent RNG stream so a flake is reproducible
        # from (batch, iteration). 50 iters/batch * 6 batches = 300 configs.
        rng = np.random.default_rng(10_000 + batch)
        names = list_transforms()
        n_checked = 0
        for it in range(50):
            seed = int(rng.integers(0, 1_000_000))
            transform = names[int(rng.integers(0, len(names)))]
            n = int(rng.integers(20, 2001))
            n_feat = int(rng.integers(1, 6))
            nan_frac = float(rng.choice([0.0, 0.0, 0.05, 0.2]))
            inject_outliers = bool(rng.integers(0, 2))
            dtype = str(rng.choice(["float64", "float32"]))

            X, y, kwargs = _make_config_frame(
                n, seed, transform, nan_frac, inject_outliers, n_feat, dtype,
            )
            est = CompositeTargetEstimator(
                base_estimator=HistGradientBoostingRegressor(
                    max_iter=15, random_state=0,
                ),
                transform_name=transform,
                **kwargs,
            )
            ctx = (
                f"batch={batch} it={it} transform={transform} n={n} "
                f"n_feat={n_feat} nan_frac={nan_frac} outliers={inject_outliers} "
                f"dtype={dtype} seed={seed}"
            )
            try:
                est.fit(X, y)
            except Exception as exc:  # noqa: BLE001 - fuzz surfaces ALL crashes
                pytest.fail(f"fit crashed [{ctx}]: {type(exc).__name__}: {exc}")

            try:
                y_hat = est.predict(X)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"predict crashed [{ctx}]: {type(exc).__name__}: {exc}")

            assert y_hat.shape == (len(X),), f"shape [{ctx}]"
            assert np.all(np.isfinite(y_hat)), f"non-finite prediction [{ctx}]"

            # Every prediction is either the inverse clipped to the train
            # y-envelope or the fallback median (itself in-envelope), so the
            # whole batch must sit inside [y_clip_low, y_clip_high].
            lo = est.fitted_params_["y_clip_low"]
            hi = est.fitted_params_["y_clip_high"]
            if np.isfinite(lo) and np.isfinite(hi):
                tol = 1e-6 * (1.0 + abs(hi - lo))
                assert np.all(y_hat >= lo - tol) and np.all(y_hat <= hi + tol), (
                    f"prediction escaped train envelope "
                    f"[{ctx}] lo={lo} hi={hi} "
                    f"pmin={y_hat.min()} pmax={y_hat.max()}"
                )
            n_checked += 1
        assert n_checked == 50

    def test_fuzz_out_of_train_base_stays_in_envelope(self) -> None:
        """A grossly out-of-train base shift must still produce finite,
        in-envelope predictions (T-clip + y-clip both engage)."""
        rng = np.random.default_rng(7)
        n = 400
        base = rng.normal(2.0, 1.0, n)
        feat = rng.normal(0.0, 1.0, n)
        y = 1.5 * base + 0.7 * feat + rng.normal(0.0, 0.2, n)
        X = pd.DataFrame({"base": base, "feat": feat})
        est = CompositeTargetEstimator(
            base_estimator=HistGradientBoostingRegressor(max_iter=15, random_state=0),
            transform_name="linear_residual",
            base_column="base",
        ).fit(X, y)
        lo = est.fitted_params_["y_clip_low"]
        hi = est.fitted_params_["y_clip_high"]
        X_wild = X.copy()
        X_wild["base"] = X_wild["base"] + 1_000.0
        y_hat = est.predict(X_wild)
        assert np.all(np.isfinite(y_hat))
        assert np.all(y_hat >= lo - 1e-6) and np.all(y_hat <= hi + 1e-6)


# ----------------------------------------------------------------------
# Metamorphic
# ----------------------------------------------------------------------
class TestMetamorphicInvariants:
    @staticmethod
    def _frame(n: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        base = rng.normal(2.0, 1.0, n)
        f0 = rng.normal(0.0, 1.0, n)
        f1 = rng.normal(0.0, 1.0, n)
        y = 1.5 * base + 0.7 * f0 - 0.4 * f1 + rng.normal(0.0, 0.2, n)
        return pd.DataFrame({"base": base, "f0": f0, "f1": f1}), y

    @pytest.mark.parametrize("transform", _ADDITIVE_SHIFT)
    def test_invariant_1_constant_y_shift_shifts_predictions(self, transform: str) -> None:
        """y -> y + c shifts additive-residual predictions by ~c.

        Additive-residual inverses are ``y = T + (base-only term)``; the
        base-only term is identical between the two fits (same X), and a
        constant y-shift maps to a constant T-shift the constant inner learns
        exactly, so the prediction delta is c on every row (away from any
        envelope-clip saturation, which a moderate c avoids)."""
        X, y = self._frame()
        c = 3.7
        e0 = CompositeTargetEstimator(
            base_estimator=_BaseOnlyInner(), transform_name=transform, base_column="base",
        ).fit(X, y)
        e1 = CompositeTargetEstimator(
            base_estimator=_BaseOnlyInner(), transform_name=transform, base_column="base",
        ).fit(X, y + c)
        delta = e1.predict(X) - e0.predict(X)
        assert np.allclose(delta, c, atol=1e-6), (
            f"{transform}: expected uniform +{c} shift, got "
            f"[{delta.min()}, {delta.max()}]"
        )

    def test_invariant_2_row_duplication_preserves_per_row_predictions(self) -> None:
        """Duplicating rows leaves per-row predictions unchanged (predict is
        stateless / row-independent for a non-recurrent transform)."""
        X, y = self._frame()
        est = CompositeTargetEstimator(
            base_estimator=_BaseOnlyInner(),
            transform_name="linear_residual",
            base_column="base",
        ).fit(X, y)
        p_single = est.predict(X)
        X_dup = pd.concat([X, X], ignore_index=True)
        p_dup = est.predict(X_dup)
        assert np.array_equal(p_dup[: len(X)], p_single)
        assert np.array_equal(p_dup[len(X):], p_single)

    def test_invariant_3_monotone_base_does_not_flip_linear_residual_sign(self) -> None:
        """A monotone-increasing base shift moves a linear_residual prediction
        in the direction of ``sign(alpha)`` on every row (alpha > 0 here)."""
        X, y = self._frame()
        # This invariant probes the linear_residual INVERSE arithmetic (sign of alpha drives the delta).
        # The +5 base shift lands several IQRs beyond the fit range, so the default-ON soft-base-shrink
        # smart-fallback would route those deep-OOD rows to the constant median -- a deliberate,
        # non-monotone OOD guard (covered by the _soft_shrink tests). Disable it here to isolate the inverse.
        est = CompositeTargetEstimator(
            base_estimator=_BaseOnlyInner(),
            transform_name="linear_residual",
            base_column="base",
            soft_base_shrink=False,
        ).fit(X, y)
        alpha = float(est.fitted_params_["alpha"])
        assert alpha > 0.0, "fixture is constructed so OLS alpha > 0"
        X_up = X.copy()
        X_up["base"] = X_up["base"] + 5.0
        delta = est.predict(X_up) - est.predict(X)
        moved = delta[np.abs(delta) > 1e-9]
        assert moved.size > 0, "an upward base shift must move predictions"
        assert np.all(np.sign(moved) == np.sign(alpha)), (
            "monotone-increasing base flipped the sign of the prediction delta "
            f"for alpha={alpha} > 0"
        )

    @pytest.mark.parametrize(
        "transform", ["diff", "linear_residual", "ratio", "logratio"],
    )
    def test_invariant_4_clone_refit_is_bit_identical(self, transform: str) -> None:
        """clone + refit on identical data yields bit-identical predictions for
        a deterministic inner (no hidden RNG / fit-order nondeterminism)."""
        X, y = self._frame()
        if transform in ("ratio", "logratio"):
            # positive-domain: shift base / y strictly positive.
            X = X.copy()
            X["base"] = np.abs(X["base"]) + 0.5
            y = np.abs(y) + 0.5
        est = CompositeTargetEstimator(
            base_estimator=_BaseOnlyInner(), transform_name=transform, base_column="base",
        ).fit(X, y)
        est_clone = clone(est).fit(X, y)
        assert np.array_equal(est.predict(X), est_clone.predict(X)), (
            f"{transform}: clone+refit predictions diverged"
        )

    def test_invariant_5_feature_permutation_preserves_predictions(self) -> None:
        """Permuting the NON-base feature columns leaves predictions unchanged
        for a base-only inner (the wrapper passes the frame through; only the
        base column and the constant inner determine the output)."""
        X, y = self._frame()
        est = CompositeTargetEstimator(
            base_estimator=_BaseOnlyInner(),
            transform_name="linear_residual",
            base_column="base",
        ).fit(X, y)
        p_orig = est.predict(X)
        p_perm = est.predict(X[["f1", "base", "f0"]])
        assert np.array_equal(p_orig, p_perm)
