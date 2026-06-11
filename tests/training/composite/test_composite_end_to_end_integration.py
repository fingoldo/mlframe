"""End-to-end + property integration suite over the WHOLE composite surface.

Robustness / QA layer that exercises realistic composite-target flows the
per-feature unit tests do not, plus invariants that must hold across the entire
transform registry and every new estimator family. Nothing here adds a transform
or touches production code -- it is a contract net over the public API:

1. ``suggest_discovery_config`` -> ``discover_and_wrap`` -> ``predict`` ->
   ``calibrate_conformal`` -> ``predict_interval`` on a temporal right-skewed
   frame (the canonical "I just want a fitted composite predictor" path).
2. :class:`CompositeQuantileEstimator` -- non-crossing + nominal coverage.
3. :class:`CompositeClassificationEstimator` -- multiclass proba + a finite
   ``calibration_report`` ECE.
4. :class:`CompositeGLMEstimator` -- Poisson predictions are non-negative + finite.
5. :class:`CompositeMultiOutputEstimator` -- ``(n, K)`` shape + finite columns.
6. Property test: for EVERY registry transform, ``fit`` -> ``predict`` on random
   in-domain data yields all-finite ``y`` inside the fitted train envelope.
7. pandas / polars parity across the high-level point-estimator path
   (bit-identical predictions).
8. pickle + ``sklearn.clone`` round-trip of each new estimator family.

Speed: synthetic n <= 800, tiny inners (``max_iter`` / ``n_estimators`` <= 60),
so each test stays well under 5 s. Optional inners (LightGBM) are guarded with
``importorskip`` -- the pinball / GLM / margin families need a gradient-boosting
inner with a raw-margin / pinball path, which the sklearn fallbacks do not give.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from mlframe.training.composite import (
    CompositeClassificationEstimator,
    CompositeGLMEstimator,
    CompositeMultiOutputEstimator,
    CompositeQuantileEstimator,
    CompositeTargetEstimator,
    discover_and_wrap,
    make_per_column_specs,
    suggest_discovery_config,
)
from mlframe.training.composite.transforms import _TRANSFORMS_REGISTRY, get_transform


# ----------------------------------------------------------------------
# Shared synthetic data builders (cheap, deterministic).
# ----------------------------------------------------------------------
def _temporal_skewed_frame(n: int = 800, seed: int = 7):
    """A time-ordered frame whose target depends strongly on a base ``lag``
    column with a right-skewed multiplicative residual -- the regime
    ``discover_and_wrap`` is built for (a dominant affine base + skew tail).

    Returns ``(df, train_idx, holdout_idx)``; the holdout is a disjoint
    forward (later-in-time) slice so the conformal calibration rows are
    exchangeable with the test rows and never seen by the inner.
    """
    rng = np.random.default_rng(seed)
    lag = rng.normal(50.0, 8.0, n)
    skew_resid = np.exp(rng.normal(0.0, 0.3, n))  # lognormal -> right skew
    y = 1.2 * lag + 3.0 * skew_resid + 0.4 * rng.normal(size=n)
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=np.int64),
            "lag": lag,
            "f1": rng.normal(size=n),
            # f2 partly collinear with lag so the base pool has a runner-up.
            "f2": 0.3 * lag + rng.normal(size=n),
            "y": y,
        }
    )
    n_tr = int(0.7 * n)
    return df, np.arange(n_tr), np.arange(n_tr, n)


def _regression_xy(n: int = 700, seed: int = 11):
    """Plain (X, y, base-name) regression problem with a strong affine base."""
    rng = np.random.default_rng(seed)
    lag = rng.normal(40.0, 6.0, n)
    y = 1.1 * lag + rng.normal(0.0, 2.0, n)
    X = pd.DataFrame(
        {"lag": lag, "f1": rng.normal(size=n), "f2": rng.normal(size=n)}
    )
    return X, y, "lag"


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# (1) Full high-level flow: suggest -> discover_and_wrap -> predict ->
#     calibrate_conformal -> predict_interval.
# ----------------------------------------------------------------------
class TestHighLevelDiscoverAndWrapFlow:
    def test_suggest_config_steers_on_skewed_temporal_frame(self) -> None:
        df, _tr, _ho = _temporal_skewed_frame()
        cfg, rationale = suggest_discovery_config(df, "y", ["lag", "f1", "f2"])
        # The suggestion must produce an enabled config and a rationale map.
        assert getattr(cfg, "enabled", False) is True
        assert isinstance(rationale, dict) and rationale
        # mi_sample_n is always steered (band-by-row-count).
        assert "mi_sample_n" in rationale

    def test_discover_predict_calibrate_interval_end_to_end(self) -> None:
        df, tr, ho = _temporal_skewed_frame()
        res = discover_and_wrap(
            df,
            "y",
            ["lag", "f1", "f2"],
            train_idx=tr,
            base_estimator=HistGradientBoostingRegressor(max_iter=50),
            calibrate_conformal=True,
            holdout_idx=ho,
            conformal_alpha=0.1,
        )
        # A spec was discovered + wrapped + calibrated.
        assert res.spec is not None
        assert res.estimator is not None
        assert res.conformal_alpha == pytest.approx(0.1)
        assert isinstance(res.report_markdown, str) and res.report_markdown

        X_ho = df.iloc[ho][["lag", "f1", "f2"]]
        y_ho = df.iloc[ho]["y"].to_numpy()
        point = res.estimator.predict(X_ho)
        assert np.isfinite(point).all()

        lower, upper = res.estimator.predict_interval(X_ho, alpha=0.1)
        # Valid band ordering + coverage near the nominal 1 - alpha. Split
        # conformal guarantees marginal coverage >= 1 - alpha under
        # exchangeability; allow a finite-sample slack downward.
        assert (lower <= upper).all()
        coverage = float(((y_ho >= lower) & (y_ho <= upper)).mean())
        assert coverage >= 0.80, f"conformal coverage {coverage:.3f} < 0.80"

    def test_discover_and_wrap_no_spec_returns_graceful_none(self) -> None:
        # Pure-noise target with no usable base -> discovery yields no spec; the
        # helper must return estimator/spec=None and a non-empty report, not crash.
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n),
                "y": rng.normal(size=n),
            }
        )
        res = discover_and_wrap(
            df, "y", ["a", "b"], train_idx=np.arange(n),
            base_estimator=HistGradientBoostingRegressor(max_iter=20),
        )
        assert res.estimator is None
        assert res.spec is None
        assert isinstance(res.report_markdown, str) and res.report_markdown


# ----------------------------------------------------------------------
# (2) Quantile estimator: non-crossing + coverage.
# ----------------------------------------------------------------------
class TestCompositeQuantileEstimator:
    def test_non_crossing_and_central_coverage(self) -> None:
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMRegressor

        X, y, base = _regression_xy(n=800, seed=21)
        q = CompositeQuantileEstimator(
            base_estimator=LGBMRegressor(n_estimators=80, verbose=-1, n_jobs=1),
            transform_name="linear_residual",
            base_column=base,
            quantiles=[0.1, 0.5, 0.9],
        )
        q.fit(X, y)
        M = q.predict_quantile(X)
        assert M.shape == (len(X), 3)
        # Non-crossing: every row ascending after the enforced sort.
        assert (np.diff(M, axis=1) >= -1e-9).all()
        # The 80% central band [q0.1, q0.9] should cover ~80% of train y; with
        # the affine residual structure this is comfortably attainable.
        cov = float(((y >= M[:, 0]) & (y <= M[:, 2])).mean())
        assert cov >= 0.70, f"central-band coverage {cov:.3f} < 0.70"

    def test_unfitted_predict_quantile_raises(self) -> None:
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMRegressor
        from sklearn.exceptions import NotFittedError

        q = CompositeQuantileEstimator(
            base_estimator=LGBMRegressor(n_estimators=10, verbose=-1),
            base_column="lag",
        )
        with pytest.raises(NotFittedError):
            q.predict_quantile(pd.DataFrame({"lag": [1.0, 2.0]}))


# ----------------------------------------------------------------------
# (3) Classification estimator: multiclass + calibration_report.
# ----------------------------------------------------------------------
class TestCompositeClassificationEstimator:
    def test_multiclass_proba_and_calibration_report(self) -> None:
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier

        rng = np.random.default_rng(31)
        n = 700
        X = pd.DataFrame(
            {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)}
        )
        # Three separable-ish classes from a linear combination + noise.
        score = X["f1"].to_numpy() + 0.5 * X["f2"].to_numpy()
        y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        clf = CompositeClassificationEstimator(
            base_estimator=LGBMClassifier(n_estimators=50, verbose=-1, n_jobs=1)
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (n, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0.0).all() and (proba <= 1.0 + 1e-9).all()
        pred = clf.predict(X)
        assert set(np.unique(pred)).issubset(set(np.unique(y)))

        rep = clf.calibration_report(X, y, n_bins=10)
        assert np.isfinite(rep["ece"])
        assert 0.0 <= rep["ece"] <= 1.0
        assert rep["bin_count"].sum() == n


# ----------------------------------------------------------------------
# (4) GLM estimator: Poisson non-negative + finite.
# ----------------------------------------------------------------------
class TestCompositeGLMEstimator:
    def test_poisson_predictions_nonnegative_finite(self) -> None:
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMRegressor

        rng = np.random.default_rng(41)
        n = 700
        lag = rng.normal(40.0, 6.0, n)
        rate = np.maximum(0.05 * lag, 0.1)
        y = rng.poisson(rate).astype(np.float64)
        X = pd.DataFrame({"lag": lag, "f1": rng.normal(size=n)})
        glm = CompositeGLMEstimator(
            base_estimator=LGBMRegressor(n_estimators=60, verbose=-1, n_jobs=1),
            family="poisson",
        )
        glm.fit(X, y)
        pred = glm.predict(X)
        assert np.isfinite(pred).all()
        assert (pred >= 0.0).all(), "Poisson log-link mean must be non-negative"

    def test_negative_target_rejected(self) -> None:
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMRegressor

        X = pd.DataFrame({"lag": [1.0, 2.0, 3.0, 4.0], "f1": [0.0, 1.0, 0.0, 1.0]})
        glm = CompositeGLMEstimator(
            base_estimator=LGBMRegressor(n_estimators=10, verbose=-1),
            family="poisson",
        )
        with pytest.raises(ValueError):
            glm.fit(X, np.array([1.0, -2.0, 3.0, 4.0]))


# ----------------------------------------------------------------------
# (5) Multi-output estimator: (n, K) shape.
# ----------------------------------------------------------------------
class TestCompositeMultiOutputEstimator:
    def test_vector_target_shape_and_finiteness(self) -> None:
        X, y, base = _regression_xy(n=600, seed=51)
        rng = np.random.default_rng(52)
        Y = np.column_stack([y, 0.8 * X["lag"].to_numpy() + rng.normal(0.0, 1.0, len(X))])
        specs = make_per_column_specs(
            2,
            shared_spec={"transform_name": "linear_residual"},
            base_columns_map={0: base, 1: base},
        )
        mo = CompositeMultiOutputEstimator(
            base_estimator=HistGradientBoostingRegressor(max_iter=40),
            column_specs=specs,
        )
        mo.fit(X, Y)
        P = mo.predict(X)
        assert P.shape == (len(X), 2)
        assert np.isfinite(P).all()
        assert mo.n_outputs_ == 2
        assert not mo.failed_columns_


# ----------------------------------------------------------------------
# (6) Property test over the WHOLE registry.
# ----------------------------------------------------------------------
def _property_frame(transform, n: int = 400, seed: int = 0):
    """Build (X, y) valid for ``transform``: strictly-positive base + target so
    the ratio / log / reciprocal / geometric domains all hold, a string group
    column for grouped transforms, and a multi-base pair for multi-base ones."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(2.0, 9.0, n)  # strictly positive -> ratio/log domains ok
    base2 = rng.uniform(2.0, 9.0, n)
    groups = rng.integers(0, 5, n).astype(str)
    # Strictly-positive target with a real dependence on base so OLS / spline /
    # PCHIP fits are well-posed; +1.0 floor keeps log/reciprocal domains valid.
    y = 1.3 * base + 0.5 * base2 + np.abs(rng.normal(0.0, 0.5, n)) + 1.0
    X = pd.DataFrame(
        {
            "base": base,
            "base2": base2,
            "f1": rng.normal(size=n),
            "grp": groups,
        }
    )
    return X, y


@pytest.mark.parametrize("transform_name", sorted(_TRANSFORMS_REGISTRY))
def test_property_every_transform_fit_predict_finite_in_envelope(transform_name) -> None:
    """For every registered transform: fit -> predict on random in-domain data
    yields all-finite y inside the fitted train envelope ``[y_clip_low,
    y_clip_high]``. This is the global contract every transform must satisfy
    through the estimator wrapper, independent of which one it is."""
    transform = get_transform(transform_name)
    # Stable per-transform seed (process-hash salting would make it flaky):
    # CRC32 of the name is deterministic across runs.
    import zlib

    seed = zlib.crc32(transform_name.encode("utf-8"))
    X, y = _property_frame(transform, seed=seed)

    kwargs: dict = {"transform_name": transform_name}
    if transform.requires_groups:
        kwargs["group_column"] = "grp"
    if transform.requires_base:
        # Multi-base transforms need >= 2 base columns; single-base get one.
        if transform_name in (
            "linear_residual_multi",
            "geometric_mean_residual",
            "pairwise_interaction_residual",
        ):
            kwargs["base_columns"] = ("base", "base2")
        else:
            kwargs["base_column"] = "base"

    est = CompositeTargetEstimator(
        base_estimator=HistGradientBoostingRegressor(max_iter=25, random_state=0),
        **kwargs,
    )
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape[0] == len(X)
    assert np.isfinite(pred).all(), f"{transform_name}: non-finite prediction"

    params = est.fitted_params_
    lo = params.get("y_clip_low", float("-inf"))
    hi = params.get("y_clip_high", float("inf"))
    # The wrapper clips every prediction into the train envelope, so this must
    # hold by construction -- a violation means the clip was bypassed.
    assert (pred >= lo - 1e-6).all(), f"{transform_name}: prediction below envelope"
    assert (pred <= hi + 1e-6).all(), f"{transform_name}: prediction above envelope"


# ----------------------------------------------------------------------
# (7) pandas / polars parity on the high-level point path.
# ----------------------------------------------------------------------
class TestPandasPolarsParity:
    def test_point_predictions_bit_identical_across_flavours(self) -> None:
        pl = pytest.importorskip("polars")
        X, y, base = _regression_xy(n=500, seed=61)
        X_pl = pl.from_pandas(X)

        def _fit_predict(X_train, X_pred):
            est = CompositeTargetEstimator(
                base_estimator=HistGradientBoostingRegressor(
                    max_iter=40, random_state=0
                ),
                transform_name="linear_residual",
                base_column=base,
            )
            est.fit(X_train, y)
            return est.predict(X_pred)

        pred_pd = _fit_predict(X, X)
        pred_pl = _fit_predict(X_pl, X_pl)
        # Same numeric data, same deterministic inner -> bit-identical output.
        assert pred_pd.shape == pred_pl.shape
        np.testing.assert_allclose(pred_pd, pred_pl, rtol=0.0, atol=0.0)


# ----------------------------------------------------------------------
# (8) pickle + clone round-trip of each new estimator family.
# ----------------------------------------------------------------------
def _families_for_roundtrip():
    """Yield ``(estimator, X, y)`` for each new estimator family, choosing a
    valid inner per family. LightGBM-dependent families are skipped when the
    optional dep is absent (the pinball / margin paths need a GBDT inner)."""
    X, y, base = _regression_xy(n=400, seed=71)
    y_pos = np.abs(y) + 1.0
    cases = [
        (
            "CompositeTargetEstimator",
            CompositeTargetEstimator(
                base_estimator=HistGradientBoostingRegressor(max_iter=20),
                transform_name="diff",
                base_column=base,
            ),
            X,
            y_pos,
        ),
        (
            "CompositeMultiOutputEstimator",
            CompositeMultiOutputEstimator(
                base_estimator=HistGradientBoostingRegressor(max_iter=20),
                column_specs={"transform_name": "diff", "base_column": base},
            ),
            X,
            np.column_stack([y_pos, y_pos * 1.1]),
        ),
    ]
    if _has_lightgbm():
        from lightgbm import LGBMClassifier, LGBMRegressor

        rng = np.random.default_rng(72)
        yb = (X["f1"].to_numpy() + rng.normal(0.0, 0.3, len(X)) > 0).astype(int)
        yc = rng.poisson(np.maximum(0.05 * X["lag"].to_numpy(), 0.1)).astype(float)
        cases += [
            (
                "CompositeQuantileEstimator",
                CompositeQuantileEstimator(
                    base_estimator=LGBMRegressor(n_estimators=30, verbose=-1),
                    transform_name="linear_residual",
                    base_column=base,
                    quantiles=[0.25, 0.75],
                ),
                X,
                y_pos,
            ),
            (
                "CompositeGLMEstimator",
                CompositeGLMEstimator(
                    base_estimator=LGBMRegressor(n_estimators=30, verbose=-1),
                    family="poisson",
                ),
                X,
                yc,
            ),
            (
                "CompositeClassificationEstimator",
                CompositeClassificationEstimator(
                    base_estimator=LGBMClassifier(n_estimators=30, verbose=-1)
                ),
                X,
                yb,
            ),
        ]
    return cases


@pytest.mark.parametrize("name,est,X,y", _families_for_roundtrip(), ids=lambda v: v if isinstance(v, str) else "")
def test_pickle_and_clone_roundtrip(name, est, X, y) -> None:
    # Unfitted clone must reproduce the constructor params without error.
    cloned = clone(est)
    assert type(cloned) is type(est)

    est.fit(X, y)

    def _predict(e):
        if hasattr(e, "predict_proba"):
            return e.predict_proba(X)
        return np.asarray(e.predict(X), dtype=np.float64)

    before = _predict(est)
    restored = pickle.loads(pickle.dumps(est))
    after = _predict(restored)
    np.testing.assert_allclose(after, before, rtol=1e-9, atol=1e-9)
