"""Extended multi-target coverage for non-MLP-non-CB-XGB-LGB strategies.

Phase additions 2026-05-07:

- ``Linear`` multi_class kwarg fix: sklearn 1.7 deprecated and 1.8
  removed ``multi_class``. Helper must NOT include it. Test verifies
  the helper output + a smoke fit doesn't crash on sklearn 1.8.
- ``NGBoost`` multiclass via ``Dist=k_categorical(K)``: trainer now
  picks the right Dist instead of the default Bernoulli (which crashed
  on K>2). Test fits 3-class NGBClassifier and asserts the fitted
  model's Dist is a Categorical.
- ``RecurrentModel`` A+B: multiclass + multilabel native via
  ``task_type='multilabel'`` switch. Tests exercise both via
  ``RecurrentClassifierWrapper`` directly (smoke -- no full suite,
  the suite path requires sequence_columns config which is its own
  thing).

LTR for Recurrent is deferred (group-aware sequence batching is its
own epic). RidgeClassifier multilabel is documented as deferred
(``predict_proba`` missing -- eval pipeline assumes it).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------------
# Linear multi_class kwarg fix (sklearn 1.7 deprecated / 1.8 removed)
# ----------------------------------------------------------------------------


class TestLinearMultiClassKwargFix:
    """``_classif_objective_kwargs('linear', MULTICLASS, K)`` must NOT
    include ``multi_class`` (removed from LogisticRegression in 1.8)."""

    def test_helper_output_excludes_multi_class(self):
        from mlframe.training.helpers import _classif_objective_kwargs
        from mlframe.training.configs import TargetTypes

        out = _classif_objective_kwargs(
            "linear",
            TargetTypes.MULTICLASS_CLASSIFICATION,
            4,
        )
        assert "multi_class" not in out, f"helper still emits 'multi_class' (removed in sklearn 1.8): {out}"

    def test_helper_keeps_solver(self):
        """``solver='lbfgs'`` is still meaningful for LR."""
        from mlframe.training.helpers import _classif_objective_kwargs
        from mlframe.training.configs import TargetTypes

        out = _classif_objective_kwargs(
            "linear",
            TargetTypes.MULTICLASS_CLASSIFICATION,
            4,
        )
        assert out.get("solver") == "lbfgs"

    def test_lr_init_with_helper_kwargs_does_not_crash(self):
        """Direct LogisticRegression(**out) must succeed on sklearn 1.8."""
        from mlframe.training.helpers import _classif_objective_kwargs
        from mlframe.training.configs import TargetTypes
        from sklearn.linear_model import LogisticRegression

        out = _classif_objective_kwargs(
            "linear",
            TargetTypes.MULTICLASS_CLASSIFICATION,
            4,
        )
        # Would have crashed pre-fix with TypeError: got an unexpected
        # keyword argument 'multi_class'.
        lr = LogisticRegression(**out, max_iter=10)
        assert lr.solver == "lbfgs"

    def test_linear_multiclass_via_suite_smoke(self):
        """End-to-end: linear+multiclass through the suite produces a
        fitted LR that auto-detects K classes."""
        from mlframe.training import train_mlframe_models_suite, TargetTypes
        from tests.training.shared import SimpleFeaturesAndTargetsExtractor

        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal((n, 5)).astype(np.float32)
        y = np.random.default_rng(43).integers(0, 3, n).astype(np.int64)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y

        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target",
            regression=False,
            target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="lin_mc",
                features_and_targets_extractor=fte,
                mlframe_models=["linear"],
                use_mlframe_ensembles=False,
                verbose=0,
            )
        assert TargetTypes.MULTICLASS_CLASSIFICATION in models


# ----------------------------------------------------------------------------
# NGBoost multiclass via Dist=k_categorical(K)
# ----------------------------------------------------------------------------


pytest.importorskip("ngboost")


class TestNGBoostMulticlass:
    """``NGBClassifier`` defaults to ``Dist=Bernoulli`` (binary only).
    For multiclass the trainer must inject ``Dist=k_categorical(K)``."""

    def test_ngb_fit_with_k_categorical_works_directly(self):
        """Sanity: NGBoost itself accepts k_categorical(3) for K=3 y."""
        from ngboost import NGBClassifier
        from ngboost.distns import k_categorical

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        y = rng.integers(0, 3, 50)

        model = NGBClassifier(Dist=k_categorical(3), n_estimators=2, verbose=False)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50, 3)
        assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)

    def test_ngb_default_dist_crashes_on_K3(self):
        """Regression-guard: confirms the bug we're fixing exists in NGB
        when defaults are kept."""
        from ngboost import NGBClassifier

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))
        y = rng.integers(0, 3, 50)

        # Default Dist=Bernoulli crashes with IndexError on K>2.
        with pytest.raises((IndexError, ValueError)):
            NGBClassifier(n_estimators=2, verbose=False).fit(X, y)

    def test_ngb_multiclass_via_suite(self):
        """End-to-end: trainer auto-picks k_categorical(K) for multiclass."""
        from mlframe.training import train_mlframe_models_suite, TargetTypes
        from tests.training.shared import SimpleFeaturesAndTargetsExtractor

        rng = np.random.default_rng(42)
        n = 400
        X = rng.standard_normal((n, 5)).astype(np.float32)
        score = X[:, 0] + 0.5 * X[:, 1]
        y = np.digitize(score, [np.quantile(score, 1 / 3), np.quantile(score, 2 / 3)]).astype(np.int64)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y

        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target",
            regression=False,
            target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="ngb_mc",
                features_and_targets_extractor=fte,
                mlframe_models=["ngb"],
                use_mlframe_ensembles=False,
                verbose=0,
            )

        # Drill into the fitted model + check Dist is Categorical.
        # k_categorical(K) builds a closure-scoped class named 'Categorical'.
        for target_models in models.values():
            for ns_list in target_models.values():
                for ns in ns_list:
                    inner = getattr(ns, "model", None)
                    if inner is not None and hasattr(inner, "Dist"):
                        dist_name = inner.Dist.__name__
                        assert dist_name == "Categorical", f"NGB Dist {dist_name!r} expected 'Categorical' (via k_categorical(K))"


# ----------------------------------------------------------------------------
# RecurrentModel multiclass + multilabel (Phase A + B)
# ----------------------------------------------------------------------------


pytest.importorskip("lightning")


class TestRecurrentStrategyFlags:
    def test_supports_native_multiclass(self):
        from mlframe.training.strategies import RecurrentModelStrategy

        assert RecurrentModelStrategy().supports_native_multiclass is True

    def test_supports_native_multilabel(self):
        from mlframe.training.strategies import RecurrentModelStrategy

        assert RecurrentModelStrategy().supports_native_multilabel is True

    def test_does_not_support_native_ranking(self):
        """LTR for recurrent is deferred (group-aware sequence
        batching is non-trivial)."""
        from mlframe.training.strategies import RecurrentModelStrategy

        assert RecurrentModelStrategy().supports_native_ranking is False


class TestRecurrentMultilabelDispatch:
    """``RecurrentModelStrategy.get_classif_objective_kwargs`` returns
    ``task_type='multilabel'`` for multilabel target -- consumed by
    RecurrentTorchModel to switch loss + activation."""

    def test_multilabel_returns_task_type_kwarg(self):
        from mlframe.training.strategies import RecurrentModelStrategy
        from mlframe.training.configs import TargetTypes

        out = RecurrentModelStrategy().get_classif_objective_kwargs(
            TargetTypes.MULTILABEL_CLASSIFICATION,
            n_classes=3,
        )
        assert out == {"task_type": "multilabel"}

    def test_binary_and_multiclass_return_empty(self):
        from mlframe.training.strategies import RecurrentModelStrategy
        from mlframe.training.configs import TargetTypes

        for tt in (TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION):
            out = RecurrentModelStrategy().get_classif_objective_kwargs(tt, n_classes=3)
            assert out == {}, f"{tt}: should return empty (defaults correct), got {out}"


class TestRecurrentMultilabelEndToEnd:
    """Smoke: ``RecurrentClassifierWrapper.fit(X, y_2d)`` no longer
    rejects 2-D y; switches to multilabel (sigmoid + BCE) under the hood."""

    def test_2d_y_no_longer_raises_not_implemented(self):
        """Pre-2026-05-07 the wrapper raised NotImplementedError on 2-D y.
        Now it should fit (smoke -- correctness covered by lower tests)."""
        from mlframe.training.neural.recurrent import (
            RecurrentClassifierWrapper,
            RecurrentConfig,
            InputMode,
            RNNType,
        )

        rng = np.random.default_rng(42)
        n = 400
        X = rng.standard_normal((n, 4)).astype(np.float32)
        Y = (rng.standard_normal((n, 3)) > 0).astype(np.int8)
        Y[Y.sum(axis=1) == 0, 0] = 1  # avoid all-zero rows

        cfg = RecurrentConfig(
            input_mode=InputMode.FEATURES_ONLY,
            rnn_type=RNNType.LSTM,
            hidden_size=8,
            num_layers=1,
            mlp_hidden_sizes=(8,),
            num_classes=3,
            max_epochs=3,
            batch_size=32,
            early_stopping_patience=10,
        )
        wrapper = RecurrentClassifierWrapper(config=cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Provide eval_set so OneCycleLR's estimated_stepping_batches > 0.
            # Should NOT raise; pre-fix: NotImplementedError.
            wrapper.fit(
                features=X[:300],
                sequences=None,
                labels=Y[:300],
                eval_set=(X[300:], Y[300:]),
            )
        # Wrapper must record multilabel state.
        assert wrapper._is_multilabel is True
        assert wrapper._n_labels == 3

    def test_predict_proba_returns_per_label_sigmoid(self):
        """For multilabel, predict_proba returns (N, K) sigmoid probs
        (each in [0, 1] independently; rows do NOT sum to 1)."""
        from mlframe.training.neural.recurrent import (
            RecurrentClassifierWrapper,
            RecurrentConfig,
            InputMode,
            RNNType,
        )

        rng = np.random.default_rng(42)
        n = 400
        X = rng.standard_normal((n, 4)).astype(np.float32)
        Y = (rng.standard_normal((n, 3)) > 0).astype(np.int8)
        Y[Y.sum(axis=1) == 0, 0] = 1

        cfg = RecurrentConfig(
            input_mode=InputMode.FEATURES_ONLY,
            rnn_type=RNNType.LSTM,
            hidden_size=8,
            num_layers=1,
            mlp_hidden_sizes=(8,),
            num_classes=3,
            max_epochs=3,
            batch_size=32,
            early_stopping_patience=10,
        )
        wrapper = RecurrentClassifierWrapper(config=cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapper.fit(
                features=X[:300],
                sequences=None,
                labels=Y[:300],
                eval_set=(X[300:], Y[300:]),
            )
            probs = wrapper.predict_proba(features=X[:5], sequences=None)
        assert probs.shape == (5, 3)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        # Multilabel: rows should NOT sum to 1 (independent labels).
        # Allow some leeway if model is ill-trained (accept any value not exactly 1).
        # Just verify they're not the softmax shape.
        assert not np.allclose(probs.sum(axis=1), 1.0, atol=0.05), f"Multilabel rows sum to ~1 -- looks like softmax not sigmoid: {probs.sum(axis=1)}"
