"""Sensor / regression tests for the MLP extreme-AR + group-aware
protection shipped 2026-05-26.

Two protections, two test classes:

1. **Skip-by-default**: ``TrainingBehaviorConfig.mlp_extreme_ar_group_aware_skip``
   default flipped from False to True. The Identity-MLP / LeakyReLU-MLP
   failure mode on extreme-AR (lag1_corr >= 0.99) + group-aware splits
   produces R²<-200 reliably across prod incidents (2026-05-22, -24, -26).
   The skip avoids ~3 min train + 126 MB checkpoint waste; the ensemble's
   dummy-floor gate drops the bad MLP from the blend anyway.

2. **Defensive y-clip**: ``_TTRWithEvalSetScaling.predict`` clips
   inverse-transformed predictions to ``[y_train_min - 3*std,
   y_train_max + 3*std]``. Catches the failure mode when the operator
   opts out of #1 (or runs an MLP outside the suite gate). Bounds the
   damage to "wrong by a few sigma" instead of "wrong by 1000 sigma".
"""
from __future__ import annotations

import numpy as np
import pytest


class TestMlpExtremeArSkipDefault:
    def test_default_is_False(self) -> None:
        """Default OFF: turning MLP off is not the fix. Damage is bounded
        by the TTR predict clip + ensemble dummy-floor gate. User has
        asked the framework to make MLP actually work on extreme-AR +
        group-aware regimes (substantive fix paths in the comment), not
        silently skip."""
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_extreme_ar_group_aware_skip is False
        assert cfg.mlp_extreme_ar_threshold == pytest.approx(0.99)

    def test_can_opt_in(self) -> None:
        """The knob still exists and accepts opt-in for users who
        explicitly want to skip the MLP fit (e.g. tight time budget)."""
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig(mlp_extreme_ar_group_aware_skip=True)
        assert cfg.mlp_extreme_ar_group_aware_skip is True

    def test_threshold_configurable(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig(mlp_extreme_ar_threshold=0.95)
        assert cfg.mlp_extreme_ar_threshold == pytest.approx(0.95)


class TestTtrPredictClip:
    def _fit_ttr(self, y_train: np.ndarray):
        """Helper: fit a TTR with StandardScaler on synthetic y."""
        from sklearn.preprocessing import StandardScaler
        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        class _MockRegressor:
            """sklearn-compat: scaled-space identity on the first feature.
            Used to inject controllable T_hat predictions."""

            def __init__(self):
                self._t_hat_to_return: np.ndarray | None = None

            def fit(self, X, y, **kw):
                self._n_features = X.shape[1] if hasattr(X, "shape") else 1
                return self

            def predict(self, X, **kw):
                if self._t_hat_to_return is not None:
                    return self._t_hat_to_return
                n = len(X) if hasattr(X, "__len__") else X.shape[0]
                return np.zeros(n, dtype=np.float64)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **p):
                return self

        ttr = _TTRWithEvalSetScaling(
            regressor=_MockRegressor(),
            transformer=StandardScaler(),
        )
        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, (len(y_train), 2)).astype(np.float64)
        ttr.fit(X_train, y_train)
        return ttr, X_train

    def test_clip_stats_stashed_after_fit(self) -> None:
        y_train = np.linspace(10500, 12800, 1000)
        ttr, _ = self._fit_ttr(y_train)
        assert hasattr(ttr, "_y_train_clip_low_")
        assert hasattr(ttr, "_y_train_clip_high_")
        # 3*std around the range; should bracket [y_min, y_max].
        assert ttr._y_train_clip_low_ < y_train.min()
        assert ttr._y_train_clip_high_ > y_train.max()

    def test_clip_fires_on_blow_up_predictions(self) -> None:
        """Inject T_hat = 100 sigma (way past 3 sigma envelope). After
        inverse_transform + clip, predictions land at the train-envelope
        boundary, not at the raw 100-sigma value."""
        y_train = np.linspace(10500, 12800, 1000).astype(np.float64)
        ttr, X_train = self._fit_ttr(y_train)
        # In scaled space, mean=0, std=1 (StandardScaler). Inject T_hat=100
        # which after inverse_transform = 100 * std + mean. With y_train
        # std ~ 660 and mean ~ 11650, raw prediction would be ~ 11650 +
        # 66000 = 77650. Clipping caps at y_max + 3 * std ~ 12800 + 1980
        # = 14780.
        n_pred = 50
        ttr.regressor_._t_hat_to_return = np.full(n_pred, 100.0)
        X_pred = X_train[:n_pred]
        preds = ttr.predict(X_pred)
        # Should NOT contain the un-clipped ~77650 value.
        assert preds.max() < ttr._y_train_clip_high_ + 1e-6
        # And the clip should leave preds near (not exactly at because
        # multiple clipped points all stack at the boundary) the high bound.
        assert preds.max() == pytest.approx(ttr._y_train_clip_high_, abs=1.0)

    def test_clip_noop_on_in_distribution_predictions(self) -> None:
        """When T_hat lands in-distribution, clip is a no-op."""
        y_train = np.linspace(10500, 12800, 1000).astype(np.float64)
        ttr, X_train = self._fit_ttr(y_train)
        ttr.regressor_._t_hat_to_return = np.zeros(50)  # mean of scaled = 0
        X_pred = X_train[:50]
        preds = ttr.predict(X_pred)
        # 0 in scaled space ~ y_mean ~ 11650 in y-space; well within
        # [10500-3*std, 12800+3*std].
        assert preds.min() > ttr._y_train_clip_low_
        assert preds.max() < ttr._y_train_clip_high_

    def test_clip_disable_env_var(self, monkeypatch) -> None:
        """Setting MLFRAME_TTR_DISABLE_PREDICT_CLIP=1 disables the clip
        (for benchmarking the failure mode)."""
        monkeypatch.setenv("MLFRAME_TTR_DISABLE_PREDICT_CLIP", "1")
        y_train = np.linspace(10500, 12800, 1000).astype(np.float64)
        ttr, X_train = self._fit_ttr(y_train)
        ttr.regressor_._t_hat_to_return = np.full(50, 100.0)
        preds = ttr.predict(X_train[:50])
        # Without clipping the raw 100-sigma value flows through.
        assert preds.max() > ttr._y_train_clip_high_ + 100


class TestSourceIntegrity:
    """Behavioural sensors so refactors that drop the protection get
    caught early."""

    def test_model_configs_has_skip_default_False(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        field = TrainingBehaviorConfig.model_fields["mlp_extreme_ar_group_aware_skip"]
        assert field.annotation is bool
        assert field.default is False
        assert TrainingBehaviorConfig().mlp_extreme_ar_group_aware_skip is False

    def test_ttr_module_has_y_train_clip(self) -> None:
        from sklearn.preprocessing import StandardScaler
        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        class _Reg:
            def fit(self, X, y, **kw):
                return self
            def predict(self, X, **kw):
                n = len(X) if hasattr(X, "__len__") else X.shape[0]
                return np.zeros(n, dtype=np.float64)
            def get_params(self, deep=True):
                return {}
            def set_params(self, **p):
                return self

        y_train = np.linspace(0.0, 100.0, 200).astype(np.float64)
        rng = np.random.default_rng(0)
        ttr = _TTRWithEvalSetScaling(regressor=_Reg(), transformer=StandardScaler())
        ttr.fit(rng.normal(0, 1, (len(y_train), 2)).astype(np.float64), y_train)
        # Clip envelope attrs are produced at fit time.
        assert hasattr(ttr, "_y_train_clip_low_") and hasattr(ttr, "_y_train_clip_high_")
        assert ttr._y_train_clip_low_ < ttr._y_train_clip_high_

    def test_ttr_predict_clip_disable_env_var_is_honored(self, monkeypatch) -> None:
        """The MLFRAME_TTR_DISABLE_PREDICT_CLIP escape hatch must actually
        change predict behaviour (proves the env var is wired, not just present)."""
        from sklearn.preprocessing import StandardScaler
        from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        class _Reg:
            t_hat = None
            def fit(self, X, y, **kw):
                return self
            def predict(self, X, **kw):
                n = len(X) if hasattr(X, "__len__") else X.shape[0]
                return np.full(n, 100.0) if self.t_hat is None else self.t_hat
            def get_params(self, deep=True):
                return {}
            def set_params(self, **p):
                return self

        y_train = np.linspace(0.0, 100.0, 200).astype(np.float64)
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (len(y_train), 2)).astype(np.float64)

        ttr = _TTRWithEvalSetScaling(regressor=_Reg(), transformer=StandardScaler())
        ttr.fit(X, y_train)
        clipped = ttr.predict(X[:20]).max()
        assert clipped <= ttr._y_train_clip_high_ + 1e-6

        monkeypatch.setenv("MLFRAME_TTR_DISABLE_PREDICT_CLIP", "1")
        ttr2 = _TTRWithEvalSetScaling(regressor=_Reg(), transformer=StandardScaler())
        ttr2.fit(X, y_train)
        assert ttr2.predict(X[:20]).max() > ttr2._y_train_clip_high_ + 100


# ---------------------------------------------------------------------
# FIX 1: bounded output activation (tanh_train_range)
# ---------------------------------------------------------------------


class TestMlpOutputActivationKnob:
    """Knob existence + default = ``linear`` (no behaviour change)."""

    def test_generate_mlp_accepts_output_activation_kwarg(self) -> None:
        import inspect
        from mlframe.training.neural.flat import generate_mlp
        sig = inspect.signature(generate_mlp)
        assert "output_activation" in sig.parameters
        assert "output_activation_scale" in sig.parameters
        assert "output_activation_center" in sig.parameters
        # Default must stay 'linear' for back-compat with every existing
        # generate_mlp caller (suite, ranker, fuzz combos).
        assert sig.parameters["output_activation"].default == "linear"

    def test_default_linear_omits_bounded_head(self) -> None:
        from mlframe.training.neural.flat import (
            generate_mlp, _BoundedTanhOutput,
        )
        model = generate_mlp(num_features=8, num_classes=1, nlayers=2, verbose=0)
        for m in model.modules():
            assert not isinstance(m, _BoundedTanhOutput), (
                "default output_activation='linear' must NOT append "
                "_BoundedTanhOutput to the head."
            )

    def test_unknown_output_activation_raises(self) -> None:
        import pytest
        from mlframe.training.neural.flat import generate_mlp
        with pytest.raises(ValueError, match="Unknown output_activation"):
            generate_mlp(
                num_features=8, num_classes=1, nlayers=2, verbose=0,
                output_activation="sigmoid_unknown_mode",
            )

    def test_tanh_train_range_requires_scale_and_center(self) -> None:
        import pytest
        from mlframe.training.neural.flat import generate_mlp
        with pytest.raises(ValueError, match="output_activation='tanh_train_range'"):
            generate_mlp(
                num_features=8, num_classes=1, nlayers=2, verbose=0,
                output_activation="tanh_train_range",
            )


class TestMlpOutputActivationBoundedBehavior:
    """Behaviour: with tanh_train_range, outputs are HARD-CAPPED to
    ``[center - scale, center + scale]`` regardless of input magnitude."""

    def test_bounded_output_caps_at_window(self) -> None:
        import numpy as np
        import torch
        from mlframe.training.neural.flat import (
            generate_mlp, _BoundedTanhOutput,
        )
        scale = 5.0
        center = 10.0
        model = generate_mlp(
            num_features=4, num_classes=1, nlayers=2, verbose=0,
            output_activation="tanh_train_range",
            output_activation_scale=scale,
            output_activation_center=center,
        )
        # Last module should be the bounded head.
        last = list(model.modules())[-1]
        assert isinstance(last, _BoundedTanhOutput)
        # Inject extreme-magnitude input (well past anything realistic);
        # the bounded head should clamp to [5, 15] regardless.
        rng = np.random.default_rng(42)
        X = torch.tensor(rng.normal(0, 1000, (32, 4)).astype(np.float32))
        with torch.no_grad():
            out = model(X).numpy().reshape(-1)
        assert out.min() >= center - scale - 1e-5
        assert out.max() <= center + scale + 1e-5

    def test_bounded_head_buffers_are_not_trainable(self) -> None:
        from mlframe.training.neural.flat import _BoundedTanhOutput
        head = _BoundedTanhOutput(scale=3.0, center=11.0)
        # Buffers MUST NOT be trainable parameters (no gradients).
        assert "scale" in dict(head.named_buffers())
        assert "center" in dict(head.named_buffers())
        assert "scale" not in dict(head.named_parameters())
        assert "center" not in dict(head.named_parameters())

    def test_source_integrity_bounded_head_present(self) -> None:
        from mlframe.training.neural.flat import generate_mlp, _BoundedTanhOutput
        # _BoundedTanhOutput is importable and the 'tanh_train_range' mode wires it in.
        model = generate_mlp(
            num_features=4, num_classes=1, nlayers=2, verbose=0,
            output_activation="tanh_train_range",
            output_activation_scale=2.0, output_activation_center=1.0,
        )
        assert any(isinstance(m, _BoundedTanhOutput) for m in model.modules())

    def test_bounded_head_forward_and_grad_equal_separate_op_reference(self) -> None:
        """The device-guarded ``addcmul`` fast path (used on CUDA) must produce
        the SAME forward output and the SAME input gradient as the plain
        ``tanh(x) * scale + center`` reference. Runs on every available device
        (CPU always; CUDA when present) so the fusion can never silently change
        the bounded head's numerics or break autograd's tanh gradient."""
        import torch
        from mlframe.training.neural.flat import _BoundedTanhOutput

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            for shape in [(2048, 1), (256, 8)]:
                head = _BoundedTanhOutput(scale=2.5, center=-1.0).to(dev)
                torch.manual_seed(0)
                x = torch.randn(*shape, device=dev) * 7.0

                xa = x.detach().clone().requires_grad_(True)
                out = head(xa)                       # path under test (addcmul on CUDA)
                out.sum().backward()

                xb = x.detach().clone().requires_grad_(True)
                ref = torch.tanh(xb) * head.scale + head.center
                ref.sum().backward()

                assert torch.equal(out, ref), f"forward drift on {dev} {shape}"
                assert torch.equal(xa.grad, xb.grad), f"grad drift on {dev} {shape}"


# ---------------------------------------------------------------------
# FIX 2: drop per-well aggregate columns from MLP input
# ---------------------------------------------------------------------


class TestMlpDropPerGroupConstantsKnob:
    def test_default_is_False(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_drop_per_group_constants is False

    def test_default_pattern(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_drop_per_group_constants_pattern == r"^group_.*_(mean|std|min|max)$"

    def test_knob_overridable(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig(
            mlp_drop_per_group_constants=True,
            mlp_drop_per_group_constants_pattern=r"^rig_.*_(mean|std)$",
        )
        assert cfg.mlp_drop_per_group_constants is True
        assert cfg.mlp_drop_per_group_constants_pattern == r"^rig_.*_(mean|std)$"


class TestMlpDropPerGroupHelpers:
    """Behavior of the per-group detector + drop helpers."""

    def test_identify_per_group_columns_matches_expected(self) -> None:
        from mlframe.training.core._phase_train_one_target_body import (
            _identify_per_group_columns,
        )
        cols = [
            "group_a_mean",  # match
            "group_b_std",   # match
            "group_c_min",   # match
            "group_d_max",   # match
            "feature_x",     # no leading 'group_' prefix
            "depth_m",       # plain feature
            "group_id",      # no trailing reducer
            "GROUP_A_MEAN",  # case-insensitive match
        ]
        dropped = _identify_per_group_columns(
            cols, r"^group_.*_(mean|std|min|max)$",
        )
        assert "group_a_mean" in dropped
        assert "group_b_std" in dropped
        assert "group_c_min" in dropped
        assert "group_d_max" in dropped
        assert "GROUP_A_MEAN" in dropped  # case-insensitive
        assert "feature_x" not in dropped
        assert "depth_m" not in dropped
        assert "group_id" not in dropped

    def test_identify_per_group_columns_empty_cols(self) -> None:
        from mlframe.training.core._phase_train_one_target_body import (
            _identify_per_group_columns,
        )
        assert _identify_per_group_columns([], r"^group_.*_(mean|std|min|max)$") == []
        assert _identify_per_group_columns(None, r"^group_.*_(mean|std|min|max)$") == []

    def test_drop_columns_for_mlp_pandas(self) -> None:
        import pandas as pd
        from mlframe.training.core._phase_train_one_target_body import (
            _drop_columns_for_mlp,
        )
        df = pd.DataFrame({
            "group_a_mean": [1.0, 2.0, 3.0],
            "depth_m": [100.0, 200.0, 300.0],
            "group_b_std": [0.1, 0.2, 0.3],
        })
        out = _drop_columns_for_mlp(df, ["group_a_mean", "group_b_std"])
        assert list(out.columns) == ["depth_m"]
        # Original frame must not be mutated (return-new-frame contract).
        assert "group_a_mean" in df.columns

    def test_drop_columns_for_mlp_polars(self) -> None:
        try:
            import polars as pl
        except ImportError:
            import pytest
            pytest.skip("polars not installed")
        from mlframe.training.core._phase_train_one_target_body import (
            _drop_columns_for_mlp,
        )
        df = pl.DataFrame({
            "group_a_mean": [1.0, 2.0, 3.0],
            "depth_m": [100.0, 200.0, 300.0],
            "group_b_std": [0.1, 0.2, 0.3],
        })
        out = _drop_columns_for_mlp(df, ["group_a_mean", "group_b_std"])
        assert out.columns == ["depth_m"]

    def test_drop_columns_for_mlp_none_passthrough(self) -> None:
        from mlframe.training.core._phase_train_one_target_body import (
            _drop_columns_for_mlp,
        )
        assert _drop_columns_for_mlp(None, ["group_a_mean"]) is None

    def test_drop_columns_for_mlp_missing_cols_no_error(self) -> None:
        import pandas as pd
        from mlframe.training.core._phase_train_one_target_body import (
            _drop_columns_for_mlp,
        )
        df = pd.DataFrame({"depth_m": [1.0, 2.0]})
        # Asking to drop a non-existent column must be a no-op, not error.
        out = _drop_columns_for_mlp(df, ["group_NOT_THERE_mean"])
        assert list(out.columns) == ["depth_m"]


# ---------------------------------------------------------------------
# FIX 3: weight_decay auto-bump on extreme-AR + group-aware
# ---------------------------------------------------------------------


class TestMlpWeightDecayBumpKnob:
    def test_factor_default(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_extreme_ar_weight_decay_factor == pytest.approx(100.0)

    def test_base_default(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_extreme_ar_weight_decay_base == pytest.approx(1e-4)

    def test_knobs_overridable(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig(
            mlp_extreme_ar_weight_decay_factor=50.0,
            mlp_extreme_ar_weight_decay_base=1e-3,
        )
        assert cfg.mlp_extreme_ar_weight_decay_factor == pytest.approx(50.0)
        assert cfg.mlp_extreme_ar_weight_decay_base == pytest.approx(1e-3)


class TestMlpWeightDecayBumpBehavior:
    """Behaviour: the bump helper actually finds the inner estimator,
    mutates its optimizer_kwargs, AND swaps Adam -> AdamW (Adam ignores
    weight_decay)."""

    def _build_mock_inner_estimator(self):
        """Mock object that mimics PytorchLightningEstimator's attrs."""
        import torch
        class _MockMLPInner:
            model_params = {
                "optimizer": torch.optim.Adam,
                "optimizer_kwargs": {},
                "learning_rate": 3e-3,
            }
            network_params = {"nlayers": 4}
        return _MockMLPInner()

    def test_bump_finds_inner_and_mutates(self):
        import torch
        from mlframe.training.core._phase_train_one_target_body import (
            _apply_mlp_extreme_ar_weight_decay_bump,
        )
        inner = self._build_mock_inner_estimator()
        ok = _apply_mlp_extreme_ar_weight_decay_bump(
            inner, factor=100.0, base_weight_decay=1e-4,
        )
        assert ok is True
        # New weight_decay = base * factor = 1e-4 * 100 = 1e-2
        assert inner.model_params["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-2)
        # Adam swapped to AdamW so weight_decay actually applies.
        assert inner.model_params["optimizer"] is torch.optim.AdamW

    def test_bump_walks_ttr_wrapper(self):
        from mlframe.training.core._phase_train_one_target_body import (
            _apply_mlp_extreme_ar_weight_decay_bump,
        )
        inner = self._build_mock_inner_estimator()
        # Mimic _TTRWithEvalSetScaling structure: outer.regressor -> inner.
        class _TTRMock:
            def __init__(self, regressor):
                self.regressor = regressor
        wrapper = _TTRMock(regressor=inner)
        ok = _apply_mlp_extreme_ar_weight_decay_bump(
            wrapper, factor=100.0, base_weight_decay=1e-4,
        )
        assert ok is True
        assert inner.model_params["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-2)

    def test_bump_preserves_existing_decay_when_set(self):
        """User-set weight_decay should be multiplied, not the base."""
        from mlframe.training.core._phase_train_one_target_body import (
            _apply_mlp_extreme_ar_weight_decay_bump,
        )
        inner = self._build_mock_inner_estimator()
        inner.model_params["optimizer_kwargs"]["weight_decay"] = 5e-3
        ok = _apply_mlp_extreme_ar_weight_decay_bump(
            inner, factor=10.0, base_weight_decay=1e-4,
        )
        assert ok is True
        # 5e-3 * 10 = 5e-2 (the prior decay, not the base)
        assert inner.model_params["optimizer_kwargs"]["weight_decay"] == pytest.approx(5e-2)

    def test_bump_returns_false_when_no_inner(self):
        from mlframe.training.core._phase_train_one_target_body import (
            _apply_mlp_extreme_ar_weight_decay_bump,
        )
        # Bare estimator without model_params (e.g. plain sklearn Ridge).
        class _Ridge:
            pass
        ok = _apply_mlp_extreme_ar_weight_decay_bump(
            _Ridge(), factor=100.0, base_weight_decay=1e-4,
        )
        assert ok is False


class TestMlpOutputActivationApplier:
    """Fix 1 applier: sets ``network_params['output_activation']`` on
    the inner estimator, respecting prior explicit user overrides."""

    def test_applier_sets_tanh_train_range(self):
        from mlframe.training.core._phase_train_one_target_body import (
            _apply_mlp_extreme_ar_output_activation,
        )
        class _MockMLPInner:
            network_params = {"nlayers": 4}
        inner = _MockMLPInner()
        ok = _apply_mlp_extreme_ar_output_activation(inner)
        assert ok is True
        assert inner.network_params["output_activation"] == "tanh_train_range"

    def test_applier_respects_explicit_user_override(self):
        from mlframe.training.core._phase_train_one_target_body import (
            _apply_mlp_extreme_ar_output_activation,
        )
        class _MockMLPInner:
            # User explicitly set something non-linear; do not overwrite.
            network_params = {"nlayers": 4, "output_activation": "some_future_mode"}
        inner = _MockMLPInner()
        ok = _apply_mlp_extreme_ar_output_activation(inner)
        assert ok is False
        assert inner.network_params["output_activation"] == "some_future_mode"


class TestPhaseBodySourceIntegrity:
    """Sensors that the per-group drop + weight_decay bump + output-activation
    helpers are present and callable in ``_phase_train_one_target_body``."""

    def test_phase_body_wires_per_group_drop(self) -> None:
        from mlframe.training.core import _phase_train_one_target_body as body
        from mlframe.training._model_configs import TrainingBehaviorConfig
        # Helpers the body uses for the per-group drop are importable + callable.
        assert callable(body._identify_per_group_columns)
        assert callable(body._drop_columns_for_mlp)
        # Wiring point: the gating knob is a real config field.
        assert "mlp_drop_per_group_constants" in TrainingBehaviorConfig.model_fields

    def test_phase_body_wires_weight_decay_bump(self) -> None:
        import torch
        from mlframe.training.core import _phase_train_one_target_body as body
        bump = body._apply_mlp_extreme_ar_weight_decay_bump
        assert callable(bump)

        class _Inner:
            model_params = {"optimizer": torch.optim.Adam, "optimizer_kwargs": {}, "learning_rate": 3e-3}
            network_params = {"nlayers": 4}
        inner = _Inner()
        assert bump(inner, factor=100.0, base_weight_decay=1e-4) is True
        assert inner.model_params["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-2)
        assert inner.model_params["optimizer"] is torch.optim.AdamW

    def test_phase_body_wires_output_activation(self) -> None:
        from mlframe.training.core import _phase_train_one_target_body as body
        apply = body._apply_mlp_extreme_ar_output_activation
        assert callable(apply)

        class _Inner:
            network_params = {"nlayers": 4}
        inner = _Inner()
        assert apply(inner) is True
        assert inner.network_params["output_activation"] == "tanh_train_range"
