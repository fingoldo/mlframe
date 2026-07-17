"""Regression: ``score_ensemble`` must not crash when member ``val_preds`` /
``test_preds`` are ``None``.

Pre-2026-05-13 the val-side path in ``_process_single_ensemble_method`` did
``el.val_preds.reshape(-1, 1)`` unconditionally; the test-side did the same.
When the suite disables ``compute_valset_metrics`` /
``compute_testset_metrics`` (a legitimate user choice -- only TEST or only
TRAIN metrics wanted), the corresponding ``*_preds`` field on each model
entry stays ``None``, and the ensemble call raised
``AttributeError: 'NoneType' object has no attribute 'reshape'``.

The fix mirrors the existing train-side ``None``-guard at the bottom of
the same function: filter ``None``-valued members out, build the
generator from the surviving members. When NO members have the split's
preds, the ensemble call gets an empty tuple and
``ensemble_probabilistic_predictions`` returns ``(None, None, None)`` as
it already does for the empty case.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.models.ensembling import _process_single_ensemble_method


def _make_member(*, val_preds=None, test_preds=None, train_preds=None, val_probs=None, test_probs=None, train_probs=None):
    """Build a minimal namespace mimicking a trained-model entry."""
    return SimpleNamespace(
        val_preds=val_preds,
        test_preds=test_preds,
        train_preds=train_preds,
        val_probs=val_probs,
        test_probs=test_probs,
        train_probs=train_probs,
        model=None,
        model_name="dummy",
        columns=None,
    )


def _common_kwargs(level_models):
    """The bag of kwargs ``_process_single_ensemble_method`` expects."""
    return dict(
        level_models_and_predictions=level_models,
        is_regression=True,
        ensembling_level=0,
        ensemble_name="test",
        target=None,
        train_idx=None,
        test_idx=None,
        val_idx=None,
        train_target=None,
        test_target=None,
        val_target=None,
        target_label_encoder=None,
        max_mae=0.0,
        max_std=0.0,
        max_mae_relative=2.5,
        max_std_relative=2.5,
        ensure_prob_limits=False,
        nbins=10,
        uncertainty_quantile=0.0,
        normalize_stds_by_mean_preds=False,
        custom_ice_metric=None,
        custom_rice_metric=None,
        subgroups=None,
        n_features=5,
        verbose=False,
        kwargs={},
    )


class TestNonePredsDoesNotCrash:
    def test_all_members_val_preds_none(self) -> None:
        """User sets compute_valset_metrics=False -- every member has
        ``val_preds=None``. Pre-fix this crashed with AttributeError; now
        the val side just returns no ensemble (and the function continues
        through to test + train)."""
        n = 32
        members = [
            _make_member(
                val_preds=None,  # disabled
                test_preds=np.linspace(0.1, 0.9, n),
                train_preds=np.linspace(0.1, 0.9, n * 2),
            ),
            _make_member(
                val_preds=None,
                test_preds=np.linspace(0.2, 0.8, n),
                train_preds=np.linspace(0.2, 0.8, n * 2),
            ),
        ]
        # Should NOT raise.
        method_name, results, conf_results = _process_single_ensemble_method(
            ensemble_method="arithm",
            **_common_kwargs(members),
        )
        assert method_name == "arithm"
        assert results is not None  # build succeeded for the available splits

    def test_mixed_val_preds_some_none(self) -> None:
        """One member has val_preds, another doesn't. The surviving member
        feeds the val ensemble; no crash."""
        n = 32
        members = [
            _make_member(
                val_preds=np.linspace(0.1, 0.9, n),
                test_preds=np.linspace(0.1, 0.9, n),
            ),
            _make_member(
                val_preds=None,  # one missing val
                test_preds=np.linspace(0.2, 0.8, n),
            ),
        ]
        method_name, results, _ = _process_single_ensemble_method(
            ensemble_method="arithm",
            **_common_kwargs(members),
        )
        assert method_name == "arithm"
        assert results is not None

    def test_all_test_preds_none(self) -> None:
        """User sets compute_testset_metrics=False -- the test-side
        guard activates symmetrically."""
        n = 32
        members = [
            _make_member(
                val_preds=np.linspace(0.1, 0.9, n),
                test_preds=None,  # disabled
                train_preds=np.linspace(0.1, 0.9, n * 2),
            ),
            _make_member(
                val_preds=np.linspace(0.2, 0.8, n),
                test_preds=None,
                train_preds=np.linspace(0.2, 0.8, n * 2),
            ),
        ]
        method_name, results, _ = _process_single_ensemble_method(
            ensemble_method="arithm",
            **_common_kwargs(members),
        )
        assert method_name == "arithm"
        assert results is not None

    def test_full_preds_baseline_still_works(self) -> None:
        """Sanity: with all preds present, the function still produces a
        non-None val ensemble (no regression from the fix)."""
        n = 32
        members = [
            _make_member(
                val_preds=np.linspace(0.1, 0.9, n),
                test_preds=np.linspace(0.1, 0.9, n),
                train_preds=np.linspace(0.1, 0.9, n * 2),
            ),
            _make_member(
                val_preds=np.linspace(0.2, 0.8, n),
                test_preds=np.linspace(0.2, 0.8, n),
                train_preds=np.linspace(0.2, 0.8, n * 2),
            ),
        ]
        method_name, results, _ = _process_single_ensemble_method(
            ensemble_method="arithm",
            **_common_kwargs(members),
        )
        assert method_name == "arithm"
        assert results is not None
        # When val_preds are present, val_ensembled_predictions should be a
        # concrete array embedded in the result. We don't peek at the exact
        # internal shape (depends on whether build_predictive_kwargs is in
        # scope), only that the call completed and emitted a result object.
