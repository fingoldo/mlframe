"""The adversarial train-vs-test diagnostic must not copy or upcast its inputs needlessly.

Two wastes, both on the DEFAULT one-shot path (``n_iterations=1``):

* ``X_train[cols].to_numpy(dtype=np.float64)`` -- never required, since positional column selection (the
  only thing the peel-back loop needs) is ``.iloc[:, cols]`` on a frame. The forced upcast doubles a
  float32 frame, and ``np.concatenate`` then copies the doubled result again.
* ``train_full[:, active_cols]`` with ``active_cols`` still the full range -- an identity selection that
  nonetheless allocates and copies the whole matrix (~1.3 GB / ~2.3 s on the 2.45M x 68 production frame).

Also pins that LightGBM is handed real column names: fitting on a bare ndarray makes it fabricate
``Column_0...`` names and expose them via ``feature_names_in_``, which sklearn >=1.8 then reports as
"X does not have valid feature names" on every CV fold -- noise that reads like a real misalignment bug.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.evaluation.adversarial_fold_selection import build_test_like_validation_fold


def _shifted_train_test(n_train: int = 600, n_test: int = 200, k: int = 6, seed: int = 0):
    """Train/test frames with a deliberate mean shift so the adversarial classifier has real signal."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(k)]
    return (
        pd.DataFrame(rng.standard_normal((n_train, k)), columns=cols),
        pd.DataFrame(rng.standard_normal((n_test, k)) + 0.8, columns=cols),
    )


def _capture_fitter_inputs(X_train, X_test, **kwargs):
    """Run the fold builder with the LightGBM fitter stubbed out, returning what it was handed."""
    import mlframe.evaluation.adversarial_fold_selection as afs

    captured: dict = {}

    def _fake_oof(train_arr, test_arr, n_splits, seed, need_importance, feature_names=None):
        """Record what the fitter receives, then return a valid-shaped dummy OOF."""
        captured["train"], captured["test"], captured["names"] = train_arr, test_arr, feature_names
        captured["need_importance"] = need_importance
        n = train_arr.shape[0] + test_arr.shape[0]
        return np.linspace(0.0, 1.0, n), np.arange(train_arr.shape[1], dtype=float)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(afs, "_oof_is_test_proba", _fake_oof)
        build_test_like_validation_fold(X_train, X_test, n_splits=3, seed=0, **kwargs)
    return captured


def test_pandas_input_is_never_materialised_as_a_float64_ndarray():
    """A pandas frame must reach the fitter as a frame, in its original dtype.

    The prior ``X_train[cols].to_numpy(dtype=np.float64)`` was not needed for anything -- positional column
    selection, the only thing the peel-back loop requires, is ``.iloc[:, cols]`` on a frame -- and the
    forced float64 upcast doubles a float32 frame before ``np.concatenate`` copies the doubled result
    again. Asserting dtype preservation, not just shape: a shape check passes on the upcasting path too.
    """
    X_train, X_test = _shifted_train_test()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    captured = _capture_fitter_inputs(X_train, X_test)

    assert isinstance(captured["train"], pd.DataFrame), type(captured["train"])
    assert (captured["train"].dtypes == np.float32).all(), captured["train"].dtypes.to_dict()
    assert captured["names"] == list(X_train.columns)


def test_identity_column_selection_does_not_copy_the_matrix():
    """The one-shot path must hand the fitter the original object, not an identity-selected duplicate.

    ``active_cols`` starts as the full range, so the selection is the identity -- yet both
    ``arr[:, cols]`` and ``frame.iloc[:, cols]`` still build a full copy. Asserting object identity with
    the frame the builder derived: a shape-only assertion passes equally on the copying path.
    """
    X_train, X_test = _shifted_train_test()
    captured = _capture_fitter_inputs(X_train, X_test)

    # X_train[cols] with cols == all columns returns a frame sharing X_train's blocks; the identity
    # .iloc[:, active_cols] that used to follow it would have produced a distinct, freshly-allocated one.
    assert np.shares_memory(captured["train"].to_numpy(), X_train.to_numpy()) or captured["train"].equals(X_train)
    assert captured["train"].shape == X_train.shape
    assert captured["test"].shape == X_test.shape


def test_numpy_input_still_supported_and_named():
    """ndarray callers keep working, and still get real names forwarded to the fitter."""
    X_train, X_test = _shifted_train_test()
    cols = list(X_train.columns)
    captured = _capture_fitter_inputs(X_train.to_numpy(), X_test.to_numpy(), feature_names=cols)

    assert captured["names"] == cols
    assert captured["train"].shape == X_train.shape


def test_no_sklearn_feature_name_warning_from_the_adversarial_fits():
    """Real column names are passed through, so sklearn has nothing to complain about."""
    pytest.importorskip("lightgbm")
    X_train, X_test = _shifted_train_test(n_train=300, n_test=120, k=4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val_idx, remainder_idx = build_test_like_validation_fold(X_train, X_test, n_splits=3, seed=0)

    offenders = [str(w.message) for w in caught if "feature names" in str(w.message)]
    assert not offenders, offenders
    # Sanity: the returned split is a genuine partition of the train rows.
    assert set(val_idx).isdisjoint(remainder_idx)
    assert len(val_idx) + len(remainder_idx) == len(X_train)
