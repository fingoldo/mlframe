"""Regression: a pipeline / selector reused with a STALE fit width must
recover via fit_transform, not crash.

When a pipeline or feature-selector object is reused across rounds it can carry
fit state for a different input width (e.g. fitted on the FS-reduced output of a
prior model, then handed the raw frame). sklearn then raises
``ValueError: Unexpected input dimension N, expected M`` (or ``X has N features,
but ... is expecting M``) from ``transform``. ``_apply_pre_pipeline_transforms``
already recovers from the NotFittedError / AttributeError flavours of stale
state by re-fitting; the dimension-mismatch ValueError is the same failure mode
and was previously uncaught -> the run crashed (surfaced by fuzz ``c0143`` once
the upstream test-transform skip was fixed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


def test_is_stale_fit_state_value_error_classifies_messages():
    """Is stale fit state value error classifies messages."""
    from mlframe.training.pipeline._pipeline_helpers import _is_stale_fit_state_value_error

    # Width / count mismatch variants.
    assert _is_stale_fit_state_value_error(ValueError("Unexpected input dimension 8, expected 4"))
    assert _is_stale_fit_state_value_error(ValueError("X has 8 features, but SelectKBest is expecting 4 features as input"))
    # Feature-name mismatch variant.
    assert _is_stale_fit_state_value_error(
        ValueError("The feature names should match those that were passed during fit. Feature names seen at fit time, yet now missing: ['a0', 'a1']")
    )
    # Unrelated ValueErrors must NOT be swallowed as stale-fit-state.
    assert not _is_stale_fit_state_value_error(ValueError("could not convert string to float: 'abc'"))
    assert not _is_stale_fit_state_value_error(ValueError("Input contains NaN"))


def test_apply_pre_pipeline_recovers_from_stale_fit_width():
    """Apply pre pipeline recovers from stale fit width."""
    from mlframe.training.pipeline._pipeline_helpers import _apply_pre_pipeline_transforms

    rng = np.random.default_rng(0)
    # Pipeline fitted on 4 features (the stale state from a prior reduced round).
    X4 = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"a{i}" for i in range(4)])
    y4 = (X4["a0"] > 0).astype(int)
    pipe = Pipeline([("pre", SelectKBest(f_classif, k=2))]).fit(X4, y4)
    assert pipe.named_steps["pre"].n_features_in_ == 4

    # Now hand it the RAW 8-feature train (the reuse mismatch). Pre-fix
    # pipe.transform raised "Unexpected input dimension 8, expected 4" (uncaught
    # -> crash); post-fix it is recognized as stale fit state and re-fitted.
    X8 = pd.DataFrame(rng.normal(size=(120, 8)), columns=[f"f{i}" for i in range(8)])
    y8 = (X8["f0"] > 0).astype(int)

    train_df, _val = _apply_pre_pipeline_transforms(
        model=object(),
        pre_pipeline=pipe,
        train_df=X8,
        val_df=None,
        train_target=y8,
        skip_pre_pipeline_transform=False,
        skip_preprocessing=False,
        use_cache=False,
        model_file_name="",
        verbose=False,
    )

    # Recovered (no crash) and re-fitted on the 8-feature frame -> 2 selected cols.
    assert train_df is not None
    assert train_df.shape[0] == 120
    assert pipe.named_steps["pre"].n_features_in_ == 8, "pipeline should have been re-fitted on the 8-feature frame"
