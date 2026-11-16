"""The multi-stage stacker passes its groups and time_aware settings through to the stage-1 out-of-fold split."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import mlframe.training.composite.stacking_multi_stage as sms


def test_fit_forwards_groups_and_time_aware_to_the_oof_split(monkeypatch):
    """Groups supplied to fit reach composite_oof_predictions; without them the old random split is used."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    groups = np.repeat(np.arange(12), 10)
    y_aux = {"aux": X["a"].to_numpy() + rng.normal(scale=0.1, size=n)}
    seen = []
    real = sms.composite_oof_predictions

    def spy(*args, **kwargs):
        """Record the split controls, then run the real out-of-fold computation."""
        seen.append((kwargs.get("groups"), kwargs.get("time_aware")))
        return real(*args, **kwargs)

    monkeypatch.setattr(sms, "composite_oof_predictions", spy)
    stacker = sms.MultiStageMetaFeatureStacker(
        stage1_estimator_factories={"aux": lambda: Ridge()},
        stage2_estimator=Ridge(),
        n_splits=3,
        quantile_transform=False,
    )
    stacker.fit(X, y_primary=rng.normal(size=n), y_auxiliary=y_aux, groups=groups, time_aware=True)
    assert len(seen) == 1
    got_groups, got_time_aware = seen[0]
    np.testing.assert_array_equal(got_groups, groups)
    assert got_time_aware is True
