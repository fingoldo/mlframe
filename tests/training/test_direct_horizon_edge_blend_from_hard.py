"""Edge blending reads the hard per-bucket predictions, so a narrow bucket is not blended twice (audit 2026-11-20, finding 27-F1)."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from mlframe.training._direct_horizon_bucket_forecaster import DirectHorizonBucketForecaster


def test_narrow_bucket_is_blended_from_its_hard_prediction():
    """Bucket (3, 3) is narrower than 2 * edge_blend_width=4 and lies in two blend zones; each zone must read the hard values.

    Constant models: buckets (1,2) -> 1, (3,3) -> 2, (4,5) -> 3. For horizon 3 the hard value is 2. The pair (3,3)|(4,5)
    gives 0.5 * 2 + 0.5 * 3 = 2.5. The old in-place code re-blended the already-blended 1.5 and returned 2.25 for the
    (1,2)|(3,3) pair, so the final value was not computed from the hard prediction.
    """
    horizon = np.array([1, 2, 3, 4, 5])
    y = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    X = pd.DataFrame({"h": horizon.astype(float)})
    fc = DirectHorizonBucketForecaster([(1, 2), (3, 3), (4, 5)], model_factory=lambda: DummyRegressor(strategy="mean"))
    fc.fit(X, y, horizon)
    out = fc.predict(X, horizon, edge_blend_width=2)
    assert np.isclose(out[2], 2.5)
