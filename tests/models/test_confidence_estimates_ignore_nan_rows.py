"""A NaN member prediction does not empty the confident set (audit 2026-11-20, finding 18-F1)."""
import numpy as np

from mlframe.models.ensembling.predict import _confidence_estimates


def test_nan_row_does_not_poison_the_confident_threshold():
    """Rows with finite disagreement stay selectable when one row is NaN; the NaN row is never confident."""
    rng = np.random.default_rng(0)
    preds = rng.uniform(0.2, 0.8, size=(3, 40, 2))
    preds[0, 7, :] = np.nan
    uncertainty, confident = _confidence_estimates(preds, uncertainty_quantile=0.5, normalize_stds_by_mean_preds=False)
    assert confident.size > 0
    assert 7 not in set(confident.tolist())
    assert np.isnan(uncertainty[7])
