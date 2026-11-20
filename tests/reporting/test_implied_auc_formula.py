"""The implied AUC of a Fisher J is Phi(sqrt(J)/sqrt(2)) (audit 2026-11-20, finding 20-F2)."""
from scipy.stats import norm

from mlframe.reporting.charts.engineered_separability import _implied_auc


def test_implied_auc_is_phi_of_sqrt_j_over_sqrt_two():
    """J = 2 gives Phi(1) = 0.8413; the earlier formula gave Phi(0.707) = 0.760."""
    assert abs(_implied_auc(2.0) - norm.cdf(1.0)) < 1e-9
    assert abs(_implied_auc(8.0) - norm.cdf(2.0)) < 1e-9
