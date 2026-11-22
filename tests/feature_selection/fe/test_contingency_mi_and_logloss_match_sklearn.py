"""The plug-in contingency MI and the general log-loss form give scikit-learn's ``mutual_info_score`` and ``log_loss``."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch_sklearn, contingency_mi_nats


@pytest.mark.parametrize("n,ka,kb", [(300, 4, 2), (2000, 10, 3), (50, 7, 7)])
def test_contingency_mi_equals_mutual_info_score(n, ka, kb):
    """Labels of any dtype, tables with empty cells."""
    from sklearn.metrics import mutual_info_score

    rng = np.random.default_rng(n + ka)
    a = rng.integers(0, ka, n)
    b = np.where(rng.random(n) < 0.6, a % kb, rng.integers(0, kb, n))
    assert contingency_mi_nats(a, b) == pytest.approx(float(mutual_info_score(a, b)), abs=1e-12)
    assert contingency_mi_nats(a.astype(str), b) == pytest.approx(float(mutual_info_score(a, b)), abs=1e-12)


def test_empty_and_constant_inputs_give_zero():
    """No rows, or a constant label, carry no information."""
    assert contingency_mi_nats(np.array([]), np.array([])) == 0.0
    assert contingency_mi_nats(np.zeros(20), np.arange(20) % 3) == pytest.approx(0.0, abs=1e-12)


def test_the_reference_batch_path_ranks_a_dependent_column_first():
    """The per-column quantile-bin reference path scores the informative column above noise."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1500, 3))
    y = (x[:, 1] > 0).astype(int)
    mi = _mi_classif_batch_sklearn(x, y, nbins=10)
    assert int(np.argmax(mi)) == 1 and mi[1] > 0.5


def test_the_general_log_loss_form_equals_sklearn_log_loss():
    """The integer-typed probability table goes through the general form: equal to ``log_loss`` over the full label set."""
    from sklearn.metrics import log_loss

    from mlframe.feature_selection.filters._usability_aware_selection import _usability_scorers

    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 400)
    scorers = _usability_scorers(True, y)
    assert scorers is not None
    _mk, logloss, _y_enc, n_classes = scorers
    proba = rng.random((60, n_classes))
    proba = proba / proba.sum(axis=1, keepdims=True)
    truth = rng.integers(0, n_classes, 60)
    assert logloss(truth, proba) == pytest.approx(float(log_loss(truth, proba, labels=np.arange(n_classes))), abs=1e-12)
    # the float path and the integer-typed path agree
    scaled = np.round(proba * 1000).astype(np.int64)
    assert logloss(truth, scaled) == pytest.approx(float(log_loss(truth, scaled / scaled.sum(axis=1, keepdims=True), labels=np.arange(n_classes))), abs=1e-9)
    with pytest.raises(ValueError):
        logloss(truth[:5], scaled[:5, :2])
