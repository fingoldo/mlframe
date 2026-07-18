"""Sensor for the metrics/core.py monolith split (wave w6b).

Verifies:
- Every previously-importable public name still resolves via the parent facade.
- Identity is preserved (parent.X is sibling.X) for moved symbols.
- Parent facade LOC stays under the 800-line budget.
- Smoke calls into moved bodies (import-only sensors miss runtime NameErrors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PARENT = "mlframe.metrics.core"
FACADE_LOC_BUDGET = 800


def test_metrics_core_facade_loc_budget():
    """Metrics core facade loc budget."""
    import mlframe.metrics.core as parent

    n = len(Path(parent.__file__).read_text(encoding="utf-8").splitlines())
    assert (
        n <= FACADE_LOC_BUDGET
    ), f"{PARENT} grew back over the budget ({n} > {FACADE_LOC_BUDGET}); carve another sibling rather than letting the facade bloat."


def test_metrics_core_re_exports_resolve():
    """All carved public symbols importable from the parent."""
    from mlframe.metrics.core import (  # noqa: F401
        numba_warmup,
        prewarm_numba_cache,
        cb_logits_to_probs_binary,
        cb_logits_to_probs_multiclass,
        fast_roc_auc,
        fast_aucs,
        fast_brier_score_loss,
        brier_and_precision_score,
        make_brier_precision_scorer,
        fast_precision,
        fast_classification_report,
        maximum_absolute_percentage_error,
    )


def test_metrics_core_identity_warmup():
    """Metrics core identity warmup."""
    import mlframe.metrics.core as parent
    from mlframe.metrics import _core_numba_warmup as sib

    assert parent.numba_warmup is sib.numba_warmup
    assert parent.prewarm_numba_cache is sib.prewarm_numba_cache
    assert parent._assert_numba_nogil_active is sib._assert_numba_nogil_active
    assert parent._prewarm_numba_cache_body is sib._prewarm_numba_cache_body


def test_metrics_core_identity_cb_logits():
    """Metrics core identity cb logits."""
    import mlframe.metrics.core as parent
    from mlframe.metrics import _core_cb_logits as sib

    assert parent.cb_logits_to_probs_binary is sib.cb_logits_to_probs_binary
    assert parent.cb_logits_to_probs_multiclass is sib.cb_logits_to_probs_multiclass
    assert parent._cb_logits_to_probs_binary_seq is sib._cb_logits_to_probs_binary_seq
    assert parent._cb_logits_to_probs_binary_par is sib._cb_logits_to_probs_binary_par
    assert parent._cb_logits_to_probs_multiclass_seq is sib._cb_logits_to_probs_multiclass_seq
    assert parent._cb_logits_to_probs_multiclass_par is sib._cb_logits_to_probs_multiclass_par


def test_metrics_core_identity_auc_brier():
    """Metrics core identity auc brier."""
    import mlframe.metrics.core as parent
    from mlframe.metrics import _core_auc_brier as sib

    assert parent.fast_roc_auc is sib.fast_roc_auc
    assert parent.fast_numba_auc_nonw is sib.fast_numba_auc_nonw
    assert parent.fast_aucs is sib.fast_aucs
    assert parent.fast_numba_aucs is sib.fast_numba_aucs
    assert parent.fast_brier_score_loss is sib.fast_brier_score_loss
    assert parent.brier_score_loss is sib.brier_score_loss
    assert parent.brier_and_precision_score is sib.brier_and_precision_score
    assert parent.make_brier_precision_scorer is sib.make_brier_precision_scorer


def test_metrics_core_identity_precision_mape():
    """Metrics core identity precision mape."""
    import mlframe.metrics.core as parent
    from mlframe.metrics import _core_precision_mape as sib

    assert parent.fast_precision is sib.fast_precision
    assert parent.fast_classification_report is sib.fast_classification_report
    assert parent._max_abs_pct_error_kernel is sib._max_abs_pct_error_kernel
    assert parent._max_abs_pct_error_kernel_par is sib._max_abs_pct_error_kernel_par
    assert parent.maximum_absolute_percentage_error is sib.maximum_absolute_percentage_error


def test_metrics_core_smoke_fast_roc_auc():
    """Exercise the moved kernel body so any runtime NameError surfaces."""
    from mlframe.metrics.core import fast_roc_auc

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    val = fast_roc_auc(y_true, y_score)
    assert np.isfinite(val)
    assert val == 1.0  # perfectly separable ordering


def test_metrics_core_smoke_cb_logits_binary():
    """Metrics core smoke cb logits binary."""
    from mlframe.metrics.core import cb_logits_to_probs_binary

    logits = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    probs = cb_logits_to_probs_binary(logits)
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_metrics_core_smoke_classification_report():
    """Metrics core smoke classification report."""
    from mlframe.metrics.core import fast_classification_report

    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 0, 1, 1], dtype=np.int64)
    out = fast_classification_report(y_true, y_pred, nclasses=2)
    # tuple of (hits, misses, accuracy, balanced_accuracy, supports, ...)
    assert len(out) == 10
    accuracy = out[2]
    assert 0.0 <= accuracy <= 1.0


def test_metrics_core_smoke_mape():
    """Metrics core smoke mape."""
    from mlframe.metrics.core import maximum_absolute_percentage_error

    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 3.0, 3.9])
    val = maximum_absolute_percentage_error(y_true, y_pred)
    assert np.isfinite(val)
    assert val > 0.0


def test_metrics_core_external_consumers_still_resolve():
    """Names imported by training.helpers / RFECV / ICE consumers must keep working."""
    from mlframe.metrics.core import (  # noqa: F401
        compute_probabilistic_multiclass_error,
        robust_mlperf_metric,
        ICE,
        create_fairness_subgroups,
    )
