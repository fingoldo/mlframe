"""log_only_except wave 3: closes 12 findings across training/feature_handling/registry.py (4,
best-effort provider-release cleanup on eviction/shutdown), training/core/
_phase_train_one_target_model_setup.py (4, best-effort diagnostic/auto-tune enhancement
failures), and training/core/_phase_composite_post_xt_ensemble/__init__.py (4, best-effort
ensemble-build/calibration/charting failures) -- all genuinely non-fatal graceful-degradation
sites where the failure is intentionally never escalated to a caller-visible collection, marked
with the scanner's recognized "best-effort" rationale comment.
"""

from __future__ import annotations

import inspect


def test_registry_release_sites_all_marked_best_effort():
    """Every provider-release except handler in feature_handling/registry.py carries a
    `best-effort` marker."""
    import mlframe.training.feature_handling.registry as registry

    src = inspect.getsource(registry)
    assert src.count("best-effort cleanup") == 4


def test_phase_train_one_target_model_setup_sites_all_marked_best_effort():
    """Every diagnostic/auto-tune except handler carries a `best-effort` marker."""
    import mlframe.training.core._phase_train_one_target_model_setup as setup_mod

    src = inspect.getsource(setup_mod)
    assert src.count("# best-effort:") == 4


def test_phase_composite_post_xt_ensemble_sites_all_marked_best_effort():
    """Every calibration/build/charting except handler carries a `best-effort` marker."""
    import mlframe.training.core._phase_composite_post_xt_ensemble as ensemble_mod

    src = inspect.getsource(ensemble_mod)
    assert src.count("# best-effort:") == 3
    assert "best-effort per-iteration fault isolation" in src
