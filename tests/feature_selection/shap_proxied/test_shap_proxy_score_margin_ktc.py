"""The SHAP-proxy score dispatch reads a registered kernel-tuning entry, and its sweep produces regions that pick a side per row count."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_score_margin_ktc as K


class _Spec:
    """Stand-in for the registered entry: answers with a fixed side."""

    def __init__(self, side: str) -> None:
        """Remember the side."""
        self.side = side

    def choose(self, **dims):
        """The tuned side, whatever the row count."""
        return self.side


def test_the_tuned_side_overrides_the_row_mark(monkeypatch):
    """A tuned answer decides over the fallback in both directions."""
    monkeypatch.setattr(K, "_SPEC", _Spec("serial"))
    assert K.use_parallel(10_000_000) is False
    monkeypatch.setattr(K, "_SPEC", _Spec("parallel"))
    assert K.use_parallel(10) is True


def test_without_a_spec_the_row_mark_decides(monkeypatch):
    """No registry: below the mark serial, at it parallel."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_objective as O

    monkeypatch.setattr(K, "_SPEC", None)
    assert K.use_parallel(O._SCORE_MARGIN_PARALLEL_MIN_ROWS - 1) is False
    assert K.use_parallel(O._SCORE_MARGIN_PARALLEL_MIN_ROWS) is True


def test_a_failing_choose_falls_back_to_the_row_mark(monkeypatch):
    """A raising cache never changes the loss, only the route."""

    class _Broken:
        """A spec whose lookup raises."""

        def choose(self, **dims):
            """Raise."""
            raise RuntimeError("cache unreadable")

    monkeypatch.setattr(K, "_SPEC", _Broken())
    assert K.use_parallel(1) is False
    assert K.use_parallel(10**7) is True


def test_the_sweep_runs_and_both_kernels_agree():
    """The sweep itself (small grid) returns regions and the two kernels' losses match to summation order."""
    margin, y = K._sweep_inputs({"n_rows": 20_000})
    assert np.isclose(K._serial(margin, y), K._parallel(margin, y), rtol=1e-9, atol=1e-12)
    regions = K._sweep()
    assert regions, "the sweep returned no regions"
