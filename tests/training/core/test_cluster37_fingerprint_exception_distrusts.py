"""Cluster 37 regression: an unverifiable precomputed bundle is not trusted (audit 2026-11-20, finding 37-F8)."""

from types import SimpleNamespace

import pandas as pd

import mlframe.training.feature_handling.fingerprint as _fp_mod
from mlframe.training.core._main_train_suite_phases import check_precomputed_fingerprint


def test_fingerprint_exception_distrusts_precomputed_bundle(monkeypatch):
    """A fingerprint that raises used to leave the bundle trusted; now it is rejected."""

    def _boom(_df):
        """Stand in for a fingerprint that cannot handle the frame."""
        raise RuntimeError("cannot fingerprint this frame")

    monkeypatch.setattr(_fp_mod, "fingerprint_df", _boom)
    bundle = SimpleNamespace(train_df_fingerprint="deadbeef")
    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert check_precomputed_fingerprint(bundle, frame) is False


def test_matching_fingerprint_keeps_bundle_trusted(monkeypatch):
    """Control: an agreeing fingerprint still trusts the bundle."""

    class _Fp:
        """Minimal fingerprint carrying a fixed short digest."""

        def short(self):
            """Return the fixed digest."""
            return "cafe1234"

    monkeypatch.setattr(_fp_mod, "fingerprint_df", lambda _df: _Fp())
    bundle = SimpleNamespace(train_df_fingerprint="cafe1234")
    assert check_precomputed_fingerprint(bundle, pd.DataFrame({"a": [1]})) is True
