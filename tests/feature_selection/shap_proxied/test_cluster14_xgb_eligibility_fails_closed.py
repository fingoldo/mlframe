"""An xgboost model whose class count cannot be read is not eligible for the single-margin TreeSHAP path (audit 2026-11-20, finding 14-F6)."""

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import is_supported_xgboost


class _Booster:
    """A booster whose config cannot be read."""

    def save_config(self):
        """Fail the config read."""
        raise RuntimeError("config unavailable")


class _Estimator:
    """An xgboost-like estimator that returns the failing booster."""

    def get_booster(self):
        """Return the booster."""
        return _Booster()


def test_unreadable_class_count_refuses_the_fast_path():
    """Unknown class count used to be eligible; it is now refused."""
    assert is_supported_xgboost(_Estimator()) is False
