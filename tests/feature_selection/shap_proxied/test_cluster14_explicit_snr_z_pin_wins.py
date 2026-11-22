"""Cluster 14 regression: an explicit su_seeded_snr_z pin wins over the auto tightening (audit 2026-11-20, finding 14-F1)."""

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_methods import (
    _AUTO_SNR_Z_DEFAULT,
    _AUTO_SNR_Z_TIGHTENED,
    ShapProxiedMethodsMixin,
)


class _Est(ShapProxiedMethodsMixin):
    """Bare mixin host carrying only the attributes _su_screen_snr_z reads."""

    def __init__(self, proxy_mode, su_seeded_interactions, su_seeded_snr_z):
        self.proxy_mode = proxy_mode
        self.su_seeded_interactions = su_seeded_interactions
        self.su_seeded_snr_z = su_seeded_snr_z


def test_explicit_three_under_auto_is_kept():
    """An explicit 3.0 under proxy_mode='auto' must not be replaced by the tightened 4.5 (14-F1)."""
    assert _Est("auto", False, 3.0)._su_screen_snr_z() == 3.0


def test_unset_under_auto_gets_the_tightened_gate():
    """Control: the unset default still tightens under auto."""
    assert _Est("auto", False, None)._su_screen_snr_z() == _AUTO_SNR_Z_TIGHTENED


def test_unset_with_explicit_interactions_keeps_historical_default():
    """Control: su_seeded_interactions=True with the unset value keeps the historical 3.0."""
    assert _Est("auto", True, None)._su_screen_snr_z() == _AUTO_SNR_Z_DEFAULT
