"""A failed device-born probe must fall back to the host residue path and say so in the log."""

import logging


def test_failed_device_born_probe_takes_host_path_and_warns(monkeypatch, caplog):
    """A probe that raises selects the host residue path (True) and logs a WARNING naming the failure."""
    from mlframe.feature_selection.filters import _gpu_strict_fe, _pairwise_modular_fe
    from mlframe.feature_selection.filters import _modulus_perm_null as mod

    def _boom():
        """Stand-in for the device-born probe failing."""
        raise RuntimeError("probe broke")

    monkeypatch.setattr(_pairwise_modular_fe, "_resident_scan_enabled", lambda: False)
    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_device_born_modular_enabled", _boom)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        assert mod.host_residue_path() is True
    assert any("device-born probe failed" in rec.getMessage() for rec in caplog.records)
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)
