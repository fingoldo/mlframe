"""Cluster 25 regression: the Miller-Madow occupancy excludes the non-finite slot (audit 2026-11-20, finding 25-F67)."""

import numpy as np

import mlframe.metrics._drift as drift


def test_bias_term_counts_histogram_bins_only(monkeypatch):
    """The occupied-bin count passed to the correction omits the trailing non-finite slot (25-F67)."""
    seen = {}
    real = drift._kl_mm_bias

    def _spy(pc, qc, np_tot, nq_tot):
        """Record what the correction receives, then delegate."""
        seen["pc_len"] = len(pc)
        seen["qc_len"] = len(qc)
        return real(pc, qc, np_tot, nq_tot)

    monkeypatch.setattr(drift, "_kl_mm_bias", _spy)
    p = np.array([0.1, 0.2, 0.3, 0.4, np.nan, np.nan])
    q = np.array([0.15, 0.25, 0.35, 0.45, 0.5, 0.6])
    drift.kl_divergence(p, q, nbins=4, bias_correction=True)
    assert seen["pc_len"] == seen["qc_len"] == 4
