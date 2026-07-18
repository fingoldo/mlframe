"""Regression test for the corrective-mechanism-on-by-default flip (2026-07-10, MRMR audit finding #41
follow-up): ``fe_univariate_fourier_adaptive_max_cols`` and ``fe_wavelet_max_cols`` now default to a bounded
value (100), not ``None`` (unlimited).

Both caps were added by an earlier fix already profiling them as the two largest default-ON pre-FE costs
(34% + 26% of the pre-categorize wall at p=420 -- the SAME scale as the audit's motivating production fit),
but shipped with ``None`` (legacy unlimited) defaults, contradicting the project's own "enable corrective
mechanisms by default" rule: the fix existed but nobody hit it without opting in. Reproduced live
(2026-07-10): a p=300 wide-frame biz_val fit with the unbounded default ran long enough to exceed a 900s
outer test timeout in the extra-basis Fourier detector alone. Flipping the default to a bounded value (100 --
preserves full legacy behaviour for any dataset at or under 100 raw columns, the vast majority of tabular
problems) closes the gap without touching either cap mechanism's own tested behaviour (their existing
``test_extra_basis_fe_adaptive_max_cols.py`` / ``test_wavelet_basis_fe_max_cols.py`` suites use p<=8 and
pass unaffected).
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def test_fourier_adaptive_max_cols_defaults_to_bounded_value():
    """Fourier adaptive max cols defaults to bounded value."""
    m = MRMR()
    assert m.fe_univariate_fourier_adaptive_max_cols == 100


def test_wavelet_max_cols_defaults_to_bounded_value():
    """Wavelet max cols defaults to bounded value."""
    m = MRMR()
    assert m.fe_wavelet_max_cols == 100


def test_explicit_none_still_restores_legacy_unbounded_behavior():
    """The pre-2026-07-10 unlimited behaviour remains reachable via an explicit override."""
    m = MRMR(fe_univariate_fourier_adaptive_max_cols=None, fe_wavelet_max_cols=None)
    assert m.fe_univariate_fourier_adaptive_max_cols is None
    assert m.fe_wavelet_max_cols is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
