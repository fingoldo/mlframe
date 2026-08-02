"""Shared ``scipy.stats.binom_test`` shim: scipy 1.7+ removed the bare-p-value ``binom_test`` in favor of
``binomtest`` (a result object), and this package (``__init__.py`` / ``_fit_explain.py``) was independently
reimplementing the same forward-compat shim in both places, consolidated here so a fix can't silently drift
out of sync across copies.
"""
from __future__ import annotations

try:
    from scipy.stats import binomtest as _binomtest

    def binom_test(x, n, p, alternative="two-sided"):
        """Shim over SciPy 1.7+ ``binomtest`` matching the removed ``scipy.stats.binom_test`` signature/return (a bare p-value, not the result object)."""
        # SciPy 1.7+ ``binomtest`` requires ``k`` integer; our hit-count vector is float (np.zeros), so coerce on the boundary.
        return _binomtest(int(x), n=int(n), p=p, alternative=alternative).pvalue

except ImportError:  # SciPy < 1.7 fallback
    from scipy.stats import binom_test  # type: ignore  # noqa: F401 -- re-exported module-level name
