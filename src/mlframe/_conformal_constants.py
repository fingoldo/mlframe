"""Default miscoverage levels for the conformal interval reports.

The transformer, the reporting config and the calibration phase fallback all report the same levels, so they read this one constant.
"""

DEFAULT_CONFORMAL_ALPHAS: tuple[float, float] = (0.1, 0.2)
