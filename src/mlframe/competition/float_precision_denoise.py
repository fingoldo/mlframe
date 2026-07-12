"""Float-precision denoising for Kaggle-style anonymized/scaled numeric columns.

COMPETITION / EXPLORATORY ONLY — NEVER wire this into production defaults.

This module consolidates three near-duplicate Kaggle writeups from the
mlframe competition-tricks tracker (MLFRAME_IDEAS_competitions.md):

    * "Denoising integer-encoded floats via floor(x*100) before feature
      engineering" (3rd_amex-default-prediction.md)
    * "Rounding-based denoising + recovering true integer denominator of
      scaled floats" (3rd_bnp-paribas-cardif-claims-management.md)
    * "Denoising raw floats by truncation to remove synthetic competition
      noise ('Amex trick')" (3rd_amex-default-prediction.md)

All three describe the same underlying phenomenon: a competition host takes
an originally coarse-grained value (an integer, or a value with few decimal
places) and injects small proportional/additive float noise on top of it
before publishing the anonymized column, to make leaked precision harder to
exploit. The fix is the same in all three writeups: find the scale factor
(power-of-10 multiplier, or general small-integer denominator) that makes
``series * scale`` closest to integers, then floor/round and rescale back.

This is explicitly a reverse-engineering trick against DELIBERATE
anonymization noise that Kaggle hosts inject. Real production sensor/
measurement noise is not artificially injected in this exploitable pattern,
so this utility has no production use case. It lives under
``mlframe.competition`` and must never be imported by production mlframe
modules or exported from mlframe's top-level ``__init__.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["FloatPrecisionDenoiser", "DenoiseResult"]


class DenoiseResult:
    """Result of a :class:`FloatPrecisionDenoiser` fit/transform.

    COMPETITION / EXPLORATORY ONLY — see module docstring.
    """

    __slots__ = ("denominator", "residual_score", "denoised")

    def __init__(self, denominator: float, residual_score: float, denoised: npt.NDArray[np.float64]) -> None:
        self.denominator = denominator
        self.residual_score = residual_score
        self.denoised = denoised


class FloatPrecisionDenoiser:
    """Recovers a coarse-grained (integer/low-decimal) true value hidden behind injected float noise.

    COMPETITION / EXPLORATORY ONLY — NEVER wire into production defaults.
    This reverses DELIBERATE anonymization noise that some Kaggle hosts
    (Amex, BNP Paribas) inject on top of originally low-precision columns
    to obscure leaked precision. Real production measurement noise is not
    artificially injected in this exploitable pattern, so this class has
    no legitimate production use.

    Algorithm
    ---------
    Given a 1-D numeric array ``x`` suspected of being
    ``round(true_value * denominator) / denominator + noise`` for some
    unknown small ``denominator`` (a power of 10, or any small integer up
    to ``max_denominator``), this searches candidate denominators and picks
    the one that minimizes the mean fractional residue of ``x * denominator``
    after rounding — i.e. the denominator under which ``x`` looks "most
    integer". The denoised series is then ``round(x * denominator) / denominator``.

    Candidates searched: powers of 10 in ``[1, 10**max_decimal_pow]`` and
    every integer in ``[1, max_denominator]`` (union, deduplicated) — this
    covers both the Amex ``floor(x * 100)`` case and the BNP "unknown
    integer denominator" case in one pass.
    """

    def __init__(self, max_decimal_pow: int = 6, max_denominator: int = 1000, use_floor: bool = False) -> None:
        if max_decimal_pow < 0:
            raise ValueError("max_decimal_pow must be >= 0")
        if max_denominator < 1:
            raise ValueError("max_denominator must be >= 1")
        self.max_decimal_pow = max_decimal_pow
        self.max_denominator = max_denominator
        self.use_floor = use_floor
        self.denominator_: float = 1.0
        self.residual_score_: float = np.inf

    def _candidate_denominators(self) -> npt.NDArray[np.float64]:
        pows = np.power(10.0, np.arange(0, self.max_decimal_pow + 1, dtype=np.float64))
        ints = np.arange(1, self.max_denominator + 1, dtype=np.float64)
        return np.unique(np.concatenate([pows, ints]))

    def _residual_for_denominator(self, x: npt.NDArray[np.float64], denominator: float) -> float:
        scaled = x * denominator
        if self.use_floor:
            frac = scaled - np.floor(scaled)
            # circular residue: distance to the nearest integer from below (floor semantics)
            return float(np.mean(frac))
        frac = np.abs(scaled - np.round(scaled))
        return float(np.mean(frac))

    def fit(self, series: npt.ArrayLike) -> "FloatPrecisionDenoiser":
        """Search candidate denominators and store the best one on the instance.

        COMPETITION / EXPLORATORY ONLY — see class docstring.
        """
        x = np.asarray(series, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("series has no finite values to fit on")

        best_denominator = 1.0
        best_score = np.inf
        for denominator in self._candidate_denominators():
            score = self._residual_for_denominator(x, denominator)
            # prefer the smallest denominator among near-ties, to avoid overfitting
            # to spuriously large denominators that trivially minimize residue
            if score < best_score - 1e-12:
                best_score = score
                best_denominator = denominator
        self.denominator_ = best_denominator
        self.residual_score_ = best_score
        return self

    def transform(self, series: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Apply the fitted denoising scale to `series`, returning the recovered coarse values.

        COMPETITION / EXPLORATORY ONLY — see class docstring.
        """
        x = np.asarray(series, dtype=np.float64)
        scaled = x * self.denominator_
        rounded = np.floor(scaled) if self.use_floor else np.round(scaled)
        return rounded / self.denominator_

    def fit_transform(self, series: npt.ArrayLike) -> DenoiseResult:
        """Fit and transform in one call, returning a :class:`DenoiseResult`.

        COMPETITION / EXPLORATORY ONLY — see class docstring.
        """
        self.fit(series)
        denoised = self.transform(series)
        return DenoiseResult(denominator=self.denominator_, residual_score=self.residual_score_, denoised=denoised)
