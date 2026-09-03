"""Uniform result object for a Phase-0 feature-selection benchmark arm.

Every arm answers with the same :class:`ArmResult` so the harness can compute the same statistic for
every arm. The critical field is ``score_kind``: ranking metrics (AP, NDCG, AUC-over-features) computed
on a SYNTHESISED per-feature score are silently a different statistic per arm, which invalidates the
comparison. An arm that cannot rank says so (``score=None, score_kind="none"``) instead of emitting a
fake ``selected -> 1 / dropped -> 0`` vector.

The class therefore enforces two invariants at construction time:

* ``support`` is a 1-D boolean array of length ``n_features_in`` -- ALWAYS, for every arm.
* ``score is not None`` exactly when ``score_kind`` is ``"continuous"`` or ``"ordinal"``. A
  ``"selection_order"`` arm (MRMR, forward selection, LARS entry order) has NO score for the features it
  never selected, so it must publish its order in ``ranked_prefix`` rather than pad a score vector.

An arm declared ``continuous`` whose score turns out to be ``None`` at runtime RAISES here; it never
degrades quietly to ``"none"``. That silent-degradation class is the one documented in ``CLAUDE.md``
(the registry's ShapProxiedFS report reader swallowing a failed importance read under a bare ``except``
at ``debug`` level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # Python 3.9 floor: ``typing.Literal`` exists since 3.8, kept explicit for clarity.
    from typing import Literal
except ImportError:  # pragma: no cover - 3.7 and older are not supported by this package
    from typing_extensions import Literal

#: The four honest answers to "what kind of per-feature score can this arm produce?".
SCORE_KINDS: Tuple[str, ...] = ("continuous", "ordinal", "selection_order", "none")

#: Kinds that MUST carry a full-length ``score`` vector; the other two MUST carry ``score=None``.
SCORED_KINDS: Tuple[str, ...] = ("continuous", "ordinal")


@dataclass(frozen=True)
class ArmResult:
    """One arm's answer on one (dataset, seed) cell.

    Attributes:
        support: Boolean mask of length ``n_features_in``, True where the arm kept the feature. Always
            present, for every arm and every ``score_kind``.
        score: Per-feature relevance, higher = more relevant, length ``n_features_in``. ``None`` for
            ``selection_order`` and ``none`` arms. Never synthesised from ``support``.
        score_kind: ``"continuous"`` (a real-valued statistic per feature), ``"ordinal"`` (ranks, ties
            included -- e.g. RFECV's ``ranking_``, which assigns rank 1 to EVERY survivor), ``"selection_order"``
            (only the selected features are ordered; the rest have no score at all) or ``"none"``.
        ranked_prefix: Feature indices in the arm's own selection order, best first. Set for
            ``selection_order`` arms; may also be set by scored arms as a convenience.
        n_features_selected: ``int(support.sum())``, validated against ``support``.
        selection_score: The arm's OWN reported best score (its internal CV optimum), for the
            winner's-curse column. ``None`` when the arm reports no such number.
        wall_time_s: Wall-clock seconds spent in the arm's fit.
        process_time_s: CPU process seconds spent in the arm's fit.
        n_model_fits: Number of downstream model fits the arm consumed, when the arm can count them.
        provenance: Free-form record (seed, kwargs, library versions, resolved backends, why an arm
            degraded). Never load-bearing for the metrics.
    """

    support: np.ndarray
    score: Optional[np.ndarray]
    score_kind: Literal["continuous", "ordinal", "selection_order", "none"]
    ranked_prefix: Optional[Tuple[int, ...]]
    n_features_selected: int
    selection_score: Optional[float]
    wall_time_s: float
    process_time_s: float
    n_model_fits: Optional[int]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the support/score/score_kind contract; raise rather than degrade silently."""
        support = np.asarray(self.support)
        if support.ndim != 1 or support.dtype != np.bool_:
            raise TypeError(f"ArmResult.support must be a 1-D boolean array; got ndim={support.ndim} dtype={support.dtype}.")
        if self.score_kind not in SCORE_KINDS:
            raise ValueError(f"ArmResult.score_kind must be one of {SCORE_KINDS}; got {self.score_kind!r}.")
        if int(support.sum()) != int(self.n_features_selected):
            raise ValueError(f"ArmResult.n_features_selected={self.n_features_selected} disagrees with support.sum()={int(support.sum())}.")
        if self.score_kind in SCORED_KINDS:
            if self.score is None:
                raise ValueError(
                    f"ArmResult declared score_kind={self.score_kind!r} but score is None. A scored arm that lost its "
                    "score is a FATAL error, not a silent degradation to 'none': the ranking metric would be computed "
                    "on a different statistic than for the other arms."
                )
            score = np.asarray(self.score)
            if score.ndim != 1 or score.shape[0] != support.shape[0]:
                raise ValueError(f"ArmResult.score must be 1-D of length {support.shape[0]}; got shape {score.shape}.")
            if not np.all(np.isfinite(score)):
                raise ValueError(f"ArmResult.score declared {self.score_kind!r} contains non-finite values ({int((~np.isfinite(score)).sum())} of {score.size}).")
        elif self.score is not None:
            raise ValueError(
                f"ArmResult declared score_kind={self.score_kind!r} but carries a score vector. Only 'continuous' and "
                "'ordinal' arms may publish a full-length score; a 'selection_order' arm has no score for the features "
                "it never selected, and padding one is exactly the synthesised-score failure this class forbids."
            )
        if self.ranked_prefix is not None:
            n = support.shape[0]
            bad = [i for i in self.ranked_prefix if not (0 <= int(i) < n)]
            if bad:
                raise ValueError(f"ArmResult.ranked_prefix holds out-of-range feature indices {bad[:10]} for n_features_in={n}.")
            if len(set(self.ranked_prefix)) != len(self.ranked_prefix):
                raise ValueError("ArmResult.ranked_prefix must not repeat a feature index.")
        if self.score_kind == "selection_order" and self.ranked_prefix is None:
            raise ValueError("ArmResult declared score_kind='selection_order' but ranked_prefix is None; the order IS the arm's only ranking signal.")

    @property
    def n_features_in(self) -> int:
        """Number of input features the arm saw (the length of ``support``)."""
        return int(np.asarray(self.support).shape[0])

    def selected_indices(self) -> Tuple[int, ...]:
        """Positional indices of the kept features, ascending."""
        return tuple(int(i) for i in np.flatnonzero(np.asarray(self.support)))


__all__ = ["ArmResult", "SCORE_KINDS", "SCORED_KINDS"]
