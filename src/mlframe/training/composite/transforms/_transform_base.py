"""Transform record, its tags, and the registry exceptions.

Leaf module: it imports nothing from this package, so the sibling transform modules can import it at top level
without a cycle through the registry facade (``__init__.py`` re-exports every name).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class UnknownTransformError(KeyError):
    """Raised when a transform name is not in :data:`_TRANSFORMS_REGISTRY`."""


class DomainViolationError(ValueError):
    """Raised at fit time when the input domain is incompatible with the
    chosen transform (e.g. ``logratio`` requested but ``y`` contains
    non-positive values).

    At predict time we do NOT raise -- per-row violations are handled
    via fall-back values + counters logged on
    ``CompositeTargetEstimator.runtime_stats_``.
    """


# ----------------------------------------------------------------------
# Transform registry
# ----------------------------------------------------------------------

# Tags used to filter the registry into presets.
TAG_CORE: str = "core"  # diff / ratio / logratio / linear_residual
TAG_EXTENDED: str = "extended"  # placeholder; future presets may add more
TAG_REGRESSION: str = "regression"


@dataclass(frozen=True)
class Transform:
    """One row of the transform registry.

    The four functions form a contract:

    - ``fit(y_train, base_train)`` -> ``dict`` of transform-specific
      fitted parameters (e.g. ``{"alpha": float, "beta": float}``).
      Pure: must NOT mutate inputs and must NOT close over external
      state. Values may be ndarrays (median/quantile/monotonic/rank/
      smoothing_spline/quantile_normal): serialise via the spec serialiser, NOT raw json.dumps.
    - ``forward(y, base, params)`` -> ``T``: applies the transform.
    - ``inverse(T_hat, base, params)`` -> ``y_hat``: applies the
      inverse. Wrapper additionally clips the output to the y-bounds
      stored alongside ``params``.
    - ``domain_check(y, base)`` -> boolean mask of valid rows. Wrapper
      uses this at fit-time to drop invalid rows BEFORE calling
      ``fit`` / ``forward``, and at predict-time to flag rows where
      the inverse cannot be applied cleanly (those rows fall back to
      ``y_train_median``).
    - ``domain_check_fitted(y, base, params)`` -> boolean mask, OPTIONAL.
      The pre-fit ``domain_check`` cannot see fitted parameters, so any
      transform whose validity depends on a learned parameter (e.g.
      ``log_y``'s ``offset`` -- rows with ``y + offset <= 0`` produce NaN
      under ``log``; ``centered_ratio``'s shift ``c`` -- rows where
      ``base + c`` lands in the near-zero eps-floor band) cannot enforce
      its true domain through ``domain_check`` alone. When a transform
      sets this hook, callers re-evaluate the valid mask AFTER ``fit`` and
      drop the newly-invalid rows so they never reach ``forward`` (which
      would silently emit NaN T). The same ``(y, base)`` -> all-True /
      ``y=None`` predict-time contract as ``domain_check`` applies: with
      ``y=None`` only the base-side, params-derived conditions are gated.
      Transforms without params-dependent validity leave this ``None`` and
      callers fall back to the params-free ``domain_check`` result.
    """

    name: str
    forward: Callable[..., np.ndarray]
    inverse: Callable[..., np.ndarray]
    fit: Callable[..., dict[str, Any]]
    domain_check: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str
    tags: frozenset[str] = field(default_factory=frozenset)
    # Optional fitted-params-aware domain refinement. See the contract note
    # above. ``None`` (default) means the transform's validity is fully
    # captured by the params-free ``domain_check``. When set, the signature
    # is ``(y, base, params) -> boolean mask`` with the same ``y=None``
    # predict-time sentinel handling as ``domain_check``.
    domain_check_fitted: Callable[..., np.ndarray] | None = None
    # Grouped-transform support. When True, the wrapper extracts a
    # ``groups`` array from a configured column and
    # passes it as a keyword argument to ``fit`` / ``forward`` /
    # ``inverse``. Used by ``linear_residual_grouped`` (per-segment alpha
    # with James-Stein shrinkage). Default False: legacy transforms
    # keep their 3-arg signatures and the wrapper never passes groups.
    requires_groups: bool = False
    # Unary y-transform support (Pack J): when False the wrapper skips
    # the base-column extraction step and feeds a placeholder ``None``
    # array as the ``base`` arg to fit / forward / inverse / domain_check.
    # Unary transforms (``cbrt_y``, ``log_y``, ``yeo_johnson_y``,
    # ``quantile_normal_y``) declare ``requires_base=False`` so callers
    # may instantiate ``CompositeTargetEstimator`` without configuring
    # a base column. Default True keeps the legacy bivariate
    # transforms unchanged.
    requires_base: bool = True
    # Time-recurrent forward support. When True, ``forward`` reads each row's
    # value in the context of its NEIGHBOURS in the row sequence (an EWMA
    # carry-forward, a centred rolling window, a fractional-difference weight
    # tail). For those transforms the fit-time domain filter must NOT compact
    # the sequence before the forward: dropping a domain-violating row would
    # close the gap and shift every later (or windowed) row's recurrence state,
    # so T near a filtered gap would differ from the predict-time T computed on
    # the uncompacted frame. The wrapper therefore runs the forward over the
    # FULL (uncompacted) y / base sequence and masks to the valid rows AFTER,
    # exactly mirroring the predict path (which never compacts). Default False:
    # the 30+ pointwise transforms see their forward applied to the already-
    # filtered rows, which is bit-identical to the full-then-mask order for
    # them. Set True only for the recurrent family: ``ewma_residual`` /
    # ``rolling_quantile_ratio`` (+ centered) / ``frac_diff`` /
    # ``volatility_normalized_residual`` and their ``*_grouped`` variants.
    recurrent: bool = False
    # Out-of-fold train forward: ``forward`` returns a DIFFERENT T for the exact
    # rows the params were fitted on than for any other rows, because a fitted
    # statistic would otherwise include each train row's own y (the classic
    # target-encoding leak; the same split sklearn's TargetEncoder draws between
    # ``fit_transform`` and ``transform``). For such a transform
    # ``inverse(forward(y_train)) != y_train`` by design -- round-trip contracts
    # must use rows outside the fit batch. Default False: every other transform's
    # forward depends only on its arguments' values.
    oof_train_forward: bool = False
    # Additive in T: the inverse is ``y = T + g(base, params)`` with g independent of T, so a T-space error equals the y-space error row
    # by row. The wrap-pass watchdog's ``MAE_T == MAE_y`` invariant holds exactly for these and for no other transform.
    additive_in_t: bool = False
    # Linear in base: additive in T with g linear in the base (``T + alpha . base + beta``), so an out-of-range base extrapolates the
    # prediction linearly; the estimator's soft base shrink guards exactly these. Implies ``additive_in_t``.
    linear_in_base: bool = False
    # Number of base columns the transform is built for: 2 marks the multi-base family, which takes an ``(n, K)`` base matrix with
    # K >= 2 and degrades to a single-column fallback when handed fewer columns (``second_diff`` becomes ``diff``).
    n_bases: int = 1
    # Scale equivariant: fitting on (s*y, s*base) gives T_s = a*T_1 + b and inverse_s(a*T_hat + b) = s * inverse_1(T_hat), so the
    # transform has one shape whatever the target's units. False only where the family is unit-dependent by definition (the
    # Yeo-Johnson (y + 1) ** lam form); a raw-unit constant elsewhere (a log offset of 1.0, arcsinh in raw units) is a defect.
    scale_equivariant: bool = True
    # Base-translation invariant: T is unchanged when every base column is shifted by a constant (the linear / polynomial
    # residual family absorbs it into the intercept). A normal-equations solve on an uncentred design breaks this.
    base_translation_invariant: bool = False
    # The two stages a chain transform composes, ``(residual, unary)``, for the chains that have them. The auto-chain search
    # proposes compositions under generated names (``chain_linear_residual_cbrt``), which is the same transform the default
    # pool ships as ``chain_linres_cbrt``; the stage pair is what tells the two apart from the names.
    chain_stages: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        """A zero sample weight drops the row from an order-free fit (see ``_zero_weight``)."""
        from ._zero_weight import with_zero_weight_drop

        object.__setattr__(self, "fit", with_zero_weight_drop(self.fit, recurrent=self.recurrent))
