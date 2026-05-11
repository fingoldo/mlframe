"""CompositeSpec dataclass: frozen description of one discovered composite target. Extracted into its own module so both composite.py (re-exporter) and composite_discovery.py can import it without circular dependency. No peer dependencies inside the composite_* family."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class CompositeSpec:
    """Frozen description of one discovered composite target.

    ``fitted_params`` is the dict returned by the corresponding
    :class:`Transform`'s ``fit`` (same shape consumed by ``forward`` /
    ``inverse``). Stored so :meth:`CompositeTargetDiscovery.iter_transform`
    can apply the transform to the full frame at integration time, and
    so downstream code can rebuild a :class:`CompositeTargetEstimator`
    with the exact same params used during discovery.

    ``mi_gain`` is the difference in MI between (T, X-without-base) and
    (y, X) -- positive means the transform makes the residual MORE
    predictable from the remaining features (the goal). Negative is
    possible (transform destroyed signal); discovery filters on
    ``eps_mi_gain``.

    ``valid_domain_frac`` is the share of train rows that pass
    ``transform.domain_check``. Discovery filters on
    ``min_valid_domain_frac`` so a transform that only works for ~half
    the rows isn't promoted.
    """

    # Canonical composite name. Format (post-2026-05-13):
    # ``{target_col}-{TRANSFORM_NAME_SHORT[transform_name]}-{base_column}``
    # e.g. ``"TVT-linres-TVT_prev"``. Built via
    # ``composite_transforms.compose_target_name``. The legacy
    # double-underscore format ``"{target}__{transform}__{base}"`` is
    # still recognised by ``is_composite_target_name`` for pickle
    # back-compat but is no longer produced by discovery.
    name: str
    target_col: str
    transform_name: str
    base_column: str
    fitted_params: Dict[str, Any]
    mi_gain: float
    mi_y: float
    mi_t: float
    valid_domain_frac: float
    n_train_rows: int
    # R10c #1 (2026-05-11): multi-base extension. Empty tuple = legacy
    # single-base spec (the ``base_column`` field above is authoritative).
    # When ``len(extra_base_columns) >= 1`` the spec is multi-base; the
    # full base list is ``(base_column,) + tuple(extra_base_columns)``
    # and the wrapper materialises a (n, 1+len(extra)) matrix at
    # predict time. Stored as a tuple so the dataclass remains frozen.
    extra_base_columns: Tuple[str, ...] = ()
