"""Warm-start / incremental discovery on appended data.

Pure orchestration over the existing screening MI primitives -- NO new numeric
path. Given a PRIOR :class:`CompositeTargetDiscovery` result (its kept
``CompositeSpec`` list + the ``data_signature`` it was fit on) and a NEW
(appended) frame, decide cheaply whether the prior specs still hold:

* **REUSE** -- re-score each kept spec's MI gain on a small sample of the new
  data; if a sufficient fraction still clear ``eps_mi_gain`` the prior specs
  are returned unchanged (skips the whole O(F * transforms) screen).
* **REDISCOVER** -- the re-scored gains have decayed below threshold (the
  data-generating process drifted), so the caller should run a full
  :meth:`CompositeTargetDiscovery.fit`.

The win: incremental growth (a frame that gains rows under the SAME DGP) pays
only K cheap per-spec MI re-scores instead of re-screening every (base,
transform) pair. When the DGP shifts, the same cheap probe TRIGGERS a full
re-discovery -- no silent reuse of stale specs.

Leakage / RAM discipline (CRITICAL)
-----------------------------------
* The re-score reads only a bounded ``sample_n`` row sample of the new frame
  via ``_extract_column_array(df, col, rows=idx)`` -- O(sample_n) per column,
  never a whole-frame materialisation (the 100+ GB-frame rule).
* The spec's ``fitted_params`` are REUSED as-is (they were fit train-only on
  the prior data); the re-score only forwards them through the new sample and
  measures MI. We never refit on the new rows here -- that is the full-fit
  path's job once drift is confirmed.
* No frame copy / clone anywhere -- only narrow column pulls + a row index
  sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..spec import CompositeSpec
from ..cache import data_signature
from ..transforms import get_transform
from .screening import (
    _extract_column_array,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
    _sample_indices,
)

logger = logging.getLogger(__name__)

# Default fraction of prior specs that must still clear ``eps_mi_gain`` on the
# new sample for a REUSE verdict. A spec dropping below threshold means its
# transform no longer opens up the residual on the new data -- if MOST specs
# survive the kept set is still good; if too many decay the DGP drifted and a
# full re-discovery is warranted. 0.5 (majority) is the conservative default:
# one unlucky spec decaying does not force a full re-screen, but a coordinated
# decay across half the set does.
_DEFAULT_MIN_SURVIVING_FRACTION: float = 0.5

# Re-score sample size. The MI re-score is noise-bounded by sampling + binning,
# so a few-thousand-row sample of the new data is enough to detect a real gain
# decay while keeping the probe O(K * sample_n) rather than O(K * n_new).
_DEFAULT_INCREMENTAL_SAMPLE_N: int = 4000


@dataclass
class IncrementalDecision:
    """Verdict of :func:`incremental_discovery_check`.

    ``reuse`` -- True when the prior specs still hold (caller skips full fit).
    ``specs`` -- the spec list to use: the prior specs unchanged on reuse, or
    ``None`` when a full re-discovery is required (caller must call ``fit``).
    ``reason`` -- short human-readable cause of the verdict.
    ``per_spec_gain`` -- re-scored MI gain per spec name (NaN when the spec
    could not be scored on the new sample, e.g. its base column vanished or
    too few rows survived the domain filter).
    ``n_surviving`` / ``n_specs`` -- specs clearing ``eps_mi_gain`` vs total.
    ``prior_signature`` / ``new_signature`` -- the data signatures compared;
    equal signatures short-circuit to a trivial reuse (data is byte-identical).
    ``n_rescored`` -- how many per-spec MI re-scores actually ran (0 on the
    signature-identical short-circuit). A cheap call-count proxy the biz_value
    test asserts against a full re-discovery's per-(base, transform) count.
    """

    reuse: bool
    specs: list[CompositeSpec] | None
    reason: str
    per_spec_gain: dict[str, float] = field(default_factory=dict)
    n_surviving: int = 0
    n_specs: int = 0
    prior_signature: str = ""
    new_signature: str = ""
    n_rescored: int = 0


def _rescore_spec_gain(
    spec: CompositeSpec,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    idx: np.ndarray,
    config: Any,
) -> float:
    """Re-score one spec's MI gain on the sampled rows of ``df``.

    Mirrors the gain the full evaluator computes (``MI(T, X\\base) -
    MI(y, X\\base)``) but REUSES the spec's already-fit ``fitted_params``
    (train-only on the prior data) -- this is a cheap validity probe, not a
    refit. Returns NaN when the spec cannot be scored (missing base column,
    too few domain-valid rows, or an empty remaining feature set).
    """
    transform = get_transform(spec.transform_name)
    y = _extract_column_array(df, target_col, rows=idx)

    # Build the base matrix (primary + any multi-base extras). Unary transforms
    # ignore base entirely -- pass None straight through (their gain is scored
    # against the FULL feature matrix, no base dropped).
    extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
    base_cols: tuple[str, ...] = ()
    if transform.requires_base:
        base_cols = (spec.base_column, *extra) if spec.base_column else tuple(extra)
        missing = [c for c in base_cols if c not in _frame_columns(df)]
        if missing:
            return float("nan")
        if len(base_cols) == 1:
            base = _extract_column_array(df, base_cols[0], rows=idx)
        else:
            base = np.column_stack([_extract_column_array(df, c, rows=idx) for c in base_cols])
    else:
        base = None

    # x_remaining = all numeric features EXCEPT the base column(s). A residual's
    # base IS removed (it isolates the transform effect); a unary drops nothing.
    drop = set(base_cols)
    cols = [c for c in feature_cols if c != target_col and c not in drop]
    cols = [c for c in cols if c in _frame_columns(df)]
    if not cols:
        return float("nan")
    x_arrays = [_extract_column_array(df, c, rows=idx) for c in cols]
    x_matrix = np.column_stack(x_arrays).astype(np.float64, copy=False)

    valid = np.asarray(transform.domain_check(y, base), dtype=bool)  # type: ignore[type-var]  # np.asarray(x, dtype=bool) is valid at runtime; numpy-stub quirk under this module's strict settings
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None and isinstance(spec.fitted_params, dict):
        try:
            vf = np.asarray(_dcf(y, base, spec.fitted_params), dtype=bool)
            if vf.shape == valid.shape:
                valid = valid & vf
        except Exception as e:  # noqa: BLE001 -- treat as no refinement
            logger.debug("swallowed exception in _incremental.py: %s", e)
            pass
    if int(valid.sum()) < 50:
        return float("nan")

    y_v = y[valid].astype(np.float64)
    base_v = None if base is None else base[valid].astype(np.float64)
    try:
        t = transform.forward(y_v, base_v, spec.fitted_params)
    except Exception as _err:  # noqa: BLE001 -- a spec that can no longer forward is invalid on the new data
        logger.debug("incremental: spec %s forward failed: %s", spec.name, _err)
        return float("nan")
    x_v = x_matrix[valid]

    nbins = int(config.mi_nbins)
    aggregation = config.mi_aggregation
    if config.mi_estimator == "bin":
        x_pb = _prebin_feature_columns(x_v, nbins=nbins)
        mi_t = _mi_to_target_prebinned(x_pb, t, nbins=nbins, aggregation=aggregation)
        mi_y = _mi_to_target_prebinned(x_pb, y_v, nbins=nbins, aggregation=aggregation)
    else:
        mi_kwargs = dict(
            n_neighbors=config.mi_n_neighbors,
            random_state=config.random_state,
            estimator=config.mi_estimator,
            nbins=nbins,
            aggregation=aggregation,
        )
        mi_t = _mi_to_target(x_v, t, **mi_kwargs)
        mi_y = _mi_to_target(x_v, y_v, **mi_kwargs)
    return float(mi_t - mi_y)


def _frame_columns(df: Any) -> Any:
    """Column-name set for a pandas / polars frame (cheap, no materialisation)."""
    cols = getattr(df, "columns", None)
    if cols is None:
        return set()
    # polars ``.columns`` is a list; pandas ``.columns`` is an Index -- both
    # iterate to the column names. ``set`` makes the membership test O(1).
    return set(cols)


def incremental_discovery_check(
    prior_specs: Sequence[CompositeSpec],
    prior_signature: str,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    config: Any,
    *,
    sample_n: int = _DEFAULT_INCREMENTAL_SAMPLE_N,
    min_surviving_fraction: float = _DEFAULT_MIN_SURVIVING_FRACTION,
    eps_mi_gain: float | None = None,
) -> IncrementalDecision:
    """Decide REUSE vs full re-discovery for prior specs on an appended frame.

    Parameters
    ----------
    prior_specs
        The kept ``CompositeSpec`` list from a prior
        :meth:`CompositeTargetDiscovery.fit`.
    prior_signature
        The ``data_signature`` of the frame those specs were fit on.
    df, target_col, feature_cols, config
        The NEW (appended) frame + the same discovery config used originally.
    sample_n
        Rows sampled from the new frame for the MI re-score (bounded; never a
        whole-frame pass).
    min_surviving_fraction
        Fraction of prior specs that must still clear ``eps_mi_gain`` on the
        new sample for a REUSE verdict.
    eps_mi_gain
        Gain threshold; defaults to ``config.eps_mi_gain``. A spec "survives"
        when its re-scored gain is ``> eps_mi_gain``.

    Returns
    -------
    :class:`IncrementalDecision`. ``reuse=True`` returns the prior specs
    unchanged; ``reuse=False`` returns ``specs=None`` -- the caller must run a
    full ``fit`` (drift detected, or no prior specs to validate).
    """
    prior_specs = list(prior_specs)
    eps = float(config.eps_mi_gain if eps_mi_gain is None else eps_mi_gain)

    # No prior specs -> nothing to warm-start from; force a full discovery.
    if not prior_specs:
        new_sig = data_signature(df, target_col, feature_cols)
        return IncrementalDecision(
            reuse=False, specs=None,
            reason="no prior specs to validate -- run full discovery",
            n_specs=0, prior_signature=prior_signature, new_signature=new_sig,
        )

    new_sig = data_signature(df, target_col, feature_cols)
    # Byte-identical data (signature unchanged) -> trivial reuse, zero re-scores.
    # This is the cheapest path: the data did not actually change, so the prior
    # specs are exactly valid (the full-fit re-discovery would be pure waste).
    if new_sig == prior_signature and prior_signature:
        return IncrementalDecision(
            reuse=True, specs=prior_specs,
            reason="data signature unchanged -- prior specs exactly valid",
            n_surviving=len(prior_specs), n_specs=len(prior_specs),
            prior_signature=prior_signature, new_signature=new_sig,
            n_rescored=0,
        )

    n_rows = len(df)
    if n_rows == 0:
        return IncrementalDecision(
            reuse=False, specs=None, reason="new frame is empty",
            n_specs=len(prior_specs), prior_signature=prior_signature,
            new_signature=new_sig,
        )
    # Sorted sample of the new frame (preserves temporal order, same strategy as
    # the screen). When the frame is smaller than ``sample_n`` we use all rows.
    idx = _sample_indices(n_rows, sample_n, int(config.random_state))

    per_spec_gain: dict[str, float] = {}
    n_rescored = 0
    n_surviving = 0
    for spec in prior_specs:
        gain = _rescore_spec_gain(spec, df, target_col, feature_cols, idx, config)
        per_spec_gain[spec.name] = gain
        n_rescored += 1
        if np.isfinite(gain) and gain > eps:
            n_surviving += 1

    n_specs = len(prior_specs)
    frac = n_surviving / n_specs if n_specs else 0.0
    reuse = frac >= float(min_surviving_fraction)
    if reuse:
        reason = (
            f"{n_surviving}/{n_specs} specs still clear eps_mi_gain={eps:.4g} "
            f"on the new sample (frac={frac:.2f} >= {min_surviving_fraction:.2f}) "
            f"-- reuse prior specs, skip full re-discovery"
        )
    else:
        reason = (
            f"only {n_surviving}/{n_specs} specs clear eps_mi_gain={eps:.4g} "
            f"on the new sample (frac={frac:.2f} < {min_surviving_fraction:.2f}) "
            f"-- DGP drift, run full re-discovery"
        )
    logger.info("[incremental] %s", reason)
    return IncrementalDecision(
        reuse=reuse,
        specs=prior_specs if reuse else None,
        reason=reason,
        per_spec_gain=per_spec_gain,
        n_surviving=n_surviving,
        n_specs=n_specs,
        prior_signature=prior_signature,
        new_signature=new_sig,
        n_rescored=n_rescored,
    )
