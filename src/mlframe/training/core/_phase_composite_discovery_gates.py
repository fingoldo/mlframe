"""Pre-discovery gates of the composite-target phase: pathology-driven auto-enable, the base-materialisability filter and the discovery-cache lookup."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..composite.cache import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from ..configs import TargetTypes
from mlframe.utils.log_throttle import log_throttle

# The parent module's logger name: these lines predate the split, and log filters and tests select them by that name.
logger = logging.getLogger("mlframe.training.core._phase_composite_discovery")
from ._phase_composite_discovery_helpers import (
    _discovery_config_signature,
)

# Pathology name PREFIXES (TargetDistributionReport.pathologies entries are formatted strings like
# "heavy_tail(excess_kurt=312.5)") that justify auto-enabling composite-target discovery: these are
# exactly the two regression pathologies target_distribution_analyzer's own docstring/comment already
# says composite discovery is "the" answer for (log/sqrt/cbrt residual targets), yet
# analyze_target_distribution deliberately issues no knob_override for them because a target transform
# is composite discovery's job, not a hyperparameter flip -- see
# ``_target_distribution_analyzer_target_fn.py``'s skewed_target branch comment. Left disconnected, the
# two diagnostics talked past each other: the target-side analyzer detects the pathology and defers, and
# composite discovery (default OFF) never actually runs unless a caller manually opts in.
_AUTO_ENABLE_DISCOVERY_PATHOLOGY_PREFIXES = ("heavy_tail", "skewed_target")


def _target_pathologies_for_auto_enable(y_train: np.ndarray, target_name: str, td_report: dict) -> list[str]:
    """Pathologies relevant to auto-enabling discovery for one regression target.

    Reuses the suite-level ``target_distribution_report`` (already computed once per suite on a picked
    representative target) when ``target_name`` IS that picked target -- no extra compute. For any other
    regression target, runs the same cheap O(n) moments-based analyzer fresh (mean/std/skew/kurtosis over
    the train slice only) since the suite-level report never covers more than one target.
    """
    if td_report.get("picked_target_name") == target_name:
        pathologies = list(td_report.get("pathologies", []))
    else:
        try:
            from ..targets import analyze_target_distribution

            pathologies = list(analyze_target_distribution(y_train, target_type="regression", has_time_axis=False).pathologies)
        except Exception as e:
            logger.debug("per-target heavy-tail/skew auto-enable check failed for %r (%s); treating as no pathology", target_name, e)
            return []
    return [p for p in pathologies if p.startswith(_AUTO_ENABLE_DISCOVERY_PATHOLOGY_PREFIXES)]


def _maybe_auto_enable_discovery(composite_target_discovery_config, *, target_by_type: dict, train_idx, metadata: dict):
    """Auto-enable composite-target discovery for a suite that left ``enabled`` at its default (never
    explicitly opted out) when a real regression target shows a heavy-tail/skewed-target pathology.

    Explicit user intent always wins either way: an explicit ``enabled=True`` runs regardless of this
    check, and an explicit ``enabled=False`` (present in ``model_fields_set``) is respected as a
    deliberate opt-out and never overridden. Returns the (possibly ``.model_copy``'d) effective config.
    """
    if composite_target_discovery_config.enabled or "enabled" in composite_target_discovery_config.model_fields_set:
        return composite_target_discovery_config
    _reasons: dict[str, list[str]] = {}
    for _tname_auto, _tvals_auto in (target_by_type.get(TargetTypes.REGRESSION) or {}).items():
        try:
            _y_auto = np.asarray(_tvals_auto)
            if _y_auto.ndim != 1:
                continue
            if train_idx is not None:
                _idx_auto = np.asarray(train_idx)
                if _idx_auto.size and _y_auto.size > int(_idx_auto.max()):
                    _y_auto = _y_auto[_idx_auto]
        except Exception as e:
            logger.debug("auto-enable target slice failed for %r (%s)", _tname_auto, e)
            continue
        _hits = _target_pathologies_for_auto_enable(_y_auto, _tname_auto, metadata.get("target_distribution_report", {}) or {})
        if _hits:
            _reasons[_tname_auto] = _hits
    if not _reasons:
        return composite_target_discovery_config
    logger.info(
        "[CompositeTargetDiscovery] auto-enabled (composite_target_discovery_config.enabled was left at its "
        "default): target(s) show a heavy-tail/skewed-target pathology composite discovery's log/cbrt/"
        "yeo_johnson/quantile_normal transforms directly address -- %s. Pass "
        "composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False) explicitly to opt out.",
        _reasons,
    )
    return composite_target_discovery_config.model_copy(update={"enabled": True})


def _drop_specs_whose_bases_the_suite_cannot_materialise(disc, split_frames, target_name: str) -> list[dict]:
    """Drop kept specs whose base columns are absent from every split frame, returning one failure record each.

    Discovery may build extra base columns on its OWN frame -- the engineered per-group causal bases are the default
    case. Those columns never reach the suite's split frames, and the column builder silently yields all-NaN for a name
    it cannot find, so such a spec would be trained on a NaN target and every gate verdict recorded for it was measured
    on a column the trainer does not have. They are screening-only by construction: the bases are functions of past y,
    which a predict frame does not carry, so there is nothing to rebuild downstream either.
    """
    specs = list(getattr(disc, "specs_", ()) or ())
    if not specs:
        return []
    available: set = set()
    for _frame in split_frames:
        if _frame is None:
            continue
        try:
            available.update(str(c) for c in _frame.columns)
        except Exception as e:
            logger.debug("reading split frame columns failed while checking spec bases: %s", e)
    if not available:  # nothing to check against: keep the specs rather than drop them on a failed lookup
        return []
    kept, dropped = [], []
    for _spec in specs:
        _needed = [str(getattr(_spec, "base_column", "") or "")]
        _needed += [str(_c) for _c in (getattr(_spec, "extra_base_columns", ()) or ())]
        _missing = [_c for _c in _needed if _c and _c not in available]
        if _missing:
            dropped.append({
                "name": getattr(_spec, "name", None) or getattr(_spec, "transform_name", "?"),
                "kept": False,
                "rejected": True,
                "reason": f"base column(s) {_missing} exist only in discovery's own frame, so the suite cannot build this target",
            })
        else:
            kept.append(_spec)
    if dropped:
        disc.specs_ = kept
        log_throttle(
            logger, "composite_spec_base_not_materialisable", logging.WARNING,
            "[CompositeTargetDiscovery] target='%s': dropped %d spec(s) whose base column is not in the training frames "
            "(%s). Engineered bases are screening-only; pass such a column in your own frame to train on it.",
            target_name, len(dropped), "; ".join(str(d["reason"]) for d in dropped[:3]),
        )
    return dropped


def discovery_inputs_digest(*, group_ids: Any = None, hint_strengths: Any = None, disc_df: Any = None, time_column: Any = None,
                            val_df: Any = None, val_y: Any = None, y_full: Any = None, val_idx: Any = None) -> str:
    """A digest of the discovery inputs that change the selected specs but are neither data columns nor config fields.

    The group ids drive the group-disjoint holdout, the GroupKFold rerank and the fragility gate; the hint strengths decide
    the hint cap and the ablation skip; the time-column values order the screen; the val frame is the y-scale gate's
    evaluation set. Without them in the key a rerun under a new group split or val set replayed specs gated on the old one.
    """
    import hashlib

    if val_y is None and y_full is not None and val_idx is not None:
        val_y = np.asarray(y_full)[val_idx]  # the val targets as the phase hands them to the y-scale gate
    if val_y is None:
        val_df = None  # the gate ignores a val frame without targets
    h = hashlib.blake2b(digest_size=16)

    def _arr(tag: str, a: Any) -> None:
        """Fold one array-like into the digest by tag, shape and dtype, with a distinct marker for a missing one."""
        h.update(tag.encode())
        if a is None:
            h.update(b"<none>")
            return
        arr = np.asarray(a)
        h.update(str((arr.shape, arr.dtype.str)).encode())
        h.update(np.ascontiguousarray(arr.astype(str) if arr.dtype == object else arr).tobytes())

    _arr("groups", group_ids)
    _arr("hints", None if hint_strengths is None else np.asarray(list(hint_strengths), dtype=np.float64))
    _time = None
    if time_column and disc_df is not None and time_column in getattr(disc_df, "columns", ()):
        _time = disc_df.get_column(time_column).to_numpy() if hasattr(disc_df, "get_column") else disc_df[time_column].to_numpy()
    _arr("time", _time)
    _arr("val_y", val_y)
    if val_df is None:
        h.update(b"val_df<none>")
    else:
        h.update(str((tuple(getattr(val_df, "columns", ())), getattr(val_df, "shape", None))).encode())
        head = val_df.head(2000)
        _rows = head.hash_rows().to_numpy() if hasattr(head, "hash_rows") else pd.util.hash_pandas_object(head, index=False).to_numpy()
        h.update(np.ascontiguousarray(_rows).tobytes())
    return h.hexdigest()


def _discovery_cache_lookup(disc_cfg, disc_df, target_name, feature_cols, cache_dir, inputs_digest: str = ""):
    """The discovery cache, its key and any cached payload for this target; a failed key build yields no cache.

    The key carries the data fingerprint, the target column and the config signature (which embeds the library
    versions, so a poisoned entry cannot survive an upgrade). A hit skips the whole MI / rerank path.
    """
    cache = None
    cache_key = None
    try:
        cache = DiscoveryCache(cache_dir)
        # ``random_state=0`` is a legitimate sklearn seed and MUST
        # reach the row-sampler verbatim. The previous ``or 42`` form
        # silently rewrote 0->42, collapsing seed=0 and seed=42 to
        # the same data_signature and breaking reproducibility for
        # any caller that passed 0. ``None`` (no attribute / unset)
        # still folds to 42 (the historical default).
        _rs_raw = getattr(disc_cfg, "random_state", 42)
        _df_sig = data_signature(
            disc_df, target_name, feature_cols,
            random_state=int(42 if _rs_raw is None else _rs_raw),
        )
        _cfg_sig = _discovery_config_signature(disc_cfg)
        # random_state is already folded into _df_sig (seeds the row-sample) and into _cfg_sig (via the dataclass dump). Passing it again to make_discovery_cache_key would be a double-fold (DISC-RANDOM-STATE-DBL): the same data + same config but with random_state mutated would produce three independent hash mixes. We rename the kwarg here to ``_legacy_random_state_sentinel=0`` so a future reader cannot misread "random_state=0" as the actual seed in use.
        # The inputs digest (group ids, hint strengths, time order, val frame) rides with the data fingerprint.
        cache_key = make_discovery_cache_key(
            f"{_df_sig}|{inputs_digest}" if inputs_digest else _df_sig, target_name, _cfg_sig,
            _legacy_random_state_sentinel=0,
        )
        payload = cache.get(cache_key)
    except Exception as _cache_err:
        logger.info(
            "[CompositeTargetDiscovery] cache key build failed for " "target='%s' (%s); proceeding without cache.",
            target_name,
            _cache_err,
        )
        payload = None
    return cache, cache_key, payload


def rank_pending_composites(pending: list) -> list:
    """The cross-target budget order: specs with a relative honest RMSE gain first, then the ones that fell back to MI.

    Each tier is ranked in its own unit (a fraction of RMSE, or nats); one sort over both ranked a fraction against nats,
    i.e. arbitrarily across targets. A non-finite gain sorts last in its tier.
    """
    return sorted(pending, key=lambda item: (bool(item.get("rmse_gain")), item["gain"] if np.isfinite(item["gain"]) else -np.inf), reverse=True)


def _maybe_narrow_to_unary_transforms(disc_cfg: Any, diag: Any, target_name: str) -> Any:
    """Drop base-dependent transforms when BaselineDiagnostics found no dominant feature to residualise against.

    BaselineDiagnostics computes a ``composite_recommendation`` whose whole purpose is to say whether composite
    discovery is worth running, and the suite already has it at the per-target decision point (the same precompute the
    dominant-features hint comes from) -- it was simply never read, so the verdict only reached the log AFTER discovery
    had finished and committed its specs.

    ``unlikely_to_help`` means "no dominant features", which is a statement about BASE-DEPENDENT families: a residual
    transform has nothing to residualise against. The base-free unary y-transforms are untouched by that finding, and
    in the production run the one composite that beat raw y was exactly one of those. So the verdict narrows discovery
    to the unary family rather than cancelling it.

    Returns ``disc_cfg`` unchanged when the diagnostic is absent, says something else, or the narrowing would leave
    nothing to search.
    """
    if not isinstance(diag, dict) or diag.get("composite_recommendation") != "unlikely_to_help":
        return disc_cfg
    try:
        from ..composite.transforms import UnknownTransformError, get_transform

        unary: list[str] = []
        for name in list(getattr(disc_cfg, "transforms", ()) or ()):
            try:
                if not get_transform(name).requires_base:
                    unary.append(name)
            except UnknownTransformError:  # noqa: PERF203 -- per-transform fault isolation is intentional; an unknown name skips, never aborts the narrowing
                continue
        if not unary or len(unary) == len(list(disc_cfg.transforms)):
            return disc_cfg
        logger.info(
            "[CompositeTargetDiscovery] target=%r: BaselineDiagnostics reports composite_recommendation="
            "'unlikely_to_help' (%s). That verdict is about BASE-DEPENDENT families -- with no dominant feature "
            "there is nothing to residualise against -- so discovery is narrowed from %d transform(s) to the %d "
            "base-free unary one(s), which the finding does not bear on.",
            target_name, diag.get("composite_recommendation_reason", "no reason recorded"),
            len(list(disc_cfg.transforms)), len(unary),
        )
        return disc_cfg.model_copy(update={"transforms": unary})
    except Exception as exc:
        logger.debug("narrowing discovery to unary transforms failed for %r (%s); full search proceeds", target_name, exc)
        return disc_cfg


_DEFAULT_MIN_HONEST_GAIN_Z: float = 2.0
"""Standard errors a spec's honest-holdout RMSE gain must clear before the spec is worth a full model fit, on top of
the constant ``min_honest_gain_to_train`` floor. 0 disables the noise-aware half and restores the constant-only bar."""


def _relative_gain_se(spec: Any, raw_rmse: Optional[float]) -> float | None:
    """The spec's paired standard error of its honest RMSE gain, on the same relative-to-raw scale as the gain itself."""
    _gain_se = getattr(spec, "honest_holdout_rmse_gain_se", None)
    return float(_gain_se) / float(raw_rmse) if (_gain_se is not None and raw_rmse is not None and raw_rmse != 0) else None


def _drop_below_honest_gain_floor(pending: list[dict], composite_target_discovery_config: Any) -> list[dict]:
    """``pending`` minus the RMSE-gain specs at or below their ship floor, logging the dropped ones.

    A constant floor cannot tell a real 0.4% gain from a 0.4% measurement error. Each spec carries the paired standard
    error of its own gain, so the bar is "beats the constant ``min_honest_gain_to_train`` AND is larger than
    ``min_honest_gain_z`` of its own noise" -- a production run shipped 9 specs at gains of +0.002..+0.011 and warned
    about GPU non-determinism of the same order in the very next log line. No-op when the constant floor is unset.
    """
    _min_gain = getattr(composite_target_discovery_config, "min_honest_gain_to_train", None)
    if _min_gain is None:
        return pending
    _min_gain_z = float(getattr(composite_target_discovery_config, "min_honest_gain_z", _DEFAULT_MIN_HONEST_GAIN_Z))

    def _floor_for(p: dict) -> float:
        """Ship/no-ship floor for one pending spec: the configured constant, raised to its measurement noise when known."""
        _se = p.get("gain_se")
        if _min_gain_z <= 0 or _se is None or not np.isfinite(_se) or _se <= 0:
            return float(_min_gain)
        return max(float(_min_gain), _min_gain_z * float(_se))

    _below = [p for p in pending if p.get("rmse_gain") and p["gain"] <= _floor_for(p)]
    if not _below:
        return pending
    logger.info(
        "[CompositeTargetDiscovery] not training %d composite target(s) whose honest-holdout RMSE gain is at or "
        "below its floor (min_honest_gain_to_train=%.3f, raised to %.1f x the gain's own paired standard error "
        "where measurable): %s",
        len(_below), float(_min_gain), _min_gain_z,
        ", ".join(
            f"{d['name']}({d['gain']:+.4f} vs floor {_floor_for(d):.4f}" + (f", se={d['gain_se']:.4f}" if d.get("gain_se") else "") + ")" for d in _below
        ),
    )
    return [p for p in pending if p not in _below]


def select_composites_to_train(pending: list[dict], cfg: Any, metadata: dict) -> list[dict]:
    """The pending composites to train: drop those at or below their honest-gain floor, rank the rest by gain across the
    whole run, keep the best ``max_total_composite_targets`` (None keeps all). Every dropped spec leaves
    ``metadata['composite_target_specs']`` and is recorded in ``composite_target_failures`` with the reason.
    """
    max_total = getattr(cfg, "max_total_composite_targets", None)
    from ._phase_composite_discovery_dedup import forget_untrained_specs

    before_floor = pending
    pending = _drop_below_honest_gain_floor(pending, cfg)
    kept_ids = {id(p) for p in pending}
    forget_untrained_specs(metadata, [p for p in before_floor if id(p) not in kept_ids], "honest-holdout RMSE gain at or below its floor")
    pending = rank_pending_composites(pending)
    if max_total is not None and len(pending) > int(max_total):
        kept = pending[: int(max_total)]
        dropped = pending[int(max_total) :]
        logger.info(
            "[CompositeTargetDiscovery] global cap: keeping the %d best-scoring composite target(s) of %d "
            "discovered (max_total_composite_targets=%d, ranked by honest-holdout OOS RMSE gain vs raw-y, "
            "%% of baseline saved). Dropped: %s",
            len(kept), len(pending), int(max_total),
            ", ".join(f"{d['name']}({d['gain']:+.3f})" for d in dropped),
        )
        forget_untrained_specs(metadata, dropped, f"global cap max_total_composite_targets={int(max_total)}")
    else:
        kept = pending
    return kept
