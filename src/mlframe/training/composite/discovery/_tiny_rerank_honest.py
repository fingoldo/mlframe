"""Honest-OOF prepass and ordering for the tiny-model rerank, plus its RAM checkpoint; split out of ``_tiny_rerank`` to keep it under the module size budget."""

from __future__ import annotations

import logging
import math


from ._tiny_rerank_waic import apply_honest_oof_floor

# The parent module's logger name: these lines predate the split, and log filters and tests select them by that name.
logger = logging.getLogger("mlframe.training.composite.discovery._tiny_rerank")


def _tiny_rerank_ram_checkpoint(label: str) -> None:
    """Log the current process memory triple right at a tiny-rerank boundary.

    Mirrors the discovery profiler format so the prod log presents a single
    coherent thread of RAM checkpoints across discovery sub-phases AND the
    tiny-rerank internals.

    The user observed a kernel-kill INSIDE tiny_model_rerank with ~20 GB of
    physical RAM still free -- not a classical OOM. Most likely cause on
    Windows: the system-wide commit-charge limit (physical + pagefile) was
    exhausted by mlframe + system baseline + the next LightGBM Dataset's
    transient allocation; the kernel-level C alloc fails, LightGBM doesn't
    handle it gracefully, and the process crashes with an access violation.
    Per-step checkpoints pin the exact step that pushed commit over the
    edge.
    """
    try:
        from ._fit import _process_mem_mb
        rss_mb, uss_mb, commit_mb = _process_mem_mb()
    except Exception as exc:
        logger.debug("tiny_rerank RAM log: memory probe failed, skipping checkpoint: %s", exc)
        return
    logger.info(
        "[CompositeTargetDiscovery.tiny_rerank.RAM] %s USS=%.0f MB (RSS=%.0f MB, commit=%.0f MB)",
        label, uss_mb, rss_mb, commit_mb,
    )


def _honest_oof_prepass(self, df, target_col, kept_specs, usable_features, train_idx, y_full, *, will_run: bool, per_bin_enabled: bool, use_wilcoxon: bool) -> "dict[str, float] | None":
    """Honest group-OOF reconstruction RMSE measured BEFORE the CV sweep, or ``None`` when the sweep must run anyway.

    Honest-OOF REPLACES the group-internal CV-RMSE of every spec it can measure, so computing those scores first was
    pure waste: on a 16-spec grouped run the sweep cost 11.6 s of model fits against 0.26 s for this measurement, and
    all 16 results were overwritten. Measuring it first lets the covered specs skip the sweep. That is only safe when
    nothing else consumes the sweep -- the per-bin regime gate reuses its first-pass breakdown and the Wilcoxon gate
    needs its per-seed vectors -- so with either enabled this returns ``None`` and every spec is fitted as before. ``None``
    (not measured) is kept distinct from ``{}`` (measured, nothing scorable) so the latter is not measured a second time.
    """
    if not (will_run and not per_bin_enabled and not use_wilcoxon):
        return None
    from ._honest_oof_select import honest_oof_reconstruction_rmse

    select_idx = getattr(self, "honest_holdout_select_idx_", None)
    if select_idx is None:
        select_idx = getattr(self, "honest_holdout_idx_", None)
    return dict(honest_oof_reconstruction_rmse(self, df, target_col, kept_specs, usable_features, train_idx, select_idx, y_full) or {})


def _put_unmeasured_on_the_honest_scale(kept_specs, agg_scores, honest_oof: dict, honest_raw: float, raw_cv: float) -> None:
    """Rescale, in place, the CV score of every spec the honest measurement could not score onto the honest scale.

    Group-internal CV is optimistic (about 9 against an honest 13.6 on the production case), so an unmeasured spec kept
    its lower CV score, sorted above the honestly measured specs and passed the honest threshold easily: the specs the
    honest path could not measure were the most likely to reach the top-M. The raw-y model's honest / CV ratio, measured on
    the same run, converts the scale; without it, an unmeasured spec is placed after every measured one.
    """
    measured = [float(v) for v in honest_oof.values() if v is not None and math.isfinite(v)]
    ratio = honest_raw / raw_cv if math.isfinite(honest_raw) and math.isfinite(raw_cv) and raw_cv > 0 else float("nan")
    for i, spec in enumerate(kept_specs):
        if honest_oof.get(spec.name) is not None or not math.isfinite(agg_scores[i]):
            continue
        agg_scores[i] = agg_scores[i] * ratio if math.isfinite(ratio) and ratio > 0 else (max(measured, default=0.0) + agg_scores[i])


def _apply_honest_oof_ordering(self, df, target_col, kept_specs, agg_scores, usable_features, train_idx, y_full, _honest_oof_pre, raw_cv_baseline: float = float("nan")):
    """Re-rank the survivors by honest group-OOF reconstruction RMSE and enforce its floor, in place of the CV order.

    Returns ``(kept_specs, agg_scores, baseline)``; without group ids and a holdout this is a no-op and the
    baseline is ``nan``, so the CV ordering stands exactly as it did.
    """
    # Honest group-OOF reconstruction RMSE becomes the load-bearing ORDERING key (and the raw-baseline gate reference)
    # when a group-disjoint honest holdout exists. The group-internal CV-RMSE above stays as the fallback for any spec
    # whose holdout measurement degenerated (too few valid rows) -- a degenerate MEASUREMENT must not auto-kill a spec;
    # only a genuine COLLAPSE returns +inf and sinks the spec. No-op (ordering bit-identical) without group ids + holdout.
    _honest_oof_baseline = float("nan")
    if (
        bool(getattr(self.config, "honest_oof_selection", True))
        and getattr(self, "_group_ids_for_rerank", None) is not None
        and getattr(self, "honest_holdout_idx_", None) is not None
    ):
        from ._honest_oof_select import honest_oof_reconstruction_rmse

        # Reuse the pre-sweep measurement when it was taken (same inputs, same selection half); otherwise measure now.
        _honest_oof = _honest_oof_pre if _honest_oof_pre is not None else honest_oof_reconstruction_rmse(
            self, df, target_col, kept_specs, usable_features,
            # Selection half: this score is a ranking key, so it must not be measured on the reported rows.
            train_idx, getattr(self, "honest_holdout_select_idx_", None) if getattr(self, "honest_holdout_select_idx_", None) is not None else getattr(self, "honest_holdout_idx_", None), y_full,
        )
        if _honest_oof:
            self._honest_oof_rmse = dict(_honest_oof)
            # Floor the gate against min(raw-y, AR-failsafe): a spec that reconstructs worse than the lag_predict
            # failsafe we would deploy anyway is worthless even if it beats the raw-y model. On non-AR data (no lag
            # column) the lag floor is nan and this reduces to the raw floor -- bit-identical to the prior behaviour.
            _honest_raw = float(getattr(self, "_honest_oof_raw_rmse", float("nan")))
            _honest_lag = float(getattr(self, "_honest_oof_lag_rmse", float("nan")))
            _floor_candidates = [v for v in (_honest_raw, _honest_lag) if math.isfinite(v)]
            _honest_oof_baseline = min(_floor_candidates) if _floor_candidates else float("nan")
            for i, _spec in enumerate(kept_specs):
                _hv = _honest_oof.get(_spec.name)
                if _hv is not None:
                    agg_scores[i] = float(_hv)
                    object.__setattr__(_spec, "honest_oof_rmse", float(_hv))
            _put_unmeasured_on_the_honest_scale(kept_specs, agg_scores, _honest_oof, _honest_raw, raw_cv_baseline)
            self._tiny_rerank_scores = {kept_specs[i].name: float(agg_scores[i]) for i in range(len(kept_specs))}
            logger.info(
                "[CompositeTargetDiscovery.honest_oof_select] ranking %d spec(s) by honest group-OOF "
                "reconstruction RMSE (floor=%.4g = min(raw-y=%.4g, AR-lag=%.4g)); %d measured, rest fall back to "
                "group-internal CV.",
                len(kept_specs), _honest_oof_baseline, _honest_raw, _honest_lag, len(_honest_oof),
            )

            # Enforce the honest-OOF floor as a REJECTION (carved to _tiny_rerank_waic for the 1k-LOC limit): drop specs
            # whose measured honest reconstruction cannot beat min(raw-y, AR-lag); otherwise honest-OOF only reorders.
            kept_specs, agg_scores = apply_honest_oof_floor(
                self, kept_specs, agg_scores, _honest_oof, _honest_oof_baseline,
            )
    return kept_specs, agg_scores, _honest_oof_baseline
