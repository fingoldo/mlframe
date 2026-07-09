"""Helpers for composite-target discovery: winning-spec diagnostics render, per-target discovery-frame
construction (no caller mutation), and the config+library-version signature used for cache invalidation.
Carved from _phase_composite_discovery.py; the parent re-imports these names."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl

from ..composite.cache import ConfigSignatureV1, compute_config_signature_v1

logger = logging.getLogger(__name__)


def _render_composite_discovery_diagnostics(
    *,
    data_dir: Any,
    raw_target_name: str,
    y_full: np.ndarray,
    t_by_spec: Dict[str, np.ndarray],
    specs_export: List[dict],
) -> List[str]:
    """Render the winning-spec target-distribution + MI-gain diagnostics under ``data_dir``.

    One ``plot_mi_gain_with_jitter`` per raw target (ranks the accepted specs), plus one
    ``plot_target_distribution`` per accepted spec (y-vs-T shape sanity-check). Both are small
    per-spec diagnostics; the helpers already subsample huge inputs internally. Returns the saved
    paths so the caller can stamp them into ``metadata`` for the chart-summary log.
    """
    import os

    import matplotlib.pyplot as plt

    from ..composite.diagnostics import plot_mi_gain_with_jitter, plot_target_distribution

    saved: List[str] = []
    _safe_target = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(raw_target_name))
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as _mk_err:
        logger.info("[CompositeTargetDiscovery] chart dir create failed (%s); diagnostics skipped.", _mk_err)
        return saved

    def _save(fig, suffix: str) -> None:
        """Save ``fig`` under a target-scoped filename, record the path in ``saved``, and always close the figure afterward to avoid leaking matplotlib figure objects across repeated discovery calls."""
        path = os.path.join(data_dir, f"composite_{_safe_target}_{suffix}.png")
        try:
            fig.savefig(path, dpi=110, bbox_inches="tight")
            saved.append(path)
        finally:
            plt.close(fig)

    if specs_export:
        try:
            _save(plot_mi_gain_with_jitter(specs_export), "mi_gain")
        except Exception as _mi_err:
            logger.info("[CompositeTargetDiscovery] mi-gain diagnostic render failed for '%s': %s.", raw_target_name, _mi_err)
    for _spec_name, _t_full in t_by_spec.items():
        _safe_spec = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(_spec_name))
        try:
            _save(
                plot_target_distribution(y_full, _t_full, title=f"Target distribution: y vs T ({_spec_name})"),
                f"tdist_{_safe_spec}",
            )
        except Exception as _td_err:
            logger.info("[CompositeTargetDiscovery] target-distribution diagnostic render failed for spec '%s': %s.", _spec_name, _td_err)
    return saved


def _build_disc_df_for_target(filtered_train_df, target_name: str, y_train_aligned):
    """Build a per-target discovery frame that injects ``target_name`` WITHOUT mutating the caller's ``filtered_train_df``.

    Pandas ``DataFrame.copy(deep=False)`` shares the underlying BlockManager with the source; a subsequent
    ``out[target_name] = arr`` setitem can promote and mutate the SHARED block depending on the existing dtype layout,
    causing the target column to intermittently appear on the caller's ``filtered_train_df`` post-loop. The per-target
    discovery loop then accumulates leakage: target_A injected for the first iter shows up as a feature when target_B is
    processed next. ``DataFrame.assign`` always builds a fresh BlockManager so the source is guaranteed untouched, at the
    same memory cost as ``copy(deep=False)+setitem`` would have paid on the new column anyway.

    Polars ``with_columns`` is naturally immutable and returns a fresh frame, so no special handling is needed there.
    """
    if isinstance(filtered_train_df, pd.DataFrame):
        # concat(axis=1) builds a fresh BlockManager (so the caller's frame
        # is NOT mutated -- same immutability guarantee as the prior
        # ``.assign``) while attaching the target as a single consolidated
        # block. On the wide, upstream-fragmented discovery frame this also
        # silences pandas' "highly fragmented" PerformanceWarning that
        # per-column ``.assign``/insert triggers.
        target_series = pd.Series(
            y_train_aligned, index=filtered_train_df.index, name=target_name,
        )
        cols_wo_target = [c for c in filtered_train_df.columns if c != target_name]
        return pd.concat([filtered_train_df[cols_wo_target], target_series], axis=1)
    return filtered_train_df.with_columns(pl.Series(target_name, y_train_aligned))


def _discovery_config_signature(config: Any) -> ConfigSignatureV1:
    """Stable JSON-derived signature of a CompositeTargetDiscoveryConfig.

    Combined with library versions so a dependency bump invalidates
    cached specs - this is the cache-poisoning protection: a CatBoost
    upgrade changes MI bin boundaries, a polars 1->2 bump changes
    categorical codes, a numpy 2.x bump changes RNG semantics, so we
    MUST refit. The version tuple covers every library whose semantics
    can shift the discovered specs:

      * ``mlframe`` - our own version (any change is a refit signal)
      * ``sklearn`` - shared transformers; MI estimator lives here
      * ``lightgbm`` / ``catboost`` / ``xgboost`` - inner models for
        the tiny-model rerank phase
      * ``polars`` - categorical/string dtype codes that feed into
        domain checks + signatures
      * ``numpy`` - dtype promotions + RNG defaults changed in 2.x
      * ``scipy`` - Wilcoxon implementation
      * ``pandas`` - dtype dispatch on the fallback path
      * ``python`` - major.minor (3.11 -> 3.12 changes pickle proto +
        dict ordering side-effects in some serialisers)
    """
    import sys

    versions: dict[str, str] = {}
    try:
        from mlframe import __version__ as _mlv
        versions["mlframe"] = _mlv
    except Exception:
        versions["mlframe"] = "?"
    for _name in (
        "sklearn",
        "lightgbm",
        "catboost",
        "xgboost",
        "polars",
        "numpy",
        "scipy",
        "pandas",
    ):
        try:
            mod = __import__(_name)
            _ver_str = str(getattr(mod, "__version__", "?"))
            # Major.minor only -- patch bumps invalidate every cached spec even though MI /
            # Wilcoxon / boosting math is unchanged. Strip patch + any dev / rc tags.
            _parts = _ver_str.split(".")
            if len(_parts) >= 2 and _parts[0].isdigit():
                _ver_str = f"{_parts[0]}.{_parts[1].split('+')[0].split('rc')[0].split('dev')[0]}"
            versions[_name] = _ver_str
        except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            versions[_name] = "absent"
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    return compute_config_signature_v1(config, library_versions=versions)
