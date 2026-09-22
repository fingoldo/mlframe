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

_CONCAT_NO_COPY: Dict[str, Any] = {} if int(pd.__version__.split(".")[0]) >= 3 else {"copy": False}
"""``pd.concat`` keywords that avoid copying the frame: ``copy=False`` before pandas 3, nothing on 3+ (copy-on-write)."""


def _render_composite_discovery_diagnostics(
    *,
    data_dir: Any,
    raw_target_name: str,
    y_full: np.ndarray,
    t_by_spec: Dict[str, np.ndarray],
    specs_export: List[dict],
    train_idx: Any = None,
) -> List[str]:
    """Render the winning-spec target-distribution + MI-gain diagnostics into the run's chart tree.

    The y-vs-T chart is a train-time selection diagnostic, so it plots the ``train_idx`` rows only, and only those with a
    finite T: a domain-violating row has no T (drawing it at an imputed constant made a spurious spike), and val / test
    rows must not shape it. The number of rows left out is in the title.

    One ``plot_mi_gain_with_jitter`` per raw target (ranks the accepted specs), plus one
    ``plot_target_distribution`` per accepted spec (y-vs-T shape sanity-check). Both are small
    per-spec diagnostics; the helpers already subsample huge inputs internally. Returns the saved
    paths so the caller can stamp them into ``metadata`` for the chart-summary log.

    Layout: ``<data_dir>/charts/<target>/composite_discovery/``, matching the ``<data_dir>/charts/<target>/...``
    convention every other chart type follows via ``_setup_model_directories``. These used to land flat in
    ``data_dir`` as ``composite_<target>_<suffix>.png``, so a multi-target run dumped every composite chart
    into the run root alongside the data artifacts instead of grouping them per target under ``charts/``.
    Discovery runs before any model exists, hence a ``composite_discovery`` leaf rather than the per-model
    ``<model>/<target_type>/<cur_target>`` tail the trained-model charts use.
    """
    from ..composite._row_roles import note_rows

    note_rows("train", "plot", "discovery_target_distribution_chart", train_idx)
    import os

    import matplotlib.pyplot as plt
    from pyutilz.strings import slugify

    from ..composite.diagnostics import plot_mi_gain_with_jitter, plot_target_distribution

    saved: List[str] = []
    _chart_dir = os.path.join(str(data_dir), "charts", slugify(str(raw_target_name)), "composite_discovery")
    try:
        os.makedirs(_chart_dir, exist_ok=True)
    except OSError as _mk_err:
        logger.info("[CompositeTargetDiscovery] chart dir create failed (%s); diagnostics skipped.", _mk_err)
        return saved

    def _save(fig, suffix: str) -> None:
        """Save ``fig`` into the target's chart dir, record the path in ``saved``, and always close the figure afterward to avoid leaking matplotlib figure objects across repeated discovery calls."""
        # The directory already scopes the target, so the filename no longer repeats it.
        path = os.path.join(_chart_dir, f"{suffix}.png")
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
    _spec_meta = {str(d.get("name")): d for d in (specs_export or []) if isinstance(d, dict)}
    _rows = np.arange(len(y_full)) if train_idx is None else np.asarray(train_idx)
    for _spec_name, _t_full in t_by_spec.items():
        _safe_spec = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(_spec_name))
        _y_tr, _t_tr = np.asarray(y_full, dtype=np.float64)[_rows], np.asarray(_t_full, dtype=np.float64)[_rows]
        _ok = np.isfinite(_y_tr) & np.isfinite(_t_tr)
        _left_out = f", {int((~_ok).sum())} train rows without a finite T left out" if not _ok.all() else ""
        try:
            _save(
                plot_target_distribution(
                    _y_tr[_ok], _t_tr[_ok], title=f"Target distribution on train: y vs T ({_spec_name}{_left_out})",
                    y_name=str(raw_target_name),
                    transform_name=_spec_meta.get(_spec_name, {}).get("transform_name"),
                    base_column=_spec_meta.get(_spec_name, {}).get("base_column") or None,
                ),
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
        # Under pandas 1.5-2.x (no copy-on-write) a list-column selection materialises the whole frame and concat's default
        # copy=True copies it again: up to two transient train-frame copies per target just to attach y. Skip the selection
        # when the target is not already a column (the usual case) and ask concat not to copy; the result is still a new
        # frame, so the caller's is untouched, and discovery only reads it. pandas 3 is zero-copy here and deprecates the
        # ``copy`` keyword, so it is passed only on older versions.
        base = filtered_train_df.drop(columns=[target_name]) if target_name in filtered_train_df.columns else filtered_train_df
        return pd.concat([base, target_series], axis=1, **_CONCAT_NO_COPY)
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
    except Exception as e:
        logger.debug("could not resolve mlframe version: %s", e)
        versions["mlframe"] = "?"
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    from ..composite.discovery._algo_version import DISCOVERY_ALGO_VERSION

    # The selection logic's own version: a discovery fix inside a release must invalidate warm caches.
    versions["discovery_algo"] = str(DISCOVERY_ALGO_VERSION)
    # Distribution metadata, not ``__import__``: reading a version string must not load catboost / lightgbm / xgboost.
    for _name, _dist in (
        ("sklearn", "scikit-learn"), ("lightgbm", "lightgbm"), ("catboost", "catboost"), ("xgboost", "xgboost"),
        ("polars", "polars"), ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
    ):
        try:
            _ver_str = _dist_version(_dist)
            # Major.minor only -- patch bumps invalidate every cached spec even though MI /
            # Wilcoxon / boosting math is unchanged. Strip patch + any dev / rc tags.
            _parts = _ver_str.split(".")
            if len(_parts) >= 2 and _parts[0].isdigit():
                _ver_str = f"{_parts[0]}.{_parts[1].split('+')[0].split('rc')[0].split('dev')[0]}"
            versions[_name] = _ver_str
        except PackageNotFoundError:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            versions[_name] = "absent"
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    return compute_config_signature_v1(config, library_versions=versions)
