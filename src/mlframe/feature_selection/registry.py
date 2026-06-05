"""Unified FeatureSelectorSpec protocol + registry for the training suite.

Replaces the per-selector dispatch in ``_build_pre_pipelines``: each selector self-describes
how to instantiate (``instantiate(**kwargs)``) and how to extract its FS report
(``report_extract(selector, kept)``). Adding a fourth selector (sklearn RFE, boruta-py, etc.)
becomes a single class registration instead of touching five edit sites across
_build_pre_pipelines / FeatureSelectionConfig / validators / _selector_kind / _build_feature_selection_report.

Lazy imports inside ``instantiate`` keep the cold-import cost (MRMR ~25s,
BorutaShap ~2s from shap/matplotlib/seaborn) gated behind the opt-in flag, preserving
the existing import-cost contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class FeatureSelectorSpec(Protocol):
    """Protocol every registered selector must satisfy.

    name : stable identifier used for selector_kind marker stamping and report keys.
    instantiate(**kwargs) : factory producing a fitted-ready selector instance from suite kwargs.
    report_extract(selector, kept) : optional. Returns a dict shape consumed by
        ``_build_feature_selection_report``. Default impl (None) signals the caller to use
        the legacy generic extraction. Per-selector custom extraction lives here so adding
        a fourth selector doesn't require editing the central report builder.
    """

    name: str
    instantiate: Callable[..., Any]
    report_extract: Callable[[Any, list[str]], dict] | None


@dataclass(frozen=True)
class _SimpleSpec:
    """Concrete FeatureSelectorSpec implementation used by the built-in registrations."""
    name: str
    instantiate: Callable[..., Any]
    report_extract: Callable[[Any, list[str]], dict] | None = None


_REGISTRY: dict[str, _SimpleSpec] = {}


def register(spec: FeatureSelectorSpec) -> None:
    """Register a selector. Overwrites any prior registration under the same name."""
    if not getattr(spec, "name", None):
        raise ValueError("FeatureSelectorSpec must have a non-empty 'name'")
    _REGISTRY[spec.name] = _SimpleSpec(
        name=spec.name,
        instantiate=spec.instantiate,
        report_extract=getattr(spec, "report_extract", None),
    )


def get(name: str) -> FeatureSelectorSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown feature selector spec: {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    """Sorted list of registered selector names."""
    return sorted(_REGISTRY)


def _instantiate_mrmr(**kwargs):
    # Deferred import: see _setup_helpers.py comment about MRMR's ~10-25s cold-import cost.
    from mlframe.feature_selection.filters import MRMR
    return MRMR(**kwargs)


def _instantiate_rfecv(**kwargs):
    # Go through the public re-export so the underscore module remains an implementation detail.
    from mlframe.feature_selection.wrappers import RFECV
    # 2026-06-03 (audit integration-defaults-3): cluster-medoid pre-reduction is
    # DEFAULT-ON for the suite's RFECV. Broad validation
    # (bench_cross_selector_diverse: synthetic make_classification with varied
    # redundancy + a signal-in-non-medoid risk case + real breast_cancer / wine /
    # digits) showed OOS AUC delta in [-0.0004, +0.0081] -- never materially
    # hurts -- with ~1.4-1.9x wall-clock on genuinely correlated data. The
    # GroupAwareMRMR guard bypasses the medoid path (running the bare RFECV on
    # full X) whenever the clustering eliminates < cluster_min_reduction of the
    # features, so it is a no-op on near-uncorrelated data. expand=True keeps
    # whole clusters (AUC-safe: a selected medoid drags in its members, so a
    # signal that lives in a non-medoid member is never dropped). Set
    # ``cluster_reduce=False`` to get the bare RFECV.
    cluster_reduce = bool(kwargs.pop("cluster_reduce", True))
    corr_threshold = float(kwargs.pop("cluster_corr_threshold", 0.9))
    min_reduction = float(kwargs.pop("cluster_min_reduction", 0.05))
    # ``cluster_corr_method`` (pearson | spearman | kendall | su). Default "pearson". The SU option captures
    # non-linear redundancy the corr methods miss; benched in _benchmarks/bench_medoid_corr_method.py before
    # any default flip -- on the broad bench Pearson and SU tied on OOS within noise so the cheaper Pearson
    # stays the default; pin "su" for known non-monotone-redundancy data.
    corr_method = str(kwargs.pop("cluster_corr_method", "pearson"))
    base = RFECV(**kwargs)
    if not cluster_reduce:
        return base
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    return GroupAwareMRMR(
        base, corr_threshold=corr_threshold, corr_method=corr_method,
        expand=True, min_reduction=min_reduction,
    )


def _instantiate_boruta_shap(**kwargs):
    # Lazy import: BorutaShap pulls in shap + matplotlib + seaborn (~2s).
    from mlframe.feature_selection.boruta_shap import BorutaShap
    # 2026-06-03 (audit integration-defaults-3): cluster-medoid pre-reduction
    # default-ON for BorutaShap too. Validated SAFE -- bench_boruta_shap_medoid
    # (synthetic varied-redundancy + real breast_cancer) gives OOS AUC delta in
    # [+0.0000, +0.0005] (never hurts; the shadow-importance null behaves --
    # collapsing redundant copies to one medoid CLEANS the per-feature SHAP test
    # rather than diluting it across near-duplicates). Speedup is modest here but
    # grows with redundancy width; the min_reduction guard makes it a no-op (bare
    # BorutaShap on full X) when clustering reduces little. GroupAwareMRMR reads
    # BorutaShap's ``accepted`` names via _inner_support_indices. cluster_reduce=
    # False restores bare BorutaShap.
    cluster_reduce = bool(kwargs.pop("cluster_reduce", True))
    corr_threshold = float(kwargs.pop("cluster_corr_threshold", 0.9))
    min_reduction = float(kwargs.pop("cluster_min_reduction", 0.05))
    corr_method = str(kwargs.pop("cluster_corr_method", "pearson"))
    base = BorutaShap(**kwargs)
    if not cluster_reduce:
        return base
    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR
    return GroupAwareMRMR(
        base, corr_threshold=corr_threshold, corr_method=corr_method,
        expand=True, min_reduction=min_reduction,
    )


def _instantiate_shap_proxied_fs(**kwargs):
    # Lazy import: ShapProxiedFS pulls in shap + a tree booster on first fit.
    # 2026-06-03 (audit integration-defaults-3): ShapProxiedFS is intentionally
    # NOT wrapped in the cluster-medoid reduction -- it ALREADY clusters
    # internally (cluster_correlated_features_su -> build_unit_matrix collapses
    # each correlated cluster to one denoised "unit" before the SHAP-proxy
    # selection; see shap_proxied_fs.py). Wrapping it would double-cluster.
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    return ShapProxiedFS(**kwargs)


register(_SimpleSpec(name="MRMR", instantiate=_instantiate_mrmr))
register(_SimpleSpec(name="RFECV", instantiate=_instantiate_rfecv))
register(_SimpleSpec(name="BorutaShap", instantiate=_instantiate_boruta_shap))
# Opt-in only: registration does NOT auto-wire ShapProxiedFS into the training suite (each selector
# is gated behind its own explicit flag in _setup_helpers_pre_pipelines); this just makes it
# discoverable via registry.get("ShapProxiedFS").
register(_SimpleSpec(name="ShapProxiedFS", instantiate=_instantiate_shap_proxied_fs))
