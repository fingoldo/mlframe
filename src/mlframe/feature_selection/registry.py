"""Unified FeatureSelectorSpec protocol + registry for the training suite.

Each selector self-describes how to instantiate (``instantiate(**kwargs)``) and how to extract its
FS report (``report_extract(selector, kept)``, now consumed by ``_build_feature_selection_report``).

Reachability contract (NOT yet a single registration): the registry is the instantiation+report
dispatch table, but wiring a selector INTO the suite still also needs (a) a ``use_<sel>`` flag +
``<sel>_kwargs`` + validator in ``FeatureSelectionConfig``, (b) a branch in ``_build_pre_pipelines``,
and (c) a kind string in ``_selector_kind``. Each registered selector here is reachable from the suite
(MRMR / RFECV / BorutaShap / ShapProxiedFS all have their flag + branch). ``report_extract`` already
lives next to the spec, so the central report builder no longer hard-codes a branch per selector.

FUTURE (data-driven ``_build_pre_pipelines`` over ``registry.available()``): collapse the per-selector
branches into one loop by extending ``FeatureSelectorSpec`` with declarative wiring metadata --
``config_enable_field`` ("use_mrmr_fs"), ``config_kwargs_field`` ("mrmr_kwargs"), ``selector_kind``, and
a ``post_instantiate`` hook for the selector-specific stamping (MRMR's seed-default + identity-cache view,
RFECV's prebuilt-instance override + cluster-medoid wrap, BorutaShap/ShapProxiedFS's classification
auto-derive). Deferred because the four branches today have materially different wiring (RFECV is passed
in PREBUILT by configure_training_params, not instantiated from kwargs like the others), so a faithful
data-driven loop needs that per-spec hook surface designed + validated against the full suite test bed
first; not a safe in-place refactor for a wiring pass.

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
    # cluster-medoid pre-reduction is DEFAULT-ON for RFECV instantiated through THIS factory (MRMR / BorutaShap
    # ranker paths). The training suite does NOT build RFECV through this factory -- it constructs RFECV directly
    # in configure_training_params and wraps it in GroupAwareMRMR inside _build_pre_pipelines, driven by the
    # FeatureSelectionConfig.rfecv_cluster_* fields (so the default-ON behaviour holds for the suite RFECV too). Broad validation
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


def _report_extract_shap_proxied_fs(selector, kept) -> dict:
    """Per-feature ShapProxiedFS report fragment consumed by ``_build_feature_selection_report``.

    Surfaces the mean-|phi| importances the selector ranked subsets by (when present) as ``scores``,
    and a kept/dropped reason map. Kept/dropped names are computed by the central builder; this only
    adds selector-specific signal. Every read is defensive: a failed extraction must never abort training.
    """
    out: dict = {"scores": None, "reason_per_feature": None}
    try:
        _rep = getattr(selector, "shap_proxy_report_", None)
        if isinstance(_rep, dict):
            _imp = _rep.get("mean_abs_shap") or _rep.get("importances")
            if isinstance(_imp, dict) and _imp:
                out["scores"] = {str(k): float(v) for k, v in _imp.items()}
    except Exception:
        out["scores"] = None
    try:
        _sel = set(str(c) for c in (getattr(selector, "selected_features_", None) or []))
        _all = getattr(selector, "feature_names_in_", None)
        if _all is not None and _sel:
            out["reason_per_feature"] = {str(c): ("selected" if str(c) in _sel else "dropped") for c in _all}
    except Exception:
        pass
    return out


register(_SimpleSpec(name="MRMR", instantiate=_instantiate_mrmr))
register(_SimpleSpec(name="RFECV", instantiate=_instantiate_rfecv))
register(_SimpleSpec(name="BorutaShap", instantiate=_instantiate_boruta_shap))
# ShapProxiedFS is reachable from the suite via ``FeatureSelectionConfig.use_shap_proxied_fs`` +
# the matching ``_build_pre_pipelines`` branch (mirrors BorutaShap). The registration also carries
# ``report_extract`` so the central report builder picks up ShapProxiedFS scores without a hard-coded branch.
register(_SimpleSpec(name="ShapProxiedFS", instantiate=_instantiate_shap_proxied_fs, report_extract=_report_extract_shap_proxied_fs))
