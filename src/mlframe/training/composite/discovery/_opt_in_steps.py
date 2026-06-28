"""Opt-in discovery steps that wire the standalone discovery helpers into the
main ``CompositeTargetDiscovery.fit`` flow.

Three capabilities exist as committed-but-research-only standalone helpers and
were never reachable from a plain ``fit`` call:

* **region-adaptive** (``_region_adaptive.fit_region_adaptive``) -- per-region
  best-transform selection routed by frozen quantile edges of the base.
* **interaction-base discovery** (``_interaction_bases.discover_interaction_bases``)
  -- surface ``a OP b`` synthetic bases whose interaction MI beats both marginals.
* **auto transform-chaining** (``_auto_chain.discover_chains``) -- compose every
  ``residual x tail-unary`` chain and keep those that beat both single stages on
  held-out y-scale RMSE.

``run_optional_discovery_steps`` runs ONLY the enabled steps (each gated by a
config flag that defaults ``True``) over the already-kept single-base
specs, stashes the per-step rich artefacts on dedicated instance attributes
(``region_adaptive_specs_`` / ``interaction_bases_`` / ``auto_chains_``), and
returns a list of well-formed extra :class:`CompositeSpec` objects the caller
appends to ``kept_specs``.

Flag discipline (CRITICAL)
--------------------------
All three flags default ``True`` (each step has test-confirmed business value).
With all three explicitly set ``False`` this function is a flag-gated no-op: it
returns an empty list and sets each artefact attribute to its empty default, so
the discovered ``specs_`` / ``report_`` are byte-identical to the pre-hook flow.

Leakage / RAM discipline (CRITICAL)
-----------------------------------
Every step reads only ``train_idx`` rows via the narrow ``_extract_column_array``
column pull (one ndarray per column, never a whole-frame copy) and a bounded
``mi_sample_n`` screening sample. The auto-chain CV / region OOF splits are
internal to the helpers (train-only by construction). No test/val row is read.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from pyutilz.parallel import cpu_count_physical

from ..spec import CompositeSpec
from ..transforms import compose_target_name
from ..transforms.registry import _TRANSFORMS_REGISTRY
from .screening import _extract_column_array, _sample_indices
from ._region_adaptive import fit_region_adaptive
from ._interaction_bases import discover_interaction_bases
from ._auto_chain import discover_chains

logger = logging.getLogger(__name__)


def _run_region_adaptive(
    self: Any, df: Any, target_col: str, kept_specs: Sequence[CompositeSpec],
    screen_idx: np.ndarray, y_screen: np.ndarray,
) -> list[Any]:
    """Fit a :class:`RegionAdaptiveSpec` per kept single-base residual spec.

    Region-adaptive needs ONE base column, so it runs only on specs that carry a
    single base (``requires_base`` transforms with no multi-base extras). The
    rich :class:`RegionAdaptiveSpec` objects are returned for the caller to stash
    on ``region_adaptive_specs_``; they are NOT appended to ``specs_`` (their
    forward/inverse signature differs from the registry ``Transform`` contract
    ``iter_transform`` consumes).
    """
    k = int(getattr(self.config, "region_adaptive_k", 4))
    rs = int(self.config.random_state)
    # Unique single-base columns, in first-seen order (region-adaptive needs exactly one base).
    bases: list[str] = []
    seen_bases: set[str] = set()
    for spec in kept_specs:
        base_col = getattr(spec, "base_column", "")
        if not base_col or getattr(spec, "extra_base_columns", ()):
            continue
        if base_col not in seen_bases:
            seen_bases.add(base_col)
            bases.append(base_col)
    if not bases:
        return []

    def _fit_one(base_col: str):
        try:
            base_screen = _extract_column_array(df, base_col, rows=screen_idx)
            return fit_region_adaptive(
                y_screen, base_screen,
                base_column=base_col, target_col=target_col,
                k=k, random_state=rs,
            )
        except Exception as exc:  # noqa: BLE001 -- a degenerate base must not abort fit
            logger.warning(
                "[CompositeTargetDiscovery.region_adaptive] base=%s failed: %s",
                base_col, exc,
            )
            return None

    # Per-base fits are independent; parallelise across physical cores (threading backend -- the
    # tiny RegionAdaptive fits release the GIL). joblib preserves input order, so ``out`` matches
    # the serial order. Mutation-free workers: results are reduced on the main thread.
    n_jobs = min(len(bases), cpu_count_physical())
    if n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(_fit_one)(b) for b in bases
        )
    else:
        results = [_fit_one(b) for b in bases]
    out: list[Any] = [r for r in results if r is not None]
    if out:
        logger.info(
            "[CompositeTargetDiscovery.region_adaptive] fitted %d region-adaptive "
            "spec(s) over base(s): %s", len(out), sorted(seen_bases),
        )
    return out


def _run_interaction_bases(
    self: Any, df: Any, screen_idx: np.ndarray, y_screen: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Surface synthetic ``a OP b`` interaction bases from the auto-base pool.

    Candidates are the train-row-restricted auto-base columns (``_auto_base_pool``)
    sampled to the screen rows. Returns ``({synth_name -> screen-row ndarray},
    [score_record])`` for the caller to stash on ``interaction_bases_``. These are
    new BASE candidates (not specs); they are reported but not appended to
    ``specs_`` (turning them into specs needs a full re-screen, out of scope for
    the cheap opt-in hook).
    """
    pool: dict[str, np.ndarray] = getattr(self, "_auto_base_pool", {}) or {}
    if len(pool) < 2:
        return {}, []
    # Map each pooled train-row array down to the screen rows. The pool stores
    # arrays already restricted to ``train_idx``; the screen sample is a subset of
    # those positions, so reuse the SAME relative offsets the main screen used.
    rel = self._screen_sample_rel_idx
    candidates: dict[str, np.ndarray] = {}
    for name, arr in pool.items():
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        if rel is not None and rel.max(initial=-1) < a.shape[0]:
            candidates[name] = a[rel]
        elif a.shape[0] == y_screen.shape[0]:
            candidates[name] = a
    candidates = {n: c for n, c in candidates.items() if c.shape == y_screen.shape}
    if len(candidates) < 2:
        return {}, []
    top_k = int(getattr(self.config, "interaction_base_top_k", 4))
    max_pairs = int(getattr(self.config, "interaction_base_max_pairs", 3))
    synth, records = discover_interaction_bases(
        candidates, y_screen, top_k=top_k, max_pairs=max_pairs,
        nbins=int(self.config.mi_nbins),
    )
    return synth, records


def _run_auto_chain(
    self: Any, df: Any, target_col: str, kept_specs: Sequence[CompositeSpec],
    usable_features: Sequence[str], screen_idx: np.ndarray, y_screen: np.ndarray,
) -> list[CompositeSpec]:
    """Discover ``residual x tail-unary`` chains that beat both single stages.

    Runs per distinct single base among the kept residual specs. A winning
    :class:`ChainCandidate` carries a real composed ``Transform``; we register it
    into ``_TRANSFORMS_REGISTRY`` (so ``iter_transform`` / downstream
    ``get_transform`` resolve it) and emit a well-formed :class:`CompositeSpec`
    the caller appends to ``kept_specs``.
    """
    res_names = sorted({
        s.transform_name for s in kept_specs
        if s.transform_name in ("linear_residual", "monotonic_residual")
    })
    if not res_names:
        return []
    bases = []
    seen: set[str] = set()
    for s in kept_specs:
        bc = getattr(s, "base_column", "")
        if bc and bc not in seen and not getattr(s, "extra_base_columns", ()):
            seen.add(bc)
            bases.append(bc)
    if not bases:
        return []
    feat_cols = [c for c in usable_features if c != target_col]
    if not feat_cols:
        return []
    # Build the screen-sample feature matrix ONCE (per-column pull from the Polars/pandas frame),
    # then derive each base's "all features except this base" matrix via np.delete on the in-RAM
    # matrix -- bit-identical to a per-base ``column_stack`` of ``feat_cols`` minus the base (the
    # surviving column order is preserved), but it re-gathers ~400 columns from the frame ONCE
    # instead of once per base.
    x_full = np.column_stack(
        [_extract_column_array(df, c, rows=screen_idx) for c in feat_cols]
    )
    col_index = {c: i for i, c in enumerate(feat_cols)}

    def _chains_for_base(base_col: str):
        # x_cols == feat_cols minus this base; empty only when the base is the sole feature.
        if len(feat_cols) == 1 and feat_cols[0] == base_col:
            return base_col, []
        try:
            base_screen = _extract_column_array(df, base_col, rows=screen_idx)
            x_matrix = (
                np.delete(x_full, col_index[base_col], axis=1)
                if base_col in col_index else x_full
            )
            chains = discover_chains(
                y=y_screen, base=base_screen, x_matrix=x_matrix,
                residual_names=res_names,
                min_valid_domain_frac=float(self.config.min_valid_domain_frac),
                cv_folds=int(self.config.tiny_model_cv_folds),
                random_state=int(self.config.random_state),
                mi_estimator=self.config.mi_estimator,
                mi_nbins=int(self.config.mi_nbins),
                top_k=int(getattr(self.config, "auto_chain_top_k", 2)),
            )
            return base_col, chains
        except Exception as exc:  # noqa: BLE001 -- a degenerate base must not abort fit
            logger.warning(
                "[CompositeTargetDiscovery.auto_chain] base=%s failed: %s",
                base_col, exc,
            )
            return base_col, []

    # Per-base chain searches are independent (``discover_chains`` is serial internally); run them
    # across physical cores. Workers stay mutation-free -- registry / provenance / diag writes all
    # happen on the main thread below so the shared ``_TRANSFORMS_REGISTRY`` is never raced.
    n_jobs = min(len(bases), cpu_count_physical())
    if n_jobs > 1:
        from joblib import Parallel, delayed
        per_base = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(_chains_for_base)(b) for b in bases
        )
    else:
        per_base = [_chains_for_base(b) for b in bases]

    extra_specs: list[CompositeSpec] = []
    self._auto_chains_diag: list[Any] = []
    from ..provenance import register_chain_provenance
    for base_col, chains in per_base:
        for cand in chains:
            # Register the composed transform so name-based lookup resolves it,
            # plus its provenance so the chain self-describes in reports (and is
            # not a coverage gap now that auto-chaining is default-ON).
            _TRANSFORMS_REGISTRY.setdefault(cand.chain_name, cand.transform)
            register_chain_provenance(cand.chain_name, cand.residual_name, cand.unary_name)
            spec = CompositeSpec(
                name=compose_target_name(target_col, cand.chain_name, base_col),
                target_col=target_col,
                transform_name=cand.chain_name,
                base_column=base_col,
                fitted_params=dict(cand.fitted_params),
                mi_gain=float(cand.mi_gain) if np.isfinite(cand.mi_gain) else 0.0,
                mi_y=float("nan"),
                mi_t=float("nan"),
                valid_domain_frac=float(cand.valid_domain_frac),
                n_train_rows=int(screen_idx.size),
            )
            extra_specs.append(spec)
            self._auto_chains_diag.append(cand)
    if extra_specs:
        logger.info(
            "[CompositeTargetDiscovery.auto_chain] surfaced %d chain spec(s): %s",
            len(extra_specs), [s.name for s in extra_specs],
        )
    return extra_specs


def run_optional_discovery_steps(
    self: Any,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    kept_specs: Sequence[CompositeSpec],
    config: Any,
) -> list[CompositeSpec]:
    """Run the enabled opt-in discovery steps; return extra appendable specs.

    Gated by three config flags (all default ``True``):

    * ``region_adaptive_enabled`` -> ``region_adaptive_specs_`` artefact.
    * ``interaction_base_discovery_enabled`` -> ``interaction_bases_`` artefact.
    * ``auto_chain_discovery_enabled`` -> appendable chain :class:`CompositeSpec`s.

    With all three off this is a no-op returning ``[]`` (and the empty-default
    artefacts), so the discovered specs are byte-identical to the pre-hook flow.
    Each step is defensively isolated: a step that raises logs a warning and is
    skipped, never aborting ``fit``.
    """
    # Empty-default artefacts so attribute access is always safe + the OFF path
    # is observably a no-op.
    self.region_adaptive_specs_ = []
    self.interaction_bases_ = {}
    self.interaction_base_records_ = []
    self.auto_chains_ = []

    ra_on = bool(getattr(config, "region_adaptive_enabled", False))
    ib_on = bool(getattr(config, "interaction_base_discovery_enabled", False))
    ac_on = bool(getattr(config, "auto_chain_discovery_enabled", False))
    if not (ra_on or ib_on or ac_on) or not kept_specs:
        return []

    train_idx = np.asarray(train_idx)
    y_full = _extract_column_array(df, target_col)
    y_train = y_full[train_idx]
    # Build the screen sample once and remember the train-relative offsets so the
    # interaction step can index ``_auto_base_pool`` (already train-restricted).
    sample_idx = _sample_indices(
        train_idx.size, config.mi_sample_n, config.random_state,
        strategy=getattr(config, "mi_sample_strategy", "random"),
        y=y_train,
        n_strata=getattr(config, "mi_n_strata", 10),
    )
    self._screen_sample_rel_idx = np.asarray(sample_idx)
    screen_idx = train_idx[sample_idx]
    y_screen = y_full[screen_idx]

    extra: list[CompositeSpec] = []
    if ra_on:
        self.region_adaptive_specs_ = _run_region_adaptive(
            self, df, target_col, kept_specs, screen_idx, y_screen,
        )
    if ib_on:
        synth, records = _run_interaction_bases(self, df, screen_idx, y_screen)
        self.interaction_bases_ = synth
        self.interaction_base_records_ = records
    if ac_on:
        chain_specs = _run_auto_chain(
            self, df, target_col, kept_specs, feature_cols,
            screen_idx, y_screen,
        )
        self.auto_chains_ = list(getattr(self, "_auto_chains_diag", []))
        extra.extend(chain_specs)
    return extra
