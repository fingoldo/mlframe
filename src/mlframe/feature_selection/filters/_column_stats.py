"""Per-column statistics MRMR already computed during ``fit``, returned as ``MRMR.column_stats_`` (a DataFrame, one row per column of the final code matrix) and ``MRMR.fit_meta_`` (a dict of fit-level values).

NOTHING is computed here: every number is read from a table the fit built for its own use (the code matrix's per-column bin counts, the screen's marginal-MI cache ``cached_MIs``, its confirmed-MI cache
``cached_confident_MIs`` and its entropy cache). A column the fit never scored in a table has ``NaN`` in that field.

META-LEARNING NOTE (MBHO, the hyper-parameter tuner): these rows are the intended training data of its meta-learner. One fit yields one (column descriptors -> fate of the column) table: how many bins the
column was cut into, how much information it carries about the target (``mi_with_target``, its confirmed form ``mi_confident`` with ``mi_confidence``), the entropy of its codes, and, after the fit, whether
the column was selected (``in_output``, ``support_rank``, ``mrmr_gain``, ``origin``, ``dropped_by`` joined by name from ``fe_provenance_``). ``fit_meta_`` carries the fit-level context that makes tables of different
fits comparable (rows, classes of the target and their entropy, the effective relevance gate, the binning strategy). Keep the field names stable: a tuner reading old tables relies on them.

Columns of ``column_stats_``:
  ``column``           name in the final column list (raw input columns, then engineered ones)
  ``position``         index of the column in the code matrix
  ``is_engineered``    True for a column built by the FE stages (not one of ``feature_names_in_``)
  ``nbins``            number of bins (distinct codes allowed) of the column in the code matrix
  ``mi_with_target``   marginal MI(column; target) in nats, from the screen's cache
  ``mi_confident``     the bootstrapped (permutation-confirmed) gain of the column, when the screen confirmed it
  ``mi_confidence``    the confidence of that gain
  ``entropy``          H(column codes) in nats, from the screen's entropy cache. That cache is filled only by the relevance / redundancy routes that need entropies of variable sets (fleuret, JMIM); on the
                       default route the entropies are formed inside the njit MI kernels and not kept, so the field is NaN there (nothing is recomputed to fill it)
  ``su``               symmetric uncertainty ``2 * MI / (H(column) + H(target))`` of the two numbers above (NaN when an entropy is missing)
  ``added_by_emit_both`` (True: a raw column in the output only because ``redundancy_policy='emit_both'`` re-attached it as an operand of a selected recipe; also listed in ``emit_both_added_``) and
  ``origin``, ``in_output``, ``support_rank``, ``mrmr_gain``, ``dropped_by``  from ``fe_provenance_`` once the fit has finished (see :func:`join_provenance`)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

__all__ = ["record_column_stats", "join_provenance"]

_PROVENANCE_FIELDS = ("origin", "in_output", "support_rank", "mrmr_gain", "dropped_by")


def _binning_meta(self) -> dict:
    """The resolved binning of this fit for ``fit_meta_``: scheme, count rule (the scheme's own when none was given), cap."""
    from ._binning_spec import spec_as_dict, spec_from_estimator

    try:
        return dict(spec_as_dict(spec_from_estimator(self)))
    except ValueError:
        return {"binning_scheme": getattr(self, "binning_scheme", None), "nbins_rule": getattr(self, "nbins_rule", None), "nbins_cap": getattr(self, "nbins_cap", None)}


def _confidence_meta(self) -> dict:
    """What ``mi_confidence`` (``column_stats_``) is in this fit. The serial confirmation (one worker, or a budget no larger than ``NMAX_NONPARALLEL_ITERS``) reports ``1 - nfailed / nchecked``, the raw rate of
    failed permutation re-checks of a tiny budget: ``1.0`` after a clean run of three, and not a p-value. The multi-worker confirmation reports one minus the add-one p-value of the full budget. The selection is
    the same either way; the flag only tells a reader which of the two numbers a ``mi_confidence`` is."""
    from ._internals import NMAX_NONPARALLEL_ITERS

    n_workers = getattr(self, "n_workers", 1)
    budget = getattr(self, "full_npermutations", 0)
    segmented = bool(n_workers and n_workers > 1 and budget and budget > NMAX_NONPARALLEL_ITERS)
    return {"mi_confidence_kind": "one_minus_add_one_pvalue" if segmented else "one_minus_permutation_failure_rate", "mi_confidence_is_pvalue": segmented}


def _probe_failures_this_fit(self) -> int:
    """How many accuracy-gate probes raised during this fit and kept their candidate (fail-open); -1 when the fit start was not marked."""
    from ._fe_accuracy_gate import probe_failure_count

    start = getattr(self, "_fe_probe_failures_at_start", None)
    return max(0, probe_failure_count() - int(start)) if start is not None else -1


def _entropy_of(entropy_cache: Any, position: int) -> float:
    """H(codes of column ``position``) from the screen's entropy cache (keyed by the underscore-joined variable indices), NaN when the screen never needed it."""
    if not entropy_cache:
        return float("nan")
    key = str(position)
    if key in entropy_cache:
        return float(entropy_cache[key])
    return float("nan")


def _target_entropy(freqs_y: Any) -> float:
    """Entropy in nats of the target's class frequencies (the vector the fit already holds)."""
    if freqs_y is None:
        return float("nan")
    p = np.asarray(freqs_y, dtype=np.float64)  # Intentional float64: entropy of class frequencies, a statistic
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else float("nan")


def record_column_stats(self: Any, cols: Any, nbins: Any, cached_MIs: Any, screen_caches: Any, freqs_y: Any, classes_y: Any, effective_min_relevance_gain: Any = None) -> None:
    """Fill ``self.column_stats_`` and ``self.fit_meta_`` from what the fit holds (called once, after the support is assigned). Never raises: metadata must not change a fit."""
    try:
        entropy_cache, confident = (screen_caches[0], screen_caches[2]) if screen_caches else (None, None)
        names = [str(c) for c in cols]
        names_in = getattr(self, "feature_names_in_", None)
        raw_names = set(str(c) for c in (names_in if names_in is not None else []))
        n = len(names)
        nb = np.asarray(nbins).ravel() if nbins is not None else np.full(n, np.nan)
        rows: "dict[str, Any]" = {
            "column": names,
            "position": list(range(n)),
            "is_engineered": [(nm not in raw_names) for nm in names],
            "nbins": [float(nb[i]) if i < nb.size else float("nan") for i in range(n)],
            "mi_with_target": [float(cached_MIs[(i,)]) if cached_MIs is not None and (i,) in cached_MIs else float("nan") for i in range(n)],
        }
        gain, conf = [], []
        for i in range(n):
            entry = confident.get((i,)) if confident else None
            gain.append(float(entry[0]) if entry is not None else float("nan"))
            conf.append(float(entry[1]) if entry is not None else float("nan"))
        rows["mi_confident"], rows["mi_confidence"] = gain, conf
        rows["entropy"] = [_entropy_of(entropy_cache, i) for i in range(n)]
        h_y = _target_entropy(freqs_y)
        rows["su"] = [
            (2.0 * mi / (h + h_y)) if (not math.isnan(mi) and not math.isnan(h) and not math.isnan(h_y) and (h + h_y) > 0.0) else float("nan")
            for mi, h in zip(rows["mi_with_target"], rows["entropy"])
        ]
        self.column_stats_ = pd.DataFrame(rows)
        self.fit_meta_ = {
            "n_rows": len(classes_y) if classes_y is not None else -1,
            "n_columns": n,
            "n_classes_y": len(freqs_y) if freqs_y is not None else -1,
            "target_entropy": h_y,
            "class_frequencies": [float(v) for v in np.asarray(freqs_y).ravel()] if freqs_y is not None else [],
            **_binning_meta(self),
            **_confidence_meta(self),
            "fe_probe_failures": _probe_failures_this_fit(self),
            "quantization_nbins": getattr(self, "quantization_nbins", None),
            "effective_min_relevance_gain": float(effective_min_relevance_gain) if effective_min_relevance_gain is not None else float("nan"),
            "fe_interaction_ii_floor": float(_ii_floor) if (_ii_floor := getattr(self, "fe_interaction_ii_floor_", None)) is not None else float("nan"),
        }
    except Exception as exc:  # metadata only
        self.column_stats_ = None
        self.fit_meta_ = None
        log_throttle(logger, "column_stats.record", logging.WARNING, "column_stats: the per-column statistics could not be collected (%r)", exc)


def join_provenance(self: Any) -> None:
    """Add the selection fate of every column to ``column_stats_`` (``origin``, ``in_output``, ``support_rank``, ``mrmr_gain``, ``dropped_by`` of ``fe_provenance_``, matched by feature name).
    Columns the provenance does not list keep NaN / None. Never raises."""
    try:
        stats = getattr(self, "column_stats_", None)
        if stats is not None:
            added = set(getattr(self, "emit_both_added_", None) or ())
            stats["added_by_emit_both"] = stats["column"].isin(added)  # True: the raw column is in the output only because redundancy_policy='emit_both' re-attached it as an operand of a selected recipe
        prov = getattr(self, "fe_provenance_", None)
        if stats is None or prov is None or not len(prov):
            return
        keyed = prov.drop_duplicates("feature_name").set_index("feature_name")
        for field in _PROVENANCE_FIELDS:
            if field in keyed.columns:
                stats[field] = stats["column"].map(keyed[field])
        if "in_output" in stats.columns:
            stats["in_output"] = stats["in_output"].fillna(False).astype(bool)  # a column the provenance does not list was not part of the output
    except Exception as exc:  # metadata only
        log_throttle(logger, "column_stats.join", logging.WARNING, "column_stats: the selection fate could not be joined (%r)", exc)
