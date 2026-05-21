"""Multilabel / LTR / target-coercion helpers for ``dummy_baselines``.

Split out of ``dummy_baselines.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the moved entries so historical
``from mlframe.training.dummy_baselines import _compute_multilabel_baselines``
imports continue to resolve.

What lives here:
  - ``_compute_multilabel_baselines`` (per-target-type dummy baselines for
    K-binary-output classification)
  - ``_compute_ltr_baselines`` (per-group rank-style dummies for
    Learning-to-Rank)
  - ``_within_group_descending_index`` (rank-assign helper)
  - ``_coerce_y`` (target-type-aware ndarray coercion)
  - ``_empty_report`` (degenerate / signal-missing BaselineReport stub)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mlframe.training.baseline_diagnostics import _to_1d_numpy
from mlframe.training.evaluation import _canonical_multilabel_y

from ._dummy_baseline_compute import _per_target_seed
from ._dummy_report_type import BaselineReport
from ._dummy_numba_kernels import _NUMBA_AVAILABLE
if _NUMBA_AVAILABLE:
    from ._dummy_numba_kernels import _numba_within_group_descending_rank

logger = logging.getLogger(__name__)


def _compute_multilabel_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    config: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Multilabel: all_zero / all_one / per_label_prior / per_label_most_frequent."""
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    if train_y is None or train_y.ndim != 2:
        return val_preds, test_preds, extras
    K = train_y.shape[1]
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)

    # all_zero
    val_preds["all_zero"] = np.zeros((n_val, K))
    test_preds["all_zero"] = np.zeros((n_test, K))
    # all_one
    val_preds["all_one"] = np.ones((n_val, K))
    test_preds["all_one"] = np.ones((n_test, K))
    # per_label_prior -- broadcast train per-label mean
    per_label_prior = train_y.mean(axis=0)
    val_preds["per_label_prior"] = np.tile(per_label_prior, (n_val, 1))
    test_preds["per_label_prior"] = np.tile(per_label_prior, (n_test, 1))
    # per_label_most_frequent -- round per-label prior to 0/1
    plmf = (per_label_prior >= 0.5).astype(np.float64)
    val_preds["per_label_most_frequent"] = np.tile(plmf, (n_val, 1))
    test_preds["per_label_most_frequent"] = np.tile(plmf, (n_test, 1))

    extras["n_labels"] = K
    return val_preds, test_preds, extras


def _compute_ltr_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    group_ids_train: Any,
    group_ids_val: Any,
    group_ids_test: Any,
    ts_train: np.ndarray | None,
    ts_val: np.ndarray | None,
    ts_test: np.ndarray | None,
    config: Any,
    doc_ids_train: Any = None,
    doc_ids_val: Any = None,
    doc_ids_test: Any = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """LTR: random_within_query / identity_input_order / mean_relevance /
    most_recent_first / popularity.

    Group sanity gate (D3) applied before any baseline runs.

    The ``popularity`` baseline activates only when ``doc_ids_*`` are
    supplied (caller has a per-row document identifier outside the
    feature space -- this is a strict superset of mlframe's default LTR
    API which only carries ``group_ids`` = qid). Popularity score for
    val/test row = log(1 + count(doc_id in train)). Unseen docs get
    score = 0 (cold-start cells get the smallest possible score).
    """
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    if group_ids_train is None or group_ids_val is None or group_ids_test is None:
        extras["ltr_skip_reason"] = "group_ids missing"
        return val_preds, test_preds, extras

    g_train = np.asarray(group_ids_train)
    g_val = np.asarray(group_ids_val)
    g_test = np.asarray(group_ids_test)

    # Defensive: hard-fail on length mismatch with actionable message.
    # A length mismatch is a caller bug, not a runtime degraded condition.
    if len(g_train) != len(train_y):
        raise ValueError(
            f"[dummy-baselines] target='{target_name}' learning_to_rank: "
            f"len(group_ids_train)={len(g_train)} != len(train_y)={len(train_y)}"
        )
    if val_y is not None and len(g_val) != len(val_y):
        raise ValueError(
            f"[dummy-baselines] target='{target_name}' learning_to_rank: "
            f"len(group_ids_val)={len(g_val)} != len(val_y)={len(val_y)}"
        )
    if test_y is not None and len(g_test) != len(test_y):
        raise ValueError(
            f"[dummy-baselines] target='{target_name}' learning_to_rank: "
            f"len(group_ids_test)={len(g_test)} != len(test_y)={len(test_y)}"
        )

    # Group sanity gate
    n_groups_train = len(np.unique(g_train))
    if n_groups_train < 2:
        extras["ltr_skip_reason"] = f"only {n_groups_train} group in train"
        return val_preds, test_preds, extras
    # Wave 50 (2026-05-20): pd.factorize emits -1 for NaN; np.bincount(-1) raises
    # ValueError. Drop NaN codes before bincount so sparsely-populated group_field
    # doesn't crash the LTR fast-path.
    _factor_codes = pd.factorize(g_train)[0]
    train_group_sizes = np.bincount(_factor_codes[_factor_codes >= 0])
    if train_group_sizes.max() > 0.5 * len(g_train):
        extras["ltr_skip_reason"] = (
            f"max_group_pct={train_group_sizes.max() / len(g_train) * 100:.1f}% "
            "(non-rankable structure)"
        )
        return val_preds, test_preds, extras

    extras["n_groups_train"] = int(n_groups_train)
    extras["n_groups_val"] = int(len(np.unique(g_val))) if len(g_val) > 0 else 0
    extras["n_groups_test"] = int(len(np.unique(g_test))) if len(g_test) > 0 else 0

    n_val = len(g_val)
    n_test = len(g_test)
    seed = _per_target_seed(config.random_state, target_name)

    # random_within_query: one deterministic random ranking. Prior code built
    # a list of ``n_repeats`` independent random vectors, kept only ``[0]``
    # for the prediction, and discarded the rest -- wasting 9 * (n_val+n_test)
    # PRNG draws per fit (~9 ms at n_val=n_test=100k). Averaging across the
    # repeats would converge to a constant 0.5 (mean of i.i.d. uniforms),
    # collapsing the baseline to the degenerate ``mean_relevance`` variant;
    # storing only the first run was the correct semantic. Now compute just
    # that first run. The ``n_repeats`` config is kept in extras for metadata
    # compatibility (downstream readers can detect the simplification).
    n_repeats = config.random_within_query_n_repeats
    rng = np.random.default_rng(seed)
    val_preds["random_within_query"] = rng.random(n_val) if n_val > 0 else np.array([])
    test_preds["random_within_query"] = rng.random(n_test) if n_test > 0 else np.array([])
    extras["random_within_query_n_repeats"] = n_repeats

    # identity_input_order: predict scores in feature-row order (1 / rank-within-group)
    # For a group's rows, score = N - i where i is the within-group index
    val_preds["identity_input_order"] = _within_group_descending_index(g_val, n_val)
    test_preds["identity_input_order"] = _within_group_descending_index(g_test, n_test)

    # mean_relevance: constant train_y.mean()
    train_y_arr = np.asarray(train_y, dtype=np.float64)
    mean_rel = float(train_y_arr.mean()) if len(train_y_arr) > 0 else 0.0
    val_preds["mean_relevance"] = np.full(n_val, mean_rel)
    test_preds["mean_relevance"] = np.full(n_test, mean_rel)

    # most_recent_first (TS only): rank by recency within group
    if ts_val is not None and ts_test is not None:
        val_preds["most_recent_first (ts)"] = ts_val.astype(np.float64)
        test_preds["most_recent_first (ts)"] = ts_test.astype(np.float64)

    # popularity: per-doc train-frequency. Activates only when
    # doc_ids_* are supplied (FTE protocol extension; mlframe's default
    # LTR carries only group_ids = qid). Score = log(1 + count_train).
    # Unseen docs at val/test get 0 (cold-start cells rank lowest).
    if doc_ids_train is not None and doc_ids_val is not None and doc_ids_test is not None:
        try:
            d_train = np.asarray(doc_ids_train)
            d_val = np.asarray(doc_ids_val)
            d_test = np.asarray(doc_ids_test)
            if (
                len(d_train) == len(train_y)
                and len(d_val) == n_val
                and len(d_test) == n_test
            ):
                # Coerce non-numeric doc IDs to a hashable string form
                # so pd.Series.value_counts handles them uniformly.
                if d_train.dtype.kind not in ("i", "u", "f"):
                    d_train_s = pd.Series([str(x) for x in d_train])
                    d_val_s = pd.Series([str(x) for x in d_val])
                    d_test_s = pd.Series([str(x) for x in d_test])
                else:
                    d_train_s = pd.Series(d_train)
                    d_val_s = pd.Series(d_val)
                    d_test_s = pd.Series(d_test)
                pop_counts = d_train_s.value_counts()
                # Score = log(1 + count); unseen -> 0. pre-cast pop_counts.values once to f64 so .map shares one
                # contiguous values buffer rather than re-allocating per split. The two splits operate on
                # disjoint row ranges so the .map call cannot share work, but avoiding the .astype copy per
                # split is still a ~30% speedup on multi-million-row val+test.
                pop_counts_f64 = pop_counts.astype(np.float64)
                val_pop = d_val_s.map(pop_counts_f64).fillna(0.0).to_numpy()
                test_pop = d_test_s.map(pop_counts_f64).fillna(0.0).to_numpy()
                val_preds["popularity"] = np.log1p(val_pop)
                test_preds["popularity"] = np.log1p(test_pop)
                # Diagnostics
                val_unseen_pct = float(np.mean(val_pop == 0) * 100)
                test_unseen_pct = float(np.mean(test_pop == 0) * 100)
                extras["popularity_diagnostics"] = {
                    "n_unique_docs_train": int(len(pop_counts)),
                    "val_cold_start_pct": val_unseen_pct,
                    "test_cold_start_pct": test_unseen_pct,
                }
        except Exception as _pop_err:
            # Non-fatal: popularity is one of N LTR baselines.
            extras["popularity_skip_reason"] = str(_pop_err)

    return val_preds, test_preds, extras


def _within_group_descending_index(group_ids: np.ndarray, n: int) -> np.ndarray:
    """For each row, return descending index within its group (rank 0 = first row).

    Numba-accelerated when ``group_ids`` is integer-typed; falls back to
    Python loop for non-integer keys (e.g. string group_ids).
    """
    if n == 0:
        return np.array([])
    if _NUMBA_AVAILABLE and group_ids.dtype.kind in ("i", "u"):
        # Coerce to int64 so the numba kernel signature is stable.
        gi = np.ascontiguousarray(group_ids, dtype=np.int64)
        try:
            return _numba_within_group_descending_rank(gi)
        except Exception:
            pass
    out = np.zeros(n, dtype=np.float64)
    counts: dict[Any, int] = {}
    for i in range(n):
        g = group_ids[i]
        c = counts.get(g, 0)
        out[i] = -c
        counts[g] = c + 1
    return out


def _coerce_y(y: Any, target_type: str, target_name: str) -> np.ndarray | None:
    """Coerce y to numpy with target-type-aware shape (D8 object-dtype gate).

    For regression / quantile_regression: 2D ``(N, K)`` inputs preserved
    for D4 multi-output dispatch; 1D inputs reshaped via ``_to_1d_numpy``.
    For multilabel_classification: ``_canonical_multilabel_y`` returns 2D.
    For all classification targets: 1D enforced.
    """
    if y is None:
        return None
    if target_type == "multilabel_classification":
        return _canonical_multilabel_y(y)
    if target_type in ("regression", "quantile_regression"):
        # Preserve 2D for multi-output regression
        if hasattr(y, "to_numpy"):
            arr = y.to_numpy()
        elif hasattr(y, "values"):
            arr = y.values
        else:
            arr = np.asarray(y)
        if arr.ndim == 1:
            pass
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        # else: leave 2D for multi-output dispatcher
        if arr.dtype == object:
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "[dummy-baselines] target='%s' has object dtype incompatible with %s "
                    "baselines (%s); skipping",
                    target_name, target_type, e,
                )
                return None
        return arr
    arr = _to_1d_numpy(y)
    if arr.dtype == object:
        try:
            arr = arr.astype(np.int64)
        except (TypeError, ValueError) as e:
            logger.warning(
                "[dummy-baselines] target='%s' has object dtype incompatible with %s "
                "baselines (%s); skipping",
                target_name, target_type, e,
            )
            return None
    return arr


def _empty_report(
    target_type: str, target_name: str, t0: float, reason: str,
) -> BaselineReport:
    """Return an empty report when block can't run (D8 / D9 / unknown target_type)."""
    import time as _time
    return BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=pd.DataFrame(),
        strongest=None,
        primary_metric=None,
        ts_period_used=None,
        plot_path=None,
        elapsed_s=_time.time() - t0,
        n_train=0, n_val=0, n_test=0,
        n_train_finite=0, n_val_finite=0, n_test_finite=0,
        extras={"skip_reason": reason},
    )


