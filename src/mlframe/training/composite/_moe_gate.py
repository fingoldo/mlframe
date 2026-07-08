"""Mixture-of-experts SELECTION gate for composite targets: never-worse-than-lag routing.

On a strong-AR target the composite point prediction ``y_hat = inverse(T_hat, base)`` can be WORSE than the
``lag_predict`` failsafe (``y_hat = y_prev``) on some rows / groups, yet still get deployed and drag the
ensemble down. This gate takes the validation-split predictions of the three experts -- ``composite``,
raw-y model (``raw``), and the ``lag`` failsafe -- plus the true ``y`` (and optional group ids), LEARNS a
per-group (fallback: global) choice of which expert to trust, and exposes a predict-time router that
deploys that choice with a hard guarantee.

Selection rule (per group ``g``, on the fit / selection split)
--------------------------------------------------------------
* Matched set ``M_g`` = rows of ``g`` where ``y`` and EVERY expert prediction are finite and the weight is
  finite and > 0 (an apples-to-apples comparison, same rows for every expert; mirrors the drop-any-nonfinite
  semantics of :mod:`_value_report`). ``RMSE_g(e) = sqrt( sum_{M_g} w (e - y)^2 / sum_{M_g} w )``.
* When a lag expert exists and ``M_g`` is non-empty (>= ``min_group_rows`` rows): deploy the non-lag expert
  with the smallest ``RMSE_g`` ONLY IF it beats ``RMSE_g(lag)`` by the relative ``shrink_rtol`` margin;
  otherwise deploy ``lag``. Ties among non-lag experts break toward ``prefer`` order.
* When ``M_g`` is empty (lag NaN across the whole group) the tier-2 set ``N_g`` (drop lag, keep the rest)
  decides among the non-lag experts.
* Groups with fewer than ``min_group_rows`` matched rows, and groups unseen at predict, DEFER to the GLOBAL
  fallback, which is the lag failsafe itself whenever a lag expert is present (else the pooled-best expert).

Not-worse-than-lag guarantee (proof)
-------------------------------------
For every group with lag available, ``SSE_g(deployed) <= SSE_g(lag)``: the gate only leaves lag for a
non-lag expert with a STRICTLY smaller SSE (``shrink_rtol >= 0``), and deferring groups deploy lag itself
(equality). Summing over groups on the matched selection rows, ``pooled SSE(gate) <= pooled SSE(lag)``,
hence ``pooled RMSE(gate) <= pooled RMSE(lag)``. With ``shrink_rtol == 0`` each per-group deploy is the exact
argmin over ALL experts, so additionally ``pooled SSE(gate) <= pooled SSE(f)`` for EVERY single expert ``f``
(not only lag). Raising ``shrink_rtol`` keeps the vs-lag inequality exact while trading a little in-sample
optimality vs the OTHER experts for held-out robustness (fewer noise-driven flips away from the failsafe).
These inequalities are recomputed numerically after fit and pinned in ``guarantee_``.

This is a SELECTION gate (deploys one expert's prediction per row), not a blend -- selection cannot invent a
prediction outside the experts' convex hull and cannot be worse than the failsafe where the failsafe wins,
which is exactly the production requirement (prod: 13.30 ensemble vs 11.58 lag floor).

Performance (cProfile, ``_benchmarks/bench_moe_gate.py``, 1M rows / 500 groups, 3 experts). Fit is a fused
single-pass ``numba.njit`` per-group weighted-SSE reduction (:func:`_grouped_sse_njit`, default) with an
``np.bincount`` fallback (:func:`_grouped_sse_bincount`, bit-close: sequential vs bincount summation ~1e-12,
never a choice flip); ``pd.factorize`` of the group ids is the next O(n) cost, everything after is
O(n_groups * K). Route was originally 4.45 s -- ``_codes_for`` converted every predict label with a per-row
Python ``_to_native`` before mapping (5M calls, ~20 s cumtime in the 5-iter loop); keying ``_label_to_code``
by the RAW factorized labels lets predict map in ONE vectorized ``pd.Series.map`` (numpy scalars hash-equal to
their Python counterparts), cutting route to 64 ms (~70x) and the full 5-iter fit+route loop from 27.8 s to
0.55 s. Remaining route cost is the ``column_stack`` of the experts + the vectorized gather -- no actionable
further speedup at this shape.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

try:
    import pandas as pd

    _HAVE_PANDAS = True
except Exception:  # pragma: no cover - pandas is a hard dep in practice
    _HAVE_PANDAS = False

try:
    import numba

    @numba.njit(cache=True)
    def _grouped_sse_njit(codes, w, y, P, n_groups, n_experts, lag_idx):
        """Fused single-pass grouped reduction over K experts, computing BOTH matched sets in one sweep.

        ``all``  = rows where ``y``, the weight (> 0) and EVERY expert are finite -> ``sse_all``, ``W_all``.
        ``nolag`` = rows where ``y``, the weight and every NON-lag expert are finite (lag may be NaN) ->
        ``sse_nl``, ``W_nl`` (only the non-lag columns are meaningful). ``lag_idx < 0`` means no lag expert,
        so the two sets coincide. Avoids materializing per-row finite masks / squared-error temporaries.
        """
        rows_all = np.zeros(n_groups, dtype=np.int64)
        W_all = np.zeros(n_groups)
        sse_all = np.zeros((n_groups, n_experts))
        rows_nl = np.zeros(n_groups, dtype=np.int64)
        W_nl = np.zeros(n_groups)
        sse_nl = np.zeros((n_groups, n_experts))
        for i in range(codes.shape[0]):
            g = codes[i]
            if g < 0:
                continue
            wi = w[i]
            if not (wi > 0.0) or not np.isfinite(wi):
                continue
            yi = y[i]
            if not np.isfinite(yi):
                continue
            all_ok = True
            nolag_ok = True
            for k in range(n_experts):
                if not np.isfinite(P[i, k]):
                    all_ok = False
                    if k != lag_idx:
                        nolag_ok = False
                        break
            if nolag_ok:
                rows_nl[g] += 1
                W_nl[g] += wi
                for k in range(n_experts):
                    if k != lag_idx:
                        e = P[i, k] - yi
                        sse_nl[g, k] += wi * e * e
            if all_ok:
                rows_all[g] += 1
                W_all[g] += wi
                for k in range(n_experts):
                    e = P[i, k] - yi
                    sse_all[g, k] += wi * e * e
        return rows_all, W_all, sse_all, rows_nl, W_nl, sse_nl

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is an optional accelerator here
    _HAVE_NUMBA = False


__all__ = ["MoESelectionGate"]


def _as1d(a: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(-1))


def _to_native(v: Any) -> Any:
    """JSON / dict-key-native group label (numpy scalars -> python; bytes -> ascii; else str)."""
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, np.generic):
        n = v.item()
        return n if isinstance(n, (str, bool, int, float)) else str(n)
    if isinstance(v, bytes):
        return v.decode("ascii", "replace")
    return str(v)


def _factorize(group_ids: Any) -> tuple[np.ndarray, list]:
    """(codes, unique_labels). NaN / null labels map to code -1 (excluded from selection + routed global)."""
    if _HAVE_PANDAS:
        codes, uniq = pd.factorize(np.asarray(group_ids), sort=False)
        return np.asarray(codes, dtype=np.int64), list(uniq)
    arr = np.asarray(group_ids)
    uniq, codes = np.unique(arr, return_inverse=True)
    return np.asarray(codes, dtype=np.int64), list(uniq)


def _grouped_sse_bincount(codes, w, y, P, n_groups, n_experts, lag_idx):
    """Vectorized fallback: build the two matched masks, then O(n) bincount reductions per expert."""
    finite_w = np.isfinite(w) & (w > 0)
    finite_y = np.isfinite(y)
    finite_P = np.isfinite(P)  # (n, K)
    base = finite_w & finite_y & (codes >= 0)
    if lag_idx >= 0:
        nolag_cols = [k for k in range(n_experts) if k != lag_idx]
        nolag_ok = base & finite_P[:, nolag_cols].all(axis=1) if nolag_cols else base.copy()
    else:
        nolag_ok = base & finite_P.all(axis=1)
    all_ok = base & finite_P.all(axis=1)

    def _reduce(mask):
        cv = codes[mask]
        wv = w[mask]
        yv = y[mask]
        rows = np.bincount(cv, minlength=n_groups).astype(np.int64)
        W = np.bincount(cv, weights=wv, minlength=n_groups)
        sse = np.zeros((n_groups, n_experts))
        for k in range(n_experts):
            ek = P[mask, k] - yv
            sse[:, k] = np.bincount(cv, weights=wv * ek * ek, minlength=n_groups)
        return rows, W, sse

    rows_all, W_all, sse_all = _reduce(all_ok)
    rows_nl, W_nl, sse_nl = _reduce(nolag_ok)
    return rows_all, W_all, sse_all, rows_nl, W_nl, sse_nl


def _grouped_sse(codes, w, y, P, n_groups, n_experts, lag_idx):
    if _HAVE_NUMBA:
        return _grouped_sse_njit(np.ascontiguousarray(codes, dtype=np.int64), w, y, np.ascontiguousarray(P), n_groups, n_experts, lag_idx)
    return _grouped_sse_bincount(codes, w, y, P, n_groups, n_experts, lag_idx)


class MoESelectionGate:
    """Learn a per-group (fallback: global) choice among expert predictions with a not-worse-than-lag guarantee.

    Parameters
    ----------
    failsafe
        Name of the lag / failsafe expert key in ``preds``. Its predictions may be NaN where no lag is
        available (e.g. the first row of a group); the guarantee only binds where lag is finite. If the key is
        absent the gate degrades to a plain per-group argmin selector (the vs-lag clause is vacuous).
    shrink_rtol
        Relative RMSE margin a non-lag expert must beat lag by, per group, to be deployed over the failsafe.
        ``0.0`` gives the exact in-sample argmin (pooled gate RMSE <= EVERY single expert). Larger values keep
        the vs-lag guarantee exact while resisting noise-driven flips away from the failsafe on held-out data.
    tie_rtol
        Relative band within which two non-lag experts count as tied; ties break toward ``prefer`` order.
    min_group_rows
        Groups with fewer matched selection rows defer to the global fallback (the lag failsafe when present).
    prefer
        Preference order (earlier = preferred) for breaking ties; defaults to failsafe-first, then the order
        the experts appear in ``preds``.

    Attributes (set by :meth:`fit`)
    -------------------------------
    expert_names_ : list[str]
    group_choice_ : dict[label, str]      chosen expert per selection group
    global_choice_ : str                  fallback expert for unseen / low-data groups
    guarantee_ : dict                     pooled RMSEs + the verified not-worse-than-lag / best-single flags
    """

    def __init__(
        self,
        *,
        failsafe: str = "lag",
        shrink_rtol: float = 0.0,
        tie_rtol: float = 1e-9,
        min_group_rows: int = 1,
        prefer: Optional[Sequence[str]] = None,
    ) -> None:
        if shrink_rtol < 0:
            raise ValueError(f"shrink_rtol must be >= 0; got {shrink_rtol}")
        if tie_rtol < 0:
            raise ValueError(f"tie_rtol must be >= 0; got {tie_rtol}")
        if min_group_rows < 1:
            raise ValueError(f"min_group_rows must be >= 1; got {min_group_rows}")
        self.failsafe = failsafe
        self.shrink_rtol = float(shrink_rtol)
        self.tie_rtol = float(tie_rtol)
        self.min_group_rows = int(min_group_rows)
        self.prefer = None if prefer is None else list(prefer)

    # -- fit -----------------------------------------------------------------

    def _stack(self, preds: Mapping[str, Any], n_expected: Optional[int] = None) -> np.ndarray:
        """Column-stack the experts in the fitted order into a C-contiguous (n, K) float64 matrix."""
        cols = [_as1d(preds[name]) for name in self.expert_names_]
        n = cols[0].shape[0]
        for name, c in zip(self.expert_names_, cols):
            if c.shape[0] != n:
                raise ValueError(f"expert '{name}' length {c.shape[0]} != {n}")
        if n_expected is not None and n != n_expected:
            raise ValueError(f"prediction length {n} != group_ids/y length {n_expected}")
        return np.ascontiguousarray(np.column_stack(cols)) if len(cols) > 1 else cols[0].reshape(-1, 1)

    def _priority_order(self) -> list[int]:
        """Column indices in tie-break preference order (earlier = preferred)."""
        if self.prefer is not None:
            order = [self.expert_names_.index(n) for n in self.prefer if n in self.expert_names_]
        else:
            order = []
            if self._lag_idx >= 0:
                order.append(self._lag_idx)
        for k in range(len(self.expert_names_)):
            if k not in order:
                order.append(k)
        return order

    def fit(
        self,
        y_true: Any,
        preds: Mapping[str, Any],
        *,
        group_ids: Any = None,
        sample_weight: Any = None,
    ) -> "MoESelectionGate":
        """Learn the per-group choice on the SELECTION split (the rows passed here).

        ``preds`` maps expert name -> 1-D predictions aligned with ``y_true``. The guarantee is proven on these
        rows; hold out a DISJOINT split for honest scoring / routing (this method must not see the score rows).
        """
        if not preds:
            raise ValueError("preds must contain at least one expert")
        self.expert_names_ = list(preds.keys())
        self._lag_idx = self.expert_names_.index(self.failsafe) if self.failsafe in self.expert_names_ else -1

        y = _as1d(y_true)
        n = y.shape[0]
        P = self._stack(preds, n_expected=n)
        K = P.shape[1]

        self._groupless = group_ids is None
        if self._groupless:
            codes = np.zeros(n, dtype=np.int64)
            uniq: list = [None]
        else:
            codes, uniq = _factorize(group_ids)
            if codes.shape[0] != n:
                raise ValueError(f"group_ids length {codes.shape[0]} != y length {n}")
        n_groups = len(uniq)

        w = np.ones(n, dtype=np.float64) if sample_weight is None else _as1d(sample_weight)
        if w.shape[0] != n:
            raise ValueError(f"sample_weight length {w.shape[0]} != y length {n}")

        self._n_groups = n_groups
        # Keyed by the RAW factorized labels so predict-time mapping is a single vectorized ``pd.Series.map``
        # (no per-row Python conversion); ``group_choice_`` below re-keys to JSON-native labels for reporting.
        self._label_to_code = {uniq[g]: g for g in range(n_groups)}
        self._priority_idx = self._priority_order()

        if n == 0 or n_groups == 0:
            self.group_choice_idx_ = np.zeros(n_groups, dtype=np.int64)
            self._global_idx = self._lag_idx if self._lag_idx >= 0 else self._priority_idx[0]
            self.group_choice_ = {}
            self.global_choice_ = self.expert_names_[self._global_idx]
            self.guarantee_ = self._empty_guarantee()
            return self

        rows_all, W_all, sse_all, rows_nl, W_nl, sse_nl = _grouped_sse(codes, w, y, P, n_groups, K, self._lag_idx)

        self._global_idx = self._pick_global(sse_all, W_all, sse_nl, W_nl)
        self.group_choice_idx_ = self._pick_per_group(rows_all, W_all, sse_all, rows_nl, W_nl, sse_nl)

        # Groupless fit has a single global model: the fallback IS that one group's choice (no unseen groups
        # can exist). Grouped fits keep the lag-failsafe fallback for genuinely unseen groups.
        if self._groupless:
            self._global_idx = int(self.group_choice_idx_[0])

        self.group_choice_ = {_to_native(uniq[g]): self.expert_names_[int(self.group_choice_idx_[g])] for g in range(n_groups)}
        self.global_choice_ = self.expert_names_[self._global_idx]
        self.guarantee_ = self._verify_guarantee(sse_all, W_all)
        return self

    def _pick_global(self, sse_all, W_all, sse_nl, W_nl) -> int:
        """Fallback expert for unseen / low-data groups: the lag failsafe when present, else the pooled best."""
        if self._lag_idx >= 0 and float(W_all.sum()) > 0:
            return self._lag_idx
        # No lag anywhere: pooled-best non-lag expert on the nolag matched set.
        pooled_W = float(W_nl.sum()) if float(W_nl.sum()) > 0 else float(W_all.sum())
        sse_src = sse_nl if float(W_nl.sum()) > 0 else sse_all
        if pooled_W <= 0:
            return self._priority_idx[0]
        pooled_sse = sse_src.sum(axis=0)
        rmse = np.sqrt(pooled_sse / pooled_W)
        return self._argmin_pref(rmse, exclude_lag=(self._lag_idx >= 0))

    def _argmin_pref(self, rmse: np.ndarray, *, exclude_lag: bool) -> int:
        """Argmin of ``rmse`` breaking near-ties (within ``tie_rtol``) toward ``_priority_idx`` order."""
        cand = [k for k in range(rmse.shape[0]) if np.isfinite(rmse[k]) and not (exclude_lag and k == self._lag_idx)]
        if not cand:
            return self._priority_idx[0]
        best = min(rmse[k] for k in cand)
        band = best * (1.0 + self.tie_rtol) if best > 0 else self.tie_rtol
        tied = [k for k in cand if rmse[k] <= band or rmse[k] <= best]
        for k in self._priority_idx:
            if k in tied:
                return k
        return min(cand, key=lambda k: rmse[k])

    def _pick_per_group(self, rows_all, W_all, sse_all, rows_nl, W_nl, sse_nl) -> np.ndarray:
        choice = np.empty(self._n_groups, dtype=np.int64)
        m = self.min_group_rows
        for g in range(self._n_groups):
            if rows_all[g] >= m and W_all[g] > 0:
                choice[g] = self._choose_tier1(sse_all[g], W_all[g])
            elif rows_nl[g] >= m and W_nl[g] > 0:
                choice[g] = self._argmin_pref(np.sqrt(sse_nl[g] / W_nl[g]), exclude_lag=(self._lag_idx >= 0))
            else:
                choice[g] = self._global_idx
        return choice

    def _choose_tier1(self, sse_g: np.ndarray, W_g: float) -> int:
        """Per-group choice on the fully-matched set: deploy lag unless a non-lag expert beats it by shrink_rtol."""
        rmse = np.sqrt(sse_g / W_g)
        if self._lag_idx < 0:
            return self._argmin_pref(rmse, exclude_lag=False)
        lag_rmse = rmse[self._lag_idx]
        best_nl = self._argmin_pref(rmse, exclude_lag=True)
        if np.isfinite(rmse[best_nl]) and rmse[best_nl] < lag_rmse * (1.0 - self.shrink_rtol):
            return best_nl
        return self._lag_idx

    def _verify_guarantee(self, sse_all: np.ndarray, W_all: np.ndarray) -> dict:
        """Recompute pooled RMSEs on the fully-matched selection rows and pin the not-worse-than-lag flags."""
        pooled_W = float(W_all.sum())
        pooled_single = sse_all.sum(axis=0)
        gate_sse = float(sum(sse_all[g, int(self.group_choice_idx_[g])] for g in range(self._n_groups)))
        eps = 1e-9
        out: dict[str, Any] = {
            "pooled_rmse_per_expert": {},
            "pooled_rmse_gate": None,
            "not_worse_than_lag": True,
            "not_worse_than_best_single": True,
            "matched_weight": pooled_W,
            "shrink_rtol": self.shrink_rtol,
        }
        if pooled_W <= 0:
            return out
        gate_rmse = float(np.sqrt(gate_sse / pooled_W))
        out["pooled_rmse_gate"] = gate_rmse
        best_single = np.inf
        for k, name in enumerate(self.expert_names_):
            rk = float(np.sqrt(pooled_single[k] / pooled_W))
            out["pooled_rmse_per_expert"][name] = rk
            best_single = min(best_single, rk)
            if k == self._lag_idx and gate_rmse > rk * (1.0 + eps):
                out["not_worse_than_lag"] = False
        if gate_rmse > best_single * (1.0 + eps):
            out["not_worse_than_best_single"] = False
        return out

    def _empty_guarantee(self) -> dict:
        return {
            "pooled_rmse_per_expert": {}, "pooled_rmse_gate": None,
            "not_worse_than_lag": True, "not_worse_than_best_single": True,
            "matched_weight": 0.0, "shrink_rtol": self.shrink_rtol,
        }

    # -- predict / route -----------------------------------------------------

    def _codes_for(self, group_ids: Any, n: int) -> np.ndarray:
        """Map predict-time group labels to fitted codes; unseen / null labels -> -1 (global fallback)."""
        if self._groupless:
            return np.zeros(n, dtype=np.int64)  # single global model: ignore any labels, route to group 0
        if group_ids is None:
            return np.full(n, -1, dtype=np.int64)
        labels = np.asarray(group_ids)
        if labels.shape[0] != n:
            raise ValueError(f"group_ids length {labels.shape[0]} != prediction length {n}")
        if _HAVE_PANDAS:
            mapped = pd.Series(labels).map(self._label_to_code)
            return np.asarray(mapped.fillna(-1).to_numpy(dtype=np.int64))
        return np.array([self._label_to_code.get(v, -1) for v in labels], dtype=np.int64)

    def route_labels(self, group_ids: Any = None, *, n: Optional[int] = None) -> np.ndarray:
        """Per-row chosen expert NAME (before per-row NaN fallback). Needs ``n`` when ``group_ids is None``."""
        if group_ids is not None:
            n = np.asarray(group_ids).shape[0]
        if n is None:
            raise ValueError("route_labels: pass group_ids or n")
        codes = self._codes_for(group_ids, n)
        idx = self._choice_idx_per_row(codes)
        names = np.asarray(self.expert_names_, dtype=object)
        return np.asarray(names[idx])

    def _choice_idx_per_row(self, codes: np.ndarray) -> np.ndarray:
        seen = (codes >= 0) & (codes < self._n_groups)
        idx = np.full(codes.shape[0], self._global_idx, dtype=np.int64)
        if seen.any():
            idx[seen] = self.group_choice_idx_[codes[seen]]
        return idx

    def predict(self, preds: Mapping[str, Any], *, group_ids: Any = None) -> np.ndarray:
        """Route: deploy each row's group choice; if that expert is NaN for the row, fall back along ``prefer``.

        Returns the per-row deployed prediction (never worse, in selection-split expectation, than the lag
        failsafe on rows where lag is available -- see the module guarantee).
        """
        P = self._stack(preds)
        n = P.shape[0]
        codes = self._codes_for(group_ids, n)
        idx = self._choice_idx_per_row(codes)
        out = P[np.arange(n), idx]
        bad = ~np.isfinite(out)
        if bad.any():
            for k in self._priority_idx:
                if not bad.any():
                    break
                take = bad & np.isfinite(P[:, k])
                out[take] = P[take, k]
                bad &= ~take
        return np.asarray(out)
