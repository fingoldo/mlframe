"""Layer 100 (2026-06-01): UNIFY scorer-selection under the Param-Oracle.

Two scorer-selection mechanisms shipped before this layer:

* **Layer 68** (``_orthogonal_scorer_auto_fe``): per-column bootstrap-LCB
  bake-off that runs ALL scorers ({plug_in, KSG, copula, dCor, HSIC}) and
  picks the winner. Correct but EXPENSIVE -- O(n_scorers * n_boot) MI
  compute on every column, paid on every fit.
* **Layer 76** (``_orthogonal_meta_scorer_fe``): cheap fingerprint -> rule
  cascade (``predict_best_scorer``) that runs ONE scorer. Fast but STATIC
  -- the rules are distilled from the L75 matrix and never improve from
  observed downstream quality.

Layer 98 built the Param-Oracle (``mlframe.utils._param_oracle``): a
fingerprint -> recommend cache that LEARNS from recorded history. Layer 99
used it for FE-flag recommendation. This layer applies it to the scorer
choice, UNIFYING the two prior mechanisms into one adaptive path:

* **cold-start = L76**: with no learned history for a fingerprint bucket,
  fall back to the L76 ``predict_best_scorer`` rule cascade (reused, not
  reimplemented). The static rules are the right prior before any data.
* **benchmark = L68-as-populator**: the expensive L68-style bake-off runs
  ONCE via :meth:`OracleScorerSelector.benchmark_all_scorers`, recording
  every scorer's measured quality into the oracle. The cost amortises --
  run the bake-off once, recommend forever.
* **learned**: once the oracle has confident history (>= ``min_observations``
  observations) for a fingerprint bucket, the learned-best scorer wins
  over the cold-start rule. The cascade is the prior; observed quality is
  the posterior.

So Layer 100 = L76 cold-start prior + L68 bake-off as the populator +
Param-Oracle learning over time, all keyed on the SAME fingerprint that
backs the oracle (``mlframe.utils._param_oracle.default_fingerprint``).

The store is **stat-only** (Param-Oracle's hard constraint): only scalar
fingerprint stats, the scorer name, and the recorded quality ever touch
disk -- never the raw arrays.

Wiring: opt-in via ``MRMR(fe_hybrid_orth_default_scorer="auto_oracle")``.
The existing ``"auto"`` (L68) and ``"meta"`` (L76) values keep working
unchanged; ``"auto_oracle"`` is the new unified path.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.utils import (
    ParamOracle,
    bucketize_fingerprint,
    default_fingerprint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ORACLE_SCORER_NAMES",
    "ORACLE_FN_NAME",
    "OracleScorerSelector",
]


# The full scorer pool the oracle ranges over. Superset of the L68 pool
# (plug_in/ksg/copula/dcor/hsic) plus the redundancy-aware MI scorers the
# L76 cascade can dispatch to (jmim/tc/cmim). Keeping the union here means
# a benchmark run can record every scorer the unified path might ever
# recommend, so the learned posterior is never narrower than either prior.
ORACLE_SCORER_NAMES = (
    "plug_in", "ksg", "copula", "dcor", "hsic", "jmim", "tc", "cmim",
)

# Stable ``fn_name`` under which all scorer-selection observations are
# keyed in the oracle store. A single key means cold-start rules, the
# bake-off, and the learned posterior all share one history namespace.
ORACLE_FN_NAME = "orth_scorer_select"


def _quality_objective(output: Any, elapsed_s: float, rss_delta_mb):
    """Objective_fn for the oracle: the scorer's selection quality.

    ``output`` is expected to be a ``(scorer_name, quality_float)`` tuple
    emitted by the bake-off closure. The oracle maximises ``quality`` and
    tie-breaks on ``elapsed_s`` (cheaper wins) -- so two scorers of equal
    quality resolve to the faster one, exactly the L76 amortisation intent.
    """
    try:
        _scorer, q = output
    except Exception:
        q = float("nan")
    return {"quality": float(q), "elapsed_s": float(elapsed_s)}


class OracleScorerSelector:
    """Unified, learning scorer-selector wrapping a :class:`ParamOracle`.

    Keyed on the dataset fingerprint (the SAME
    :func:`mlframe.utils._param_oracle.default_fingerprint` the oracle uses
    everywhere -- not a duplicate fingerprinter). The param space is the
    single ``"scorer"`` axis over :data:`ORACLE_SCORER_NAMES`; the objective
    is the scorer's selection quality (maximised, cheaper-tie-broken).

    Parameters
    ----------
    store_path:
        Param-Oracle parquet store path. A bare filename resolves under
        :func:`mlframe.utils._param_oracle.default_store_dir`.
    min_observations:
        Confidence gate -- the learned-best scorer is only trusted once its
        fingerprint bucket has at least this many observations; otherwise we
        fall back to the L76 cold-start cascade.
    scorer_names:
        Override the scorer pool (defaults to :data:`ORACLE_SCORER_NAMES`).
    """

    def __init__(
        self,
        store_path: str = "orth_scorer_select.parquet",
        *,
        min_observations: int = 3,
        scorer_names: Sequence[str] = ORACLE_SCORER_NAMES,
    ):
        self.store_path = store_path
        self.min_observations = int(min_observations)
        self.scorer_names = tuple(scorer_names)
        self.oracle = ParamOracle(
            store_path,
            objective_fn=_quality_objective,
            param_space={"scorer": list(self.scorer_names)},
            mode="inference",
            maximize="quality",
            min_observations=int(min_observations),
        )

    # ----- fingerprint (REUSE the oracle's default_fingerprint) -----

    def fingerprint(self, X, y=None) -> dict:
        """Stat-only fingerprint of ``X`` via the Param-Oracle's
        :func:`default_fingerprint`. ``y`` is accepted for API symmetry but
        the redundancy/skew/cardinality signal the scorer choice keys on
        lives in ``X``; passing ``y`` as a second positional keeps the
        fingerprint aligned with the L68/L76 ``(X, y)`` call shape without
        leaking the target into the bucket key."""
        args = (X,) if y is None else (X, y)
        return default_fingerprint(args, {})

    # ----- recommend (cold-start = L76, learned = oracle) -----

    def recommend_scorer(self, X, y=None) -> str:
        """Recommend a scorer for ``(X, y)``.

        Resolution:
          1. LEARNED -- if the oracle has a confident (>= ``min_observations``)
             best scorer for this fingerprint bucket, return it.
          2. COLD-START -- otherwise fall back to L76's
             :func:`predict_best_scorer` on the SAME ``(X, y)`` (reused, not
             reimplemented). On any failure (e.g. ``y`` is None / non-pandas)
             degrade to the first scorer in the pool.
        """
        fp = self.fingerprint(X, y)
        learned = self._learned_scorer(fp)
        if learned is not None:
            return learned
        return self._cold_start_scorer(X, y)

    def _learned_scorer(self, fp: Mapping[str, Any]) -> Optional[str]:
        """Return the oracle's confident best scorer for this fingerprint
        bucket, or ``None`` if the bucket has no confident history yet.

        We do NOT use ``oracle.recommend`` here because its cold-store /
        k-NN / global-best fallbacks would mask the "no confident history"
        signal -- we WANT that signal so the caller can route to the L76
        cold-start cascade instead of an over-eager global best. So we read
        the exact-bucket rows directly and apply only the confidence gate.
        """
        rows = [r for r in self.oracle.store.read_rows() if r.get("fn_name") == ORACLE_FN_NAME and r.get("host") == self.oracle.host]
        if not rows:
            return None
        from mlframe.utils import stable_json
        target_key = stable_json(bucketize_fingerprint(fp))
        exact = [r for r in rows if r.get("fp_bucket_json") == target_key]
        best = self.oracle._best_row(exact, require_confident=True)
        if best is None:
            return None
        from mlframe.utils import loads_json
        combo = loads_json(best.get("param_combo_json"))
        scorer = combo.get("scorer")
        if scorer in self.scorer_names:
            return str(scorer)
        return None

    def _cold_start_scorer(self, X, y) -> str:
        """L76 fingerprint -> rule cascade. Reused verbatim from
        ``_orthogonal_meta_scorer_fe`` (no reimplementation)."""
        try:
            from ._orthogonal_meta_scorer_fe import (
                fingerprint_signal,
                predict_best_scorer,
            )
            if y is None:
                raise ValueError("cold-start cascade needs y")
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
            y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            fp_signal = fingerprint_signal(X_df, y_arr)
            scorer = predict_best_scorer(fp_signal)
            if scorer in self.scorer_names:
                return scorer
            # The cascade can return scorers outside our pool only if the
            # pool was narrowed via ``scorer_names``; clamp to the pool.
            return self.scorer_names[0]
        except Exception as exc:
            logger.warning(
                "OracleScorerSelector cold-start cascade failed (%s: %s); "
                "defaulting to %r.",
                type(exc).__name__, exc, self.scorer_names[0],
            )
            return self.scorer_names[0]

    # ----- observe (record a single observation) -----

    def observe_scorer(self, X, scorer: str, quality: float, y=None, ts: Optional[str] = None) -> None:
        """Record one ``(fingerprint, scorer) -> quality`` observation so
        future :meth:`recommend_scorer` calls learn from it.

        Stat-only: only the bucketed fingerprint scalars, the scorer name,
        and the scalar quality are persisted.
        """
        fp = self.fingerprint(X, y)
        self.oracle.record(
            fp, {"scorer": str(scorer)}, {"quality": float(quality)},
            ts=ts, fn_name=ORACLE_FN_NAME,
        )

    # ----- benchmark (L68-style bake-off, run ONCE, amortise forever) -----

    def benchmark_all_scorers(
        self,
        X: pd.DataFrame,
        y,
        *,
        cols: Optional[Sequence[str]] = None,
        degrees: Sequence[int] = (2, 3),
        basis: str = "auto",
        n_boot: int = 5,
        random_state: int = 0,
        ts: Optional[str] = None,
    ) -> dict:
        """Run the L68-style bake-off ONCE and record EVERY scorer's quality
        into the oracle for this dataset's fingerprint.

        This is the "benchmark mode" that POPULATES the oracle so the
        expensive all-scorers compute amortises: one call here lets every
        future :meth:`recommend_scorer` on a similar fingerprint return the
        learned-best scorer without re-running the bake-off.

        Quality is the per-scorer MEAN normalised LCB across engineered
        columns (the same ``lcb_norm_per_scorer`` headroom metric Layer 68
        computes), which is dimensionless and cross-scorer comparable. The
        scorer that gets closest to its own ceiling on the engineered
        columns earns the highest recorded quality -- exactly the L68
        selection criterion, now persisted instead of discarded.

        Returns
        -------
        dict ``{scorer_name: quality_float}`` for inspection. Side effect:
        one oracle observation recorded per scorer.
        """
        from ._orthogonal_scorer_auto_fe import (
            SCORER_NAMES as _L68_SCORER_NAMES,
            select_best_scorer_per_column,
        )
        from ._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        engineered = generate_univariate_basis_features(
            X_df, cols=cols, degrees=degrees, basis=basis,
        )
        raw_X = X_df[[c for c in (cols or X_df.columns) if c in X_df.columns and pd.api.types.is_numeric_dtype(X_df[c])]]

        qualities: dict[str, float] = {}
        if not engineered.empty:
            table = select_best_scorer_per_column(
                raw_X, engineered, y_arr,
                n_boot=int(n_boot), random_state=int(random_state),
            )
            # Per-scorer mean normalised LCB across engineered columns: the
            # cross-scorer-comparable headroom metric. ``lcb_norm_per_scorer``
            # is a dict per row; average each scorer over the rows.
            if not table.empty and "lcb_norm_per_scorer" in table.columns:
                for s in _L68_SCORER_NAMES:
                    vals = [float(d.get(s, 0.0)) for d in table["lcb_norm_per_scorer"] if isinstance(d, dict)]
                    if vals:
                        qualities[s] = float(np.mean(vals))

        # Scorers in our pool that the L68 bake-off does not cover
        # (jmim/tc/cmim) get the L76 cold-start signal as a fallback quality
        # so the recorded posterior still ranks them: the cold-start pick
        # earns a small positive quality, the rest a baseline. This keeps a
        # single benchmark call from leaving the redundancy-aware scorers
        # permanently unobservable, while never inventing fake LCB numbers
        # for them (they are recorded at the cold-start prior, not measured).
        cold = self._cold_start_scorer(X_df, y)
        for s in self.scorer_names:
            if s in qualities:
                continue
            qualities[s] = 0.5 if s == cold else 0.0

        for s in self.scorer_names:
            q = qualities.get(s, 0.0)
            self.observe_scorer(X_df, s, q, y=y, ts=ts)
        return qualities

    # ----- pickle / clone friendliness -----

    def __getstate__(self) -> dict:
        # The ParamOracle holds only a store-path + scalar config; it is
        # itself picklable, but we rebuild it on unpickle so a pickled
        # selector reconnects to the live on-disk store rather than caching
        # a stale handle.
        return {
            "store_path": self.store_path,
            "min_observations": self.min_observations,
            "scorer_names": self.scorer_names,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(
            store_path=state["store_path"],
            min_observations=state["min_observations"],
            scorer_names=state["scorer_names"],
        )
