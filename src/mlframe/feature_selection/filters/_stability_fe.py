"""Layer 36 stability-aware FE wrapper around MRMR.

Why
----
MRMR's hybrid FE pipeline (orth-poly + MI-greedy + kfold_te + count/freq /
cat-num residual) uses two-gate selection on a single fit. The MI-greedy
gate compares a candidate's MI vs the seed-feature MI (relative uplift)
and against an absolute floor; borderline candidates flip in / out of the
support based on finite-sample noise. For high-stakes production
analyses (an analyst building a permanent model for a regulated /
externally-audited use case) it is more valuable to know which
engineered columns CONSISTENTLY appear across bootstrap subsamples than
which set survives a single fit.

This module implements Meinshausen-Buhlmann stability selection
applied to FE: refit MRMR n_bootstraps times on size-fraction
subsamples, count how often each engineered NAME shows up in any of the
six FE attribute lists, and surface the columns whose selection
frequency clears a user-defined threshold.

Public surface
--------------
``stability_select_fe`` -- one-shot helper for an analyst.
``StabilityFESelector`` -- sklearn-compatible estimator (fit / transform
/ get_params / set_params; survives pickle + sklearn.base.clone).

Both share the canonical bookkeeping list of FE attributes that MRMR
populates per Layer 35 (``hybrid_orth_features_``, ``mi_greedy_features_``,
``kfold_te_features_``, ``count_encoding_features_``,
``frequency_encoding_features_``, ``cat_num_interaction_features_``).

A union over those six lists captures every engineered name a single
MRMR fit emits, regardless of which constructor stage produced it.
``source_mechanism`` records, per engineered name, the FE constructor it
first appeared in over the bootstrap sweep (ties broken by the
left-to-right order of the six attribute lists).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Canonical bookkeeping list. Order matters for ``source_mechanism`` tie
# breaking and for the column order in the returned DataFrame.
FE_ATTR_NAMES: tuple[str, ...] = (
    "hybrid_orth_features_",
    "mi_greedy_features_",
    "kfold_te_features_",
    "count_encoding_features_",
    "frequency_encoding_features_",
    "cat_num_interaction_features_",
)


def _engineered_union(fitted_mrmr) -> dict[str, str]:
    """Return {engineered_name: source_mechanism} for one fitted MRMR.

    Tie-breaking: a name that surfaces in multiple constructor lists is
    attributed to the LEFTMOST attribute in ``FE_ATTR_NAMES`` it
    appeared in. This is a stable, deterministic rule independent of
    bootstrap ordering.
    """
    out: dict[str, str] = {}
    for attr in FE_ATTR_NAMES:
        cols = getattr(fitted_mrmr, attr, None) or []
        for c in cols:
            if c not in out:
                out[c] = attr
    return out


def _resolve_mrmr_cls():
    """Lazy import so ``import mlframe.feature_selection.filters._stability_fe``
    does not eagerly pay the MRMR class-build cost (numba JIT subgraph
    + GPU NVRTC kernel preflight). Mirrors the lazy-MRMR pattern used in
    ``_cluster_aggregate`` -- see MEMORY note 2026-05-21 on
    monolith-split via re-export."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR


def _bootstrap_indices(n: int, sample_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a size-fraction subsample WITHOUT replacement.

    Meinshausen-Buhlmann stability selection uses subsampling-without-
    replacement (size = floor(n * sample_fraction)) rather than
    bootstrap-with-replacement; the no-replacement variant has tighter
    theoretical guarantees on the family-wise error rate of the stable
    set. ``sample_fraction=0.75`` follows the paper's default.
    """
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1]; got {sample_fraction!r}")
    if sample_fraction == 1.0:
        return np.arange(n, dtype=np.int64)
    size = max(int(np.floor(n * sample_fraction)), 2)
    return rng.choice(n, size=size, replace=False)


def _coerce_pandas(X, y) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce X / y to (DataFrame, Series). Avoids surprises from
    numpy-array inputs whose .iloc-style row selection would otherwise
    require dispatch per bootstrap.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y must be 1-D (Series / ndarray / single-col frame); " f"got DataFrame with shape {y.shape}")
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        y = pd.Series(np.asarray(y), name="y")
    return X, y


def _run_bootstraps(
    X: pd.DataFrame,
    y: pd.Series,
    base_mrmr_params: Mapping[str, Any],
    n_bootstraps: int,
    sample_fraction: float,
    rng: np.random.Generator,
) -> list[dict[str, str]]:
    """Fit MRMR ``n_bootstraps`` times on size-fraction subsamples; return
    one ``{engineered_name: source_mechanism}`` dict per bootstrap.

    Per-bootstrap MRMR uses a per-bootstrap ``random_seed`` derived from
    the wrapper RNG so noise across bootstraps is independent (callers
    needing exact reproducibility set ``random_state`` at the wrapper
    level).
    """
    MRMR = _resolve_mrmr_cls()
    n = len(X)
    per_boot: list[dict[str, str]] = []
    n_failed = 0
    for _b in range(n_bootstraps):
        idx = _bootstrap_indices(n, sample_fraction, rng)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = y.iloc[idx].reset_index(drop=True)
        kwargs = dict(base_mrmr_params)
        # Independent per-bootstrap seed so MRMR's own RNG branches differ
        # across bootstraps (otherwise n_bootstraps copies of the SAME
        # seed would collapse stability variance to zero).
        kwargs["random_seed"] = int(rng.integers(0, 2**31 - 1))
        m = MRMR(**kwargs)
        try:
            m.fit(Xb, yb)
        except Exception as exc:
            # mrmr_audit_2026-07-20 B-15: one degenerate bootstrap subsample used to crash the whole
            # sweep -- mirrors the fix applied to the sibling StabilityMRMR (stability.py) and
            # _stability_cluster.py. Excluded bootstraps are simply absent from ``per_boot``, so
            # ``_aggregate_frequencies``'s ``n_boot = len(per_boot)`` denominator already becomes the
            # effective (successful) count with no further change needed there.
            logger.warning("stability_select_fe: bootstrap %d/%d failed (%s: %s); excluded from frequencies.", _b + 1, n_bootstraps, type(exc).__name__, exc)
            n_failed += 1
            continue
        per_boot.append(_engineered_union(m))
    if not per_boot:
        # CLUSTERING_STABILITY-1 fix (mrmr_audit_2026-07-22): mirrors StabilityMRMR.fit's post-B-14
        # contract -- every bootstrap failing means the input is fundamentally too small/degenerate for
        # MRMR at this sample size, not "some unlucky draws". Raise loudly instead of silently returning
        # an empty per-bootstrap list a caller could easily mistake for "no stable engineered features".
        raise RuntimeError(
            f"stability_select_fe: all {n_bootstraps} bootstraps failed to fit (last error above); "
            f"the input is too small/degenerate for MRMR at sample_fraction={sample_fraction}."
        )
    if n_failed:
        logger.warning("stability_select_fe: %d/%d bootstraps failed; frequencies computed over the %d successful ones.", n_failed, n_bootstraps, len(per_boot))
    return per_boot


def _aggregate_frequencies(
    per_boot: Sequence[Mapping[str, str]],
) -> pd.DataFrame:
    """Aggregate per-bootstrap dicts into a frequency table.

    Returns a DataFrame with columns
    ``[engineered_name, selection_frequency, source_mechanism]`` sorted
    by descending ``selection_frequency`` then by ``engineered_name``
    (lexicographic) for a stable ordering even when multiple columns tie
    at frequency = 1.0.
    """
    if not per_boot:
        return pd.DataFrame(columns=["engineered_name", "selection_frequency", "source_mechanism"])
    n_boot = len(per_boot)
    counts: dict[str, int] = {}
    mech_choice: dict[str, str] = {}
    for boot in per_boot:
        for name, mech in boot.items():
            counts[name] = counts.get(name, 0) + 1
            # Source mechanism: the FIRST one that mentioned the name
            # across bootstrap order. Stable + reproducible.
            mech_choice.setdefault(name, mech)
    rows = [
        {
            "engineered_name": name,
            "selection_frequency": counts[name] / n_boot,
            "source_mechanism": mech_choice[name],
        }
        for name in counts
    ]
    df = pd.DataFrame(rows, columns=["engineered_name", "selection_frequency", "source_mechanism"])
    df = df.sort_values(
        by=["selection_frequency", "engineered_name"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def stability_select_fe(
    X,
    y,
    base_mrmr_params: Mapping[str, Any] | None = None,
    n_bootstraps: int = 10,
    sample_fraction: float = 0.75,
    support_threshold: float = 0.6,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Run MRMR on ``n_bootstraps`` subsamples and aggregate per-name
    selection frequencies.

    Parameters
    ----------
    X, y : 2-D + 1-D matched-length inputs (DataFrame / ndarray / Series).
    base_mrmr_params : kwargs forwarded to ``MRMR(**base_mrmr_params)``.
        Use this to fix any non-default config (FE switches, seeds,
        fe_ntop_features, etc.). The wrapper overrides ``random_seed``
        per bootstrap.
    n_bootstraps : number of subsamples (10 is the Meinshausen-Buhlmann
        default; high-stakes analysts often use 50-100).
    sample_fraction : fraction of rows per subsample (no replacement);
        0.75 is the paper's default.
    support_threshold : fraction-of-bootstraps an engineered name must
        appear in to enter the stable set (0.6 = "in at least 6 of 10
        bootstraps").
    random_state : seed for the wrapper RNG that controls subsample
        indices + per-bootstrap MRMR seeds.

    Returns
    -------
    A dict with keys:
      * ``frequencies`` : DataFrame
          ``[engineered_name, selection_frequency, source_mechanism]``,
          sorted by descending freq then by name.
      * ``stable_set`` : sorted list of engineered names with
          ``selection_frequency >= support_threshold``.
      * ``n_bootstraps`` / ``sample_fraction`` / ``support_threshold`` :
          echo of the call params (for downstream provenance / pickling).
      * ``per_bootstrap_engineered`` : list of dicts (one per bootstrap)
          mapping engineered name -> source mechanism. Cheap-now
          provenance so an analyst can inspect WHICH bootstraps picked a
          borderline name.
    """
    if base_mrmr_params is None:
        base_mrmr_params = {}
    if not (0.0 <= support_threshold <= 1.0):
        raise ValueError(f"support_threshold must be in [0, 1]; got {support_threshold!r}")
    if n_bootstraps < 1:
        raise ValueError(f"n_bootstraps must be >= 1; got {n_bootstraps!r}")

    X, y = _coerce_pandas(X, y)
    rng = np.random.default_rng(random_state)
    per_boot = _run_bootstraps(
        X, y,
        base_mrmr_params=base_mrmr_params,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        rng=rng,
    )
    freq_df = _aggregate_frequencies(per_boot)
    stable = sorted(freq_df.loc[freq_df["selection_frequency"] >= support_threshold, "engineered_name"].tolist())
    return {
        "frequencies": freq_df,
        "stable_set": stable,
        "n_bootstraps": n_bootstraps,
        "sample_fraction": sample_fraction,
        "support_threshold": support_threshold,
        "per_bootstrap_engineered": per_boot,
    }


class StabilityFESelector(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around ``stability_select_fe``.

    ``fit(X, y)`` :
      1. Runs the bootstrap sweep, stores the frequency table at
         ``frequencies_`` and the stable set at ``stable_set_``.
      2. Fits ONE final MRMR on the FULL (X, y), keeps it at
         ``full_mrmr_`` (so transform-time recipe application reuses
         the full-data fit's recipes; subsample recipes would shadow
         signal-bearing columns that only appear in some bootstraps).

    ``transform(X)`` :
      Calls ``full_mrmr_.transform(X)`` and restricts the output frame
      to (raw input columns the full MRMR kept) U (engineered columns
      in ``stable_set_``). Engineered columns whose recipes the full
      MRMR did not learn (e.g. an obscure orth-poly term that only one
      bootstrap surfaced) are silently dropped from the output -- a
      conservative "stable AND reproducible at full-data fit time"
      rule. ``stable_set_`` retains the full list for analyst inspection.

    Pickle / clone : every constructor arg is exposed via
    ``get_params`` (and is therefore preserved through
    ``sklearn.base.clone``). Fitted state (``frequencies_``,
    ``stable_set_``, ``full_mrmr_``, ``per_bootstrap_engineered_``) is
    preserved through ``pickle``.
    """

    def __init__(
        self,
        base_mrmr_params: Mapping[str, Any] | None = None,
        n_bootstraps: int = 10,
        sample_fraction: float = 0.75,
        support_threshold: float = 0.6,
        random_state: int | None = 0,
    ):
        self.base_mrmr_params = base_mrmr_params
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.support_threshold = support_threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Run bootstrap-based stability selection to get the stable feature set, then fit ONE MRMR on the full data (not a subsample) so ``transform`` can replay auditable recipes rather than a subsample-shadowed variant."""
        result = stability_select_fe(
            X, y,
            base_mrmr_params=self.base_mrmr_params or {},
            n_bootstraps=self.n_bootstraps,
            sample_fraction=self.sample_fraction,
            support_threshold=self.support_threshold,
            random_state=self.random_state,
        )
        self.frequencies_ = result["frequencies"]
        self.stable_set_ = result["stable_set"]
        self.per_bootstrap_engineered_ = result["per_bootstrap_engineered"]
        # Recipe re-use: full-data fit so transform produces engineered
        # columns from recipes the analyst can audit; subsample recipes
        # would partial-shadow the signal-bearing columns.
        X_pd, y_pd = _coerce_pandas(X, y)
        MRMR = _resolve_mrmr_cls()
        full = MRMR(**(self.base_mrmr_params or {}))
        full.fit(X_pd, y_pd)
        self.full_mrmr_ = full
        return self

    def transform(self, X):
        """Replay the full-data MRMR's recipes, then restrict engineered columns to those in ``stable_set_`` (stability AND full-data reproducibility both required); raw/non-engineered columns the full fit kept always pass through."""
        if not hasattr(self, "full_mrmr_"):
            raise RuntimeError("StabilityFESelector.transform called before fit; " "no full_mrmr_ recipes are available.")
        out = self.full_mrmr_.transform(X)
        # Stable set may name engineered columns the full MRMR did not
        # surface (e.g. a bootstrap-only borderline candidate). Keep the
        # intersection: stable AND reproducible at full-data fit time.
        stable_present = [c for c in self.stable_set_ if c in out.columns]
        # Plus the raw / non-engineered columns the full MRMR kept (those
        # never appear in any of the FE attribute lists and should pass
        # through regardless of stability).
        full_engineered = set(_engineered_union(self.full_mrmr_).keys())
        raw_kept = [c for c in out.columns if c not in full_engineered]
        keep = raw_kept + [c for c in stable_present if c not in raw_kept]
        return out.loc[:, keep]

    def fit_transform(self, X, y=None, **fit_params):
        """Fit then transform in one call."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Selected feature names (sklearn transformer contract). X_SECURITY_API_PACKAGING-1 fix
        (mrmr_audit_2026-07-22): was missing entirely, unlike ``MRMR``/``GroupAwareMRMR``/``StabilityMRMR``
        in the same module -- a ``Pipeline([("sel", StabilityFESelector(...)), ...]).get_feature_names_out()``
        raised ``AttributeError`` even though ``transform()`` already returns a well-defined column subset.
        Recomputes the SAME stable-AND-reproducible column set ``transform()`` builds (raw columns the
        full-data MRMR kept, plus stable engineered names it also surfaced) from ``full_mrmr_``'s own
        ``get_feature_names_out()`` rather than requiring an ``X`` to call ``transform`` against."""
        if not hasattr(self, "full_mrmr_"):
            raise RuntimeError("StabilityFESelector.get_feature_names_out called before fit; " "no full_mrmr_ recipes are available.")
        full_names = list(self.full_mrmr_.get_feature_names_out(input_features))
        stable_present = [c for c in self.stable_set_ if c in full_names]
        full_engineered = set(_engineered_union(self.full_mrmr_).keys())
        raw_kept = [c for c in full_names if c not in full_engineered]
        keep = raw_kept + [c for c in stable_present if c not in raw_kept]
        return np.asarray(keep, dtype=object)
