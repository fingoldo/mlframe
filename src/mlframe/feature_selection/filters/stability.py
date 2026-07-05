"""Stability selection wrapper around MRMR.

mRMR with permutation-test confidence is unstable for small N: the support depends on the seed of the permutation pass. Stability
selection (Meinshausen-Buhlmann 2010 -- analog ``RandomizedLasso`` in old sklearn) addresses this by running mRMR on ``n_bootstraps``
subsamples and recommending only features that appear in the support of at least ``support_threshold`` (default 0.6 = 60%) of runs.

Public class
------------
``StabilityMRMR(estimator, n_bootstraps=20, sample_fraction=0.5, support_threshold=0.6, random_state=None)``

Same ``.fit / .transform / .support_ / .selection_probabilities_`` surface as ``MRMR``. ``selection_probabilities_`` exposes per-feature
inclusion frequency as a numpy float vector for downstream stability plots.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

logger = logging.getLogger(__name__)

# Stratified bootstrap subsampling fires only when the smallest class's expected count in a plain subsample falls below this floor (i.e. the class is at
# real risk of being dropped/starved). Above it, an unstratified draw preserves every class with overwhelming probability, so quota-forcing only perturbs
# the subsample composition without correcting anything.
_STRATIFY_MIN_EXPECTED_PER_CLASS = 25.0


def _support_to_indices(support, n_features: int) -> np.ndarray:
    """Normalise an estimator's ``support_`` to integer column indices.

    A boolean mask (sklearn ``SelectorMixin.get_support()`` default) of length ``n_features`` is converted via ``np.flatnonzero``; an integer-index array
    is returned as-is. Pre-fix a bool mask was coerced through ``astype(int64)`` to ``[0,1,0,...]`` and indexed ``counts`` at positions 0/1 only, silently
    mis-counting every selector that exposes a mask instead of indices.
    """
    arr = np.asarray(support)
    if arr.dtype == bool:
        return np.flatnonzero(arr).astype(np.int64)
    # A full-length object/bool-like mask (e.g. list of python bools) also normalises to indices.
    if arr.size == n_features and arr.dtype == object and arr.size and isinstance(arr.flat[0], (bool, np.bool_)):
        return np.flatnonzero(arr.astype(bool)).astype(np.int64)
    return arr.astype(np.int64)


class StabilityMRMR(BaseEstimator, TransformerMixin):
    """Bootstrap-stability wrapper for mRMR-family selectors.

    Each bootstrap iteration:
    1. Sample ``sample_fraction * n_samples`` rows without replacement (with ``random_state + iteration`` seed).
    2. Fit a clone of ``estimator`` on the subsample.
    3. Record ``estimator.support_`` (set of selected feature indices).

    After all iterations:
    * ``selection_probabilities_[j] = P(feature_j in support across bootstraps)``.
    * ``support_`` = features with prob >= ``support_threshold``.

    Error control. The Meinshausen-Buhlmann (2010, JRSS-B) PFER bound ``E[V] <= q^2 / ((2*pi_thr - 1) * p)`` -- where ``q`` is the
    average number of features selected per bootstrap, ``pi_thr`` the ``support_threshold``, and ``p`` the feature count -- is derived
    under ``sample_fraction = 0.5`` (the canonical complementary-pairs / n/2 subsampling regime). It does NOT hold at other fractions,
    so the default is ``0.5``; raising ``support_threshold`` toward ~0.8 tightens control. The realized PFER bound for a fitted instance
    is exposed via ``pfer_bound_`` (and the per-bootstrap selection count via ``avg_selected_per_bootstrap_``) so callers can check the
    selection against their error budget. The bound is only valid at ``sample_fraction == 0.5``; ``pfer_bound_`` is ``nan`` otherwise.

    Parameters
    ----------
    estimator : BaseEstimator
        Any selector with ``.fit(X, y)`` and ``.support_`` attributes (typically an ``MRMR`` instance).
    n_bootstraps : int, default 20
    sample_fraction : float, default 0.5
        Fraction of rows to subsample per bootstrap. The MB PFER bound is only valid at 0.5 (complementary-pairs subsampling); other
        values trade the error guarantee for a different bias/variance point.
    support_threshold : float, default 0.6
    random_state : int, default None
    n_jobs : int, default 1
        Passes through to ``joblib.Parallel`` for the bootstrap loop.
    stratify : bool, default True
        Preserve the per-class proportions of ``y`` in each bootstrap subsample (class-stratified sampling without replacement). On rare / imbalanced targets an
        unstratified subsample can omit the minority class entirely, giving a single-class fit that the base MI/MRMR selector degenerates on -- its garbage support is then
        silently folded into the inclusion counts. Stratification draws ``round(sample_fraction * n_class)`` rows from each class (floored at 1 per present class) so every
        class survives every bootstrap. ON by default per the corrective-mechanism convention, but it ENGAGES only when a class is genuinely at risk -- when the
        smallest class's expected count in a plain subsample (``sample_fraction * min_class_size``) is below ~25; on a near-balanced target every class survives an
        unstratified draw anyway, so the plain draw is kept (quota-forcing there is a pure perturbation that needlessly shifts inclusion probabilities). Also falls back
        to the plain unstratified draw when ``y`` is not class-like (more than ``max(50, n/2)`` distinct values -- a regression / continuous target). Set ``stratify=False`` for the legacy behaviour.
        Rare classes still need adequate n: per the project rule a 1%-prevalence class needs n >~ 5000 for a reliable split even with stratification.
    """
    def __init__(
        self,
        estimator,
        n_bootstraps: int = 20,
        sample_fraction: float = 0.5,
        support_threshold: float = 0.6,
        random_state: int = None,
        n_jobs: int = 1,
        stratify: bool = True,
    ):
        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.support_threshold = support_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.stratify = stratify

    def fit(self, X, y):
        from joblib import Parallel, delayed
        # 2026-05-30 Wave 9.1 fix (loop iter 41): input validation.
        # Pre-fix:
        #   * sample_fraction=0.05 with n=10 -> sub_size=int(0.5)=0,
        #     every clone got X[0:0] (empty fit), silent garbage
        #     ``selection_probabilities_``.
        #   * n_bootstraps=0 -> counts / 0 = NaN, ``support_=[]``
        #     indistinguishable from a legitimate "all below threshold"
        #     result.
        #   * Negative / out-of-range params leaked to numpy with
        #     unhelpful errors.
        if not isinstance(self.n_bootstraps, (int, np.integer)) or self.n_bootstraps < 1:
            raise ValueError(
                f"StabilityMRMR: n_bootstraps must be a positive integer; "
                f"got {self.n_bootstraps!r}."
            )
        if not (0.0 < float(self.sample_fraction) <= 1.0):
            raise ValueError(
                f"StabilityMRMR: sample_fraction must be in (0, 1]; "
                f"got {self.sample_fraction!r}."
            )
        # support_threshold lives on the probability axis [0, 1] for
        # in-range gating, but the boundary and >1 values are useful
        # sentinels documented in the test contract: 0.0 keeps every
        # touched feature (>=0 is always satisfied) and >1.0 produces
        # an empty support (no probability can clear it). Accept the
        # extended [0, +inf) interval and let the >= mask resolve it.
        if float(self.support_threshold) < 0.0:
            raise ValueError(
                f"StabilityMRMR: support_threshold must be >= 0; "
                f"got {self.support_threshold!r}."
            )
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # Floor at 2 so a clone never sees 0/1-row fit (numerically
        # degenerate; MI estimators raise or return NaN at n<2).
        sub_size = max(2, int(round(self.sample_fraction * n_samples)))
        if sub_size > n_samples:
            raise ValueError(
                f"StabilityMRMR: sub_size ({sub_size}) exceeds n_samples "
                f"({n_samples}); reduce sample_fraction."
            )

        # Generate seeds upfront so the bootstrap is deterministic for a given ``random_state`` regardless of joblib worker order.
        seeds = rng.integers(0, 2 ** 31 - 1, size=self.n_bootstraps)

        # Decide once whether stratified sampling applies: ON when requested AND y looks class-like (few distinct values). A high-cardinality / continuous y is a
        # regression target where per-class stratification is meaningless, so fall back to the plain draw. The per-class index groups are precomputed once and reused.
        _y_arr = np.asarray(y.values if hasattr(y, "values") else y)
        _class_groups = None
        if self.stratify and _y_arr.ndim == 1:
            _uniq, _counts = np.unique(_y_arr, return_counts=True)
            # Stratify only when a class is genuinely AT RISK in the plain draw. Forcing per-class quotas on a near-balanced target is a no-op for class
            # coverage (every class survives an unstratified subsample with overwhelming probability) but still perturbs the subsample composition -- it shifts
            # the noise-feature inclusion probabilities and can push a borderline false positive over ``support_threshold``. The corrective mechanism should
            # fire exactly where it corrects something: when the smallest class's EXPECTED count in an unstratified subsample is small enough that a draw could
            # realistically drop or starve it (``sample_fraction * min_class_size < _STRATIFY_MIN_EXPECTED_PER_CLASS``), per the rare-imbalance regime.
            if _uniq.size <= max(50, n_samples // 2):
                _expected_min_class = float(self.sample_fraction) * int(_counts.min())
                if _expected_min_class < _STRATIFY_MIN_EXPECTED_PER_CLASS:
                    _class_groups = [np.flatnonzero(_y_arr == c) for c in _uniq]

        def _stratified_indices(local_rng) -> np.ndarray:
            # Draw round(sample_fraction * n_class) rows from each class (>=1 per present class) so no class is dropped; the global floor of 2 total rows still holds.
            parts = []
            for grp in _class_groups:
                k = max(1, int(round(self.sample_fraction * grp.size)))
                k = min(k, grp.size)
                parts.append(local_rng.choice(grp, size=k, replace=False))
            return np.concatenate(parts)

        def _one_bootstrap(seed: int) -> np.ndarray:
            local_rng = np.random.default_rng(seed)
            if _class_groups is not None:
                idx = _stratified_indices(local_rng)
            else:
                idx = local_rng.choice(n_samples, size=sub_size, replace=False)
            X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
            est = clone(self.estimator)
            est.fit(X_sub, y_sub)
            return _support_to_indices(est.support_, n_features)

        if self.n_jobs == 1:
            supports = [_one_bootstrap(s) for s in seeds]
        else:
            # backend="threading": each bootstrap fits an estimator clone on a
            # resampled (X, y) tuple. loky's default would deep-copy the whole
            # (X, y) into each worker process -- with the stability path
            # already running inside MRMR's outer FE loop, that's a recipe for
            # the iter-371 paging cascade. Estimator clones share the parent
            # X/y arrays under threading; the inner fit holds the GIL anyway
            # so threading doesn't speed up the fits themselves, but it
            # eliminates the OOM risk and removes loky process-spawn cost.
            supports = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(_one_bootstrap)(s) for s in seeds)

        # Accumulate per-feature inclusion counts.
        counts = np.zeros(n_features, dtype=np.int64)
        for sup in supports:
            counts[sup] += 1

        self.selection_probabilities_ = counts / self.n_bootstraps
        # Compare integer counts against a ceiling rather than float prob >= float threshold: counts/n_bootstraps is not exactly representable
        # (e.g. 12/20 != 0.6 in float64), so a feature selected in exactly threshold*n runs could spuriously fail a direct float >= compare.
        import math as _math
        _min_count = int(_math.ceil(float(self.support_threshold) * self.n_bootstraps - 1e-9))
        self.support_ = np.where(counts >= _min_count)[0]
        self.n_features_ = len(self.support_)
        self.n_features_in_ = n_features

        # Meinshausen-Buhlmann PFER bound E[V] <= q^2 / ((2*pi_thr - 1) * p), with q = avg #selected per bootstrap. Only valid at
        # sample_fraction == 0.5 and pi_thr > 0.5; nan otherwise (the bound's derivation assumes complementary-pairs subsampling and a
        # threshold above 0.5). Exposed so callers can compare the realized bound against their false-positive budget.
        self.avg_selected_per_bootstrap_ = float(np.mean([len(s) for s in supports])) if supports else 0.0
        _pi = float(self.support_threshold)
        if abs(float(self.sample_fraction) - 0.5) <= 1e-9 and _pi > 0.5 and n_features > 0:
            self.pfer_bound_ = (self.avg_selected_per_bootstrap_ ** 2) / ((2.0 * _pi - 1.0) * n_features)
        else:
            self.pfer_bound_ = float("nan")
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X, y=None):
        # 2026-05-30 Wave 9.1 fix (loop iter 42): validate fit-time
        # column semantics at transform. Pre-fix the function
        # positional-indexed via ``X.iloc[:, self.support_]`` with no
        # check that the columns AT transform time match the columns
        # AT fit time. Reordering, renaming, or dropping columns
        # silently returned the wrong slice - downstream models saw
        # feature ``d`` labelled as ``b`` and vice versa.
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            cols = list(X.columns)
            fit_cols = list(self.feature_names_in_)
            if cols != fit_cols:
                # Realign by name when all fit columns are present
                # (sklearn ``_check_feature_names(reset=False)`` semantics).
                if set(fit_cols).issubset(cols):
                    X = X[fit_cols]
                else:
                    missing = sorted(set(fit_cols) - set(cols))
                    raise ValueError(
                        f"StabilityMRMR.transform: X columns differ from "
                        f"fit; missing {missing!r}."
                    )
        elif hasattr(self, "n_features_in_"):
            _ncols = int(X.shape[1])
            if _ncols != int(self.n_features_in_):
                raise ValueError(
                    f"StabilityMRMR.transform: X has {_ncols} features, "
                    f"but StabilityMRMR is expecting "
                    f"{int(self.n_features_in_)} features as input."
                )
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]
