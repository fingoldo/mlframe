"""RFECV read-only diagnostics: stability curves, 1-SE / Pareto N-rules, bootstrap CI.

These accessors read only fitted ``self.*_`` state (``cv_results_``, ``feature_importances_``,
``n_features_``) and perform no fits. They are bound onto ``RFECV`` from the parent module bottom.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def cv_results_df_(self) -> "pd.DataFrame":
    """Return cv_results_ as a pd.DataFrame for tabular operations (sort_values, query, plot, to_csv). Built lazily on access; raises if fit() has not run."""
    if not hasattr(self, "cv_results_") or "nfeatures" not in self.cv_results_:
        raise ValueError(
            "cv_results_df_ requires fit() to have been called and "
            "cv_results_ to be populated."
        )
    return pd.DataFrame(self.cv_results_)


def selection_stability_(self, metric: str = "jaccard") -> float:
    """Mean pairwise feature-selection stability across CV folds at the chosen ``n_features_``. Free signal extracted from
    feature_importances_, no extra fits required.

    Args:
        metric: 'jaccard' (default), 'dice', or 'kuncheva'.

    Returns:
        Float in [0, 1] (1 = identical selections across folds, 0 = disjoint). Returns NaN when fewer than 2 folds have FI data
        at n_features_.
    """
    if not hasattr(self, "feature_importances_") or not hasattr(self, "n_features_"):
        from sklearn.exceptions import NotFittedError as _NFE
        raise _NFE("RFECV is not fitted; call fit() first.")
    if self.n_features_ == 0:
        return float("nan")
    # Pull per-fold FI runs at the chosen N: keys are 'N_fold' strings.
    target_prefix = f"{self.n_features_}_"
    per_fold_top: list[set] = []
    for key, fi in self.feature_importances_.items():
        if not key.startswith(target_prefix):
            continue
        if len(fi) < self.n_features_:
            continue
        # Top-N features in this fold by importance value, name as secondary key so tied zero-importance features stay stable across runs.
        top_ids = sorted(fi.keys(), key=lambda k: (-fi[k], str(k)))[: self.n_features_]
        per_fold_top.append(set(top_ids))
    if len(per_fold_top) < 2:
        return float("nan")

    def _pair_stability(a: set, b: set) -> float:
        inter = len(a & b)
        if metric == "jaccard":
            union = len(a | b)
            return inter / union if union else 1.0
        if metric == "dice":
            denom = len(a) + len(b)
            return (2 * inter) / denom if denom else 1.0
        if metric == "kuncheva":
            # Kuncheva's index normalises by chance overlap; needs the universe size N. Range [-1, 1] but clamped to [0, 1] here.
            k = len(a)  # |a| == |b| == n_features_ by construction
            N = self.n_features_in_
            if k == 0 or N == 0 or k == N:
                return 1.0 if a == b else 0.0
            expected = k * k / N
            ki = (inter - expected) / (k - expected)
            return max(0.0, ki)
        raise ValueError(f"Unknown stability metric: {metric!r}")

    pairs = [
        _pair_stability(per_fold_top[i], per_fold_top[j])
        for i in range(len(per_fold_top))
        for j in range(i + 1, len(per_fold_top))
    ]
    return float(np.mean(pairs)) if pairs else float("nan")


def n_features_one_se_(self, direction: str = "min") -> int:
    """1-SE rule, parameterised by ``direction``.

    - 'min' (default, sklearn-canonical): SMALLEST N whose CV mean is within one SE of the best -> most parsimonious within band.
    - 'max' (plateau-resistant): LARGEST N within band -> retains marginally-informative features.
    Without this param the helper returned 'min' always while ``select_optimal_nfeatures_`` with ``rule='one_se_max'`` returned
    the opposite. Direct callers of the helper saw a different N than the picker.

    Returns the integer count, or n_features_ as a fallback if cv_results_ is unavailable.
    """
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' or 'max'; got {direction!r}")
    if not hasattr(self, "cv_results_") or not self.cv_results_.get("nfeatures"):
        return getattr(self, "n_features_", 0)
    nfeatures = np.asarray(self.cv_results_["nfeatures"], dtype=int)
    means = np.asarray(self.cv_results_["cv_mean_perf"], dtype=float)
    stds = np.asarray(self.cv_results_["cv_std_perf"], dtype=float)
    if len(means) == 0:
        return getattr(self, "n_features_", 0)
    # Exclude the 0-features dummy.
    nonzero = nfeatures > 0
    if not nonzero.any():
        return getattr(self, "n_features_", 0)
    nf, m, s = nfeatures[nonzero], means[nonzero], stds[nonzero]
    # mean_perf_weight + std_perf_weight are baked into final_score; for 1-SE we need the unadjusted mean - cv_mean_perf is raw.
    # Mask NaN candidates before argmax: argmax picks a NaN slot when any candidate's cv_mean_perf is all-NaN-folds.
    _finite_mask = np.isfinite(m)
    if not _finite_mask.any():
        # No usable candidates -- fall back to the cached n_features_.
        return getattr(self, "n_features_", 0)
    if not _finite_mask.all():
        nf, m, s = nf[_finite_mask], m[_finite_mask], s[_finite_mask]
    best_idx = int(np.argmax(m))
    threshold = m[best_idx] - s[best_idx]
    # Smallest OR largest N whose mean >= threshold based on direction.
    eligible = nf[m >= threshold]
    if len(eligible) == 0:
        return int(nf[best_idx])
    return int(eligible.min()) if direction == "min" else int(eligible.max())


def stability_vs_n_curve_(self, metric: str = "jaccard") -> dict:
    """Per-N stability of the top-N FI selection across CV folds.

    Returns a dict mapping ``nfeatures -> mean pairwise stability``. Built from the already-collected per-fold FI dict, so no extra
    fits are needed. Combined with ``cv_results_['cv_mean_perf']``, the curve gives a graphical view of where stability AND score
    co-peak (the elbow of stability * score). See ``n_stability_elbow_()``.

    Nogueira & Brown 2018 (JMLR v18/17-514) on stability for FS evaluation.
    """
    if not hasattr(self, "feature_importances_") or not hasattr(self, "feature_names_in_"):
        from sklearn.exceptions import NotFittedError as _NFE
        raise _NFE("RFECV is not fitted; call fit() first.")
    result: dict = {}
    # Bucket FI keys by N.
    by_n: dict = {}
    for key, fi in self.feature_importances_.items():
        try:
            n = int(str(key).split("_")[0])
        except (ValueError, IndexError):
            continue
        by_n.setdefault(n, []).append(fi)
    for n, fi_list in by_n.items():
        if n <= 0 or len(fi_list) < 2:
            continue
        tops = []
        for fi in fi_list:
            if not fi or len(fi) < n:
                continue
            _top = sorted(fi.keys(), key=lambda k: (-fi[k], str(k)))[:n]
            tops.append(set(_top))
        if len(tops) < 2:
            continue
        pairs: list = []
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                inter = len(tops[i] & tops[j])
                if metric == "jaccard":
                    union = len(tops[i] | tops[j])
                    pairs.append(inter / union if union else 1.0)
                elif metric == "dice":
                    denom = len(tops[i]) + len(tops[j])
                    pairs.append((2 * inter) / denom if denom else 1.0)
                else:
                    raise ValueError(f"metric must be 'jaccard' or 'dice'; got {metric!r}")
        if pairs:
            result[n] = float(np.mean(pairs))
    return result


def n_stability_elbow_(self, metric: str = "jaccard") -> int:
    """Pick N at the elbow of ``stability(N) * score(N)``.

    Heuristic alternative to argmax / 1-SE rules: combines selection stability with predictive score on the same N grid; the
    product peaks where both signals agree. Useful for production-retraining configs where stability matters as much as score.

    Returns the chosen N (int), or ``n_features_`` fallback if no curve was computed.
    """
    curve = self.stability_vs_n_curve_(metric=metric)
    if not curve or not hasattr(self, "cv_results_"):
        return getattr(self, "n_features_", 0)
    means = dict(zip(self.cv_results_["nfeatures"], self.cv_results_["cv_mean_perf"]))
    combined = {}
    for n, s in curve.items():
        if n in means and np.isfinite(means[n]):
            combined[n] = float(s) * float(means[n])
    if not combined:
        return getattr(self, "n_features_", 0)
    # Argmax with tie-breaker on smaller N (parsimony when scores tie).
    return int(max(combined.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def pareto_front_(self, metric: str = "jaccard") -> list:
    """Non-dominated (cv_mean_perf MAX, n_features MIN, stability MAX) points over the evaluated N grid.

    Replaces the single scalarisation ``mean*w - std*w - cost*N`` (whose weights are arbitrary) with the
    actual trade-off frontier. Returns a list of dicts ``{n, mean, stability}`` sorted by n, each
    Pareto-optimal: no other evaluated N is >= on mean AND <= on n AND >= on stability (with one strict).
    Read-only diagnostic; pairs with ``pareto_knee_``. Empty if no cv_results_.
    """
    if not hasattr(self, "cv_results_") or not self.cv_results_.get("nfeatures"):
        return []
    nfeat = np.asarray(self.cv_results_["nfeatures"], dtype=int)
    means = np.asarray(self.cv_results_["cv_mean_perf"], dtype=float)
    stab_curve = self.stability_vs_n_curve_(metric=metric) or {}
    pts = []
    for n, m in zip(nfeat, means):
        if n > 0 and np.isfinite(m):
            pts.append({"n": int(n), "mean": float(m), "stability": float(stab_curve.get(int(n), np.nan))})
    if not pts:
        return []
    have_stab = all(np.isfinite(p["stability"]) for p in pts)
    front = []
    for a in pts:
        dominated = False
        for b in pts:
            if b is a:
                continue
            ge_mean = b["mean"] >= a["mean"]; le_n = b["n"] <= a["n"]
            ge_stab = (b["stability"] >= a["stability"]) if have_stab else True
            strict = (b["mean"] > a["mean"]) or (b["n"] < a["n"]) or (have_stab and b["stability"] > a["stability"])
            if ge_mean and le_n and ge_stab and strict:
                dominated = True; break
        if not dominated:
            front.append(a)
    # dedup by n (keep best mean), sort by n
    best_by_n = {}
    for p in front:
        if p["n"] not in best_by_n or p["mean"] > best_by_n[p["n"]]["mean"]:
            best_by_n[p["n"]] = p
    return [best_by_n[k] for k in sorted(best_by_n)]


def pareto_knee_(self, metric: str = "jaccard") -> int:
    """N at the knee of the (mean MAX, n MIN[, stability MAX]) Pareto front: the front point closest to the
    ideal corner after min-max normalising each axis. Weight-free alternative to argmax / 1-SE / elbow.
    Returns n_features_ fallback when no front.

    DIAGNOSTIC, not an accuracy default. Benchmarked across 6 synthetic scenarios it LOST every cell on
    downstream LightGBM AUC (0/6) because the knee favours the parsimonious corner (picks ~4-8 features),
    which under-serves noise-robust downstreams that benefit from more features; argmax / one_se_max won
    (3/3). Use it only when parsimony is an explicit objective (few-feature deployment / interpretability),
    NOT to maximise accuracy. On small p the front can also collapse to 1-2 points (then it equals 1-SE)."""
    front = self.pareto_front_(metric=metric)
    if not front:
        return getattr(self, "n_features_", 0)
    if len(front) == 1:
        return front[0]["n"]
    means = np.array([p["mean"] for p in front]); ns = np.array([p["n"] for p in front], dtype=float)
    stabs = np.array([p["stability"] for p in front])
    have_stab = np.all(np.isfinite(stabs))
    def _nrm(v, maximize):
        lo, hi = np.min(v), np.max(v)
        u = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        return u if maximize else (1.0 - u)
    # distance to ideal (mean=1, n=0->normalised 1 for "fewest", stability=1)
    dist = (1 - _nrm(means, True)) ** 2 + (1 - _nrm(ns, False)) ** 2
    if have_stab:
        dist = dist + (1 - _nrm(stabs, True)) ** 2
    return int(front[int(np.argmin(dist))]["n"])


def n_features_bootstrap_ci_(self, n_bootstrap: int = 200, ci: float = 0.9,
                              random_state: Union[int, None] = None) -> tuple:
    """Parametric bootstrap CI on the optimal n_features_.

    Draws B bootstrap replicates of the cv_results_ score curve by sampling each (mean, std) pair as Normal(mean, std), recomputes
    argmax over the non-zero N values for each replicate, returns (low_pct, n_features_, high_pct) where the percentiles bracket
    ``ci`` mass of the bootstrap distribution.

    Use this to gauge whether n_features_=N is meaningfully different from N+/-5; a wide CI suggests caution about the exact N
    choice. Parametric bootstrap (no raw per-fold scores retained), so it under-estimates true uncertainty when fold scores are
    non-Normal.
    """
    if not hasattr(self, "cv_results_") or not self.cv_results_.get("nfeatures"):
        n = getattr(self, "n_features_", 0)
        return (n, n, n)
    # n_bootstrap<=0 degenerates to empty choices_arr, then int(np.median([])) raises ValueError with a RuntimeWarning.
    if int(n_bootstrap) <= 0:
        n = getattr(self, "n_features_", 0)
        return (n, n, n)
    nfeatures = np.asarray(self.cv_results_["nfeatures"], dtype=int)
    means = np.asarray(self.cv_results_["cv_mean_perf"], dtype=float)
    stds = np.asarray(self.cv_results_["cv_std_perf"], dtype=float)
    nonzero = nfeatures > 0
    if not nonzero.any():
        n = getattr(self, "n_features_", 0)
        return (n, n, n)
    nf, m, s = nfeatures[nonzero], means[nonzero], stds[nonzero]
    rng = np.random.default_rng(random_state)
    choices = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.normal(loc=m, scale=np.maximum(s, 1e-12))
        best_idx = int(np.argmax(sampled))
        choices.append(int(nf[best_idx]))
    choices_arr = np.asarray(choices)
    alpha = (1.0 - float(ci)) / 2.0
    low = int(np.percentile(choices_arr, 100.0 * alpha))
    high = int(np.percentile(choices_arr, 100.0 * (1.0 - alpha)))
    n = getattr(self, "n_features_", int(np.median(choices_arr)))
    return (low, int(n), high)
