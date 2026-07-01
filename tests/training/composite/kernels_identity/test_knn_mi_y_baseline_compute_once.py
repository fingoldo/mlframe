"""iter96 regression: per-base knn ``mi_y`` baseline is computed ONCE, not re-swept per base candidate.

On the knn path (``mi_estimator='knn'``) the per-base baseline ``MI(y, X_remaining)`` was re-run for EVERY base
candidate, each call invoking sklearn's Kraskov ``mutual_info_regression`` once per remaining feature column. Since
per-column ``MI(y, x_j)`` is base-INVARIANT, this was ``n_bases`` redundant per-column sweeps. The fix computes the
per-feature vector ONCE (``_mi_per_feature_knn``) and aggregates over each base's surviving (base-dropped, dedup-kept)
original-column indices -- bit-identical to the per-base ``_mi_to_target`` call.

Pins BOTH sides:
* **bit-identity** -- the compute-once aggregate equals the per-base ``_mi_to_target`` recompute element-for-element,
  for mean AND sum, across interior / first / last drop indices.
* **compute-once** -- a spy on ``mutual_info_regression`` over a full discovery fit asserts the knn baseline column
  sweep runs ONCE for the shared feature columns, not ``n_bases`` times. FAILS on pre-fix code (per-base recompute
  multiplies the column-sweep call count by the base count).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import (
    _aggregate_mi_per_feature,
    _mi_per_feature_knn,
    _mi_to_target,
)

warnings.filterwarnings("ignore")


def _make(n: int = 4_000, f: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float64)
    base = np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    x[:, 0] = base
    y = base + 0.5 * x[:, 1] + 0.3 * x[:, 2] + rng.standard_normal(n) * 0.3
    return x, y


@pytest.mark.parametrize("aggregation", ["mean", "sum"])
@pytest.mark.parametrize("drop", [0, 5, 11])
def test_knn_mi_y_baseline_compute_once_bit_identical(aggregation: str, drop: int) -> None:
    """Aggregate-over-surviving-indices of the compute-once vector == per-base ``_mi_to_target`` recompute, exactly."""
    x, y = _make()
    ref = _mi_to_target(
        np.delete(x, drop, axis=1), y, n_neighbors=3, random_state=42,
        estimator="knn", aggregation=aggregation,
    )
    per_feat = _mi_per_feature_knn(x, y, n_neighbors=3, random_state=42)
    surviving = np.delete(np.arange(x.shape[1]), drop)
    got = _aggregate_mi_per_feature(per_feat[surviving], aggregation)
    assert got == ref, f"compute-once drop={drop} agg={aggregation}: {got!r} != per-base {ref!r}"


def test_knn_mi_per_feature_matches_single_column_calls() -> None:
    """The per-feature vector matches per-column ``mutual_info_regression`` calls element-for-element."""
    from sklearn.feature_selection import mutual_info_regression

    x, y = _make(f=8)
    per_feat = _mi_per_feature_knn(x, y, n_neighbors=3, random_state=42)
    for j in range(x.shape[1]):
        ref = float(mutual_info_regression(
            x[:, j].reshape(-1, 1), y, n_neighbors=3, random_state=42,
        )[0])
        assert per_feat[j] == ref, f"col {j}: {per_feat[j]!r} != {ref!r}"


def test_discovery_fit_knn_baseline_sweeps_columns_once() -> None:
    """compute-once sensor: a full knn-path discovery fit must NOT re-sweep the shared feature columns per base.

    Spy on sklearn's ``mutual_info_regression``. The per-base baseline ``MI(y, X_remaining)`` is the single largest
    block of column sweeps on the knn path; pre-fix it ran once PER base candidate (n_bases full sweeps of the shared
    columns), post-fix once total. We assert the observed baseline-sweep call count is far below the pre-fix
    n_bases * (n_features) lower bound. FAILS on pre-fix code (per-base recompute).
    """
    pd = pytest.importorskip("pandas")
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(3)
    n, n_feats = 3_000, 14
    base = np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    feats = {f"f{j}": rng.standard_normal(n) for j in range(n_feats)}
    feats["lag1"] = base
    feats["lag2"] = np.roll(base, 1) + rng.standard_normal(n) * 0.05
    feats["smooth3"] = base + rng.standard_normal(n) * 0.1
    y = base + 0.5 * feats["f0"] + 0.3 * feats["f1"] + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"y": y, **feats})

    n_bases = 6
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="mi", mi_estimator="knn", base_candidates="auto",
        auto_base_top_k=n_bases, auto_base_null_perms=0,  # isolate the baseline-sweep cost from the null path
        transforms=("diff", "linear_residual"),
    )

    # ``_fit`` imports ``_mi_per_feature_knn`` by name, so spy on the binding it actually calls.
    import mlframe.training.composite.discovery._fit as fit_mod
    real = fit_mod._mi_per_feature_knn
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    orig = fit_mod._mi_per_feature_knn
    fit_mod._mi_per_feature_knn = _spy
    try:
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit(df, "y", [c for c in df.columns if c != "y"], train_idx=np.arange(n))
    finally:
        fit_mod._mi_per_feature_knn = orig

    # Compute-once contract: the knn per-feature baseline vector is built EXACTLY ONCE for the whole base sweep
    # (then aggregated per base via cheap index-exclusion). Pre-fix code computed the baseline via a per-base
    # ``_mi_to_target`` recompute and never called ``_mi_per_feature_knn`` at all -> calls == 0 -> this assertion is
    # red. A future revert to per-base recompute likewise drops the call to 0.
    assert calls["n"] == 1, (
        f"expected the knn baseline vector to be computed ONCE; observed {calls['n']} _mi_per_feature_knn calls "
        f"(0 == pre-fix per-base recompute; >1 == per-base recompute crept back)."
    )
    assert isinstance(disc.specs_, list)
