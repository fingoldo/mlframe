"""biz_val for MULTICOLLINEARITY pollution (Q5 coverage dimension).

The redundancy/cluster suites prove a selector aggregates noisy *reflections* of a hidden factor. This
file probes the harsher, explicitly-degenerate forms of collinearity a real frame carries and that a
selector must survive + clean up:

  (a) RANK-DEFICIENT block: ``x3 = 2*x1 - x2`` exactly -> the design Gram is SINGULAR (det 0). A selector
      must not crash on it (no LinAlgError surfacing through to the user), and ideally must not keep all
      three of {x1, x2, x3} (which keeps the rank deficiency in the selected subset).
  (b) HIGH-VIF cluster: 5 features each ~0.95-correlated off a shared base (per-feature VIF > 20). A
      multicollinearity-aware selector keeps a REPRESENTATIVE, not the whole cluster. We measure how many
      of the 5 survive.
  (c) NEAR-PERFECT collinear pair: ``pair_b ~= pair_a`` at corr ~0.999. A selector should not keep both.
  (d) LINEAR-COMBO signal: ``y = sign(x1 + x2 + eps)`` -- NEITHER x1 nor x2 alone is strong, and the
      rank-deficient surrogate ``x3`` alone reaches only ~0.6-0.68 AUC. A selector "recovers" the signal
      iff its selected subset reaches near the x1+x2 baseline AUC (~0.99).

For every selector in ``SELECTOR_SPECS`` we assert, on the SAME polluted fixture:
  1. it does NOT crash on the singular block (hard, all selectors);
  2. it keeps at most a bounded fraction of the 5-feature high-VIF cluster (representative, not all 5);
  3. its selected subset RECOVERS the linear-combo signal (downstream AUC near baseline);
  4. post-selection max-VIF / condition-number is bounded BELOW the polluted-full value (it reduced the
     multicollinearity).

Where a selector measurably fails a contract it *should* meet -- keeps the whole cluster, keeps the
singular triple so post-VIF stays unbounded, or collapses so hard it loses the signal -- the assertion is
written to the CORRECT behavior and marked ``xfail(strict=False)`` with ``reason="FS GAP: ..."``: a
documented capability gap, never a weakened assertion.

Measured floors (seed-0, n=600 unless noted; floors set ~5-15% below measured so seed noise does not trip
them, real regressions do):
  - full-frame max-VIF ~510, condition number ~6e15 (the pollution is real);
  - x1+x2 baseline AUC ~0.991; x3-alone AUC ~0.68 (the recovery bar discriminates);
  - GroupAware(RFECV) cleanly keeps {x1,x2}, 0/5 cluster, maxVIF 1.0, AUC 0.991 (the positive control);
  - plain RFECV(argmax) keeps the FULL 5/5 cluster on a majority of seeds (the cluster GAP);
  - MRMR collapses to {x3} every seed -> AUC ~0.6 (the recovery GAP);
  - ShapProxiedFS keeps the near-collinear pair both -> maxVIF stays ~506 (the VIF GAP).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from tests.feature_selection._selector_factories import SELECTOR_SPECS, selected_names, spec_params


# --------------------------------------------------------------------------- fixtures / metrics


_CLUSTER_PREFIX = "vif"
_CLUSTER_SIZE = 5


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def make_multicollinear_pollution(n: int = 600, seed: int = 0):
    """Signal+noise frame polluted with all four degeneracy forms (a)-(d). See module docstring.

    Returns ``(X, y)`` with columns:
      x1, x2          -- the two independent drivers of y (the linear-combo signal);
      x3              -- ``2*x1 - x2`` exactly (the rank-deficient / singular block);
      vif0..vif4      -- 5 features ~0.95-corr off a shared base (the high-VIF cluster, none drive y);
      pair_a, pair_b  -- a corr~0.999 near-perfect collinear pair (neither drives y);
      noise0..noise3  -- pure noise.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = 2.0 * x1 - x2  # exact linear combination -> singular Gram
    base = rng.normal(size=n)
    vif = [base + 0.33 * rng.normal(size=n) for _ in range(_CLUSTER_SIZE)]
    pair_a = rng.normal(size=n)
    pair_b = pair_a + 0.045 * rng.normal(size=n)  # corr ~0.999
    noise = [rng.normal(size=n) for _ in range(4)]

    cols: dict = {"x1": x1, "x2": x2, "x3": x3}
    for i, v in enumerate(vif):
        cols[f"{_CLUSTER_PREFIX}{i}"] = v
    cols["pair_a"] = pair_a
    cols["pair_b"] = pair_b
    for i, nz in enumerate(noise):
        cols[f"noise{i}"] = nz

    X = pd.DataFrame(cols)
    y = ((x1 + x2 + 0.25 * rng.normal(size=n)) > 0).astype(np.int64)
    return X, pd.Series(y, name="y")


def _max_vif(Xsub: pd.DataFrame) -> float:
    """Max per-feature VIF over the columns of ``Xsub`` (diag of the inverse correlation matrix).

    Returns ``inf`` if the correlation matrix is singular (exact rank deficiency, e.g. {x1,x2,x3} all
    kept) -- that is the desired signal: a selection that retains a rank-deficient block has NOT reduced
    multicollinearity, so its bound check must register an unbounded VIF rather than a misleading finite
    (possibly negative, from a numerically-singular inverse) number.
    """
    cols = [c for c in Xsub.columns]
    if len(cols) < 2:
        return 1.0
    Xv = Xsub.to_numpy(dtype=float)
    Xv = Xv - Xv.mean(axis=0)
    sd = Xv.std(axis=0)
    keep = sd > 1e-9
    if keep.sum() < 2:
        return 1.0
    c = np.corrcoef(Xv[:, keep], rowvar=False)
    # Rank test first: an exactly-singular Gram (kept rank-deficient block) -> treat VIF as unbounded.
    if np.linalg.matrix_rank(c, tol=1e-8) < c.shape[0]:
        return float("inf")
    try:
        inv = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        return float("inf")
    d = float(np.max(np.diag(inv)))
    return d if d > 0 else float("inf")


def _cluster_kept(names) -> int:
    return sum(1 for nm in names if str(nm).startswith(_CLUSTER_PREFIX))


def _selected_in_X(names, X) -> list[str]:
    return [c for c in (str(nm) for nm in names) if c in X.columns]


def _subset_auc(names, X, y, cv: int = 4) -> float:
    cols = _selected_in_X(names, X)
    if not cols:
        return float("nan")
    return float(
        cross_val_score(
            LogisticRegression(max_iter=400),
            X[cols],
            y,
            cv=cv,
            scoring="roc_auc",
        ).mean()
    )


def _baseline_auc(X, y, cv: int = 4) -> float:
    return float(
        cross_val_score(
            LogisticRegression(max_iter=400),
            X[["x1", "x2"]],
            y,
            cv=cv,
            scoring="roc_auc",
        ).mean()
    )


# --------------------------------------------------------------------------- the pollution is real


def test_fixture_pollution_is_severe():
    """Sanity floor: the polluted frame really IS multicollinear, and x3-alone cannot recover the signal.

    Pins the discriminating bars the per-selector tests lean on, so a fixture drift that quietly removed
    the pollution (or made x3 alone sufficient) would fail HERE first, not silently pass every selector.
    """
    X, y = make_multicollinear_pollution(seed=0)
    full_vif = _max_vif(X)
    cond = float(np.linalg.cond((X.to_numpy(float) - X.to_numpy(float).mean(0)) / (X.to_numpy(float).std(0) + 1e-12)))
    base = _baseline_auc(X, y)
    x3_auc = _subset_auc(["x3"], X, y)

    print(f"[mc-fixture] full max-VIF={full_vif} cond={cond:.2e} base(x1,x2)AUC={base:.3f} x3-alone AUC={x3_auc:.3f}")
    assert full_vif == float("inf"), "exact x3=2x1-x2 must make the full Gram singular (unbounded VIF)"
    assert cond > 1e6, f"condition number should be huge under exact rank deficiency, got {cond:.2e}"
    assert base >= 0.95, f"x1+x2 baseline AUC floor 0.95 (measured ~0.991), got {base:.3f}"
    assert x3_auc <= 0.80, f"x3 alone must be a WEAK recoverer (measured ~0.68), got {x3_auc:.3f}"


# --------------------------------------------------------------------------- per-selector contracts


# Selectors that, on this fixture, KEEP a majority of the 5-feature high-VIF cluster (>= 3 of 5 on a
# majority of seeds) -- a documented multicollinearity-reduction GAP. Plain RFECV(argmax) keeps the full
# 5/5 on seeds 0 and 1; HybridSelector keeps 3-4/5 on seeds 1 and 2.
_CLUSTER_KEEP_GAP = {"RFECV", "HybridSelector"}

# Selectors whose selected subset stays rank-deficient / high-VIF on this fixture (post-VIF not bounded
# below the polluted value): plain RFECV keeps the whole cluster; ShapProxiedFS keeps the near-collinear
# pair AND the singular triple; BorutaShap and HybridSelector keep the singular {x1,x2,x3} triple.
_VIF_REDUCE_GAP = {"RFECV", "ShapProxiedFS", "BorutaShap", "HybridSelector"}

# Selectors that collapse so hard they lose the linear-combo signal: MRMR reduces to {x3} every seed,
# reaching only ~0.6-0.68 AUC vs the ~0.991 x1+x2 baseline.
_RECOVERY_GAP = {"MRMR"}


def _fit_or_report(spec, X, y):
    sel = spec.make("binary")
    sel.fit(X, y)
    names = selected_names(sel)
    cols = _selected_in_X(names, X)
    return names, cols


@pytest.mark.parametrize("spec", spec_params())
def test_no_crash_on_singular_block(spec):
    """Every selector must SURVIVE the exact rank-deficient block x3=2*x1-x2 (no LinAlgError to the user).

    This is the universal, non-negotiable contract -- a singular design column is a routine real-frame
    occurrence (one-hot redundancy, accounting identities) and must never crash feature selection.
    """
    X, y = make_multicollinear_pollution(seed=0)
    names, cols = _fit_or_report(spec, X, y)
    print(f"[no-crash] {spec.name}: kept {len(names)} -> {list(map(str, names))}")
    assert len(cols) >= 0  # reaching here == it did not raise; the real assert is the absence of an exception


@pytest.mark.parametrize("spec", spec_params())
def test_does_not_keep_whole_high_vif_cluster(spec):
    """A multicollinearity-aware selector keeps a REPRESENTATIVE of the 5-feature ~0.95-corr cluster,
    not all of it. Contract: <= 2 of the 5 survive. Selectors with a documented keep-the-cluster GAP are
    xfailed to that correct bound.

    Majority-of-seeds: a single lucky seed has repeatedly misled selector benchmarks here, so the keep
    count is voted across 3 seeds and the assertion uses the median.
    """
    keeps = []
    for seed in (0, 1, 2):
        X, y = make_multicollinear_pollution(seed=seed)
        names, _ = _fit_or_report(spec, X, y)
        keeps.append(_cluster_kept(names))
    median_keep = int(np.median(keeps))
    print(f"[vif-cluster] {spec.name}: kept-of-5 per seed={keeps} median={median_keep}")

    if spec.name in _CLUSTER_KEEP_GAP:
        pytest.xfail(reason=f"FS GAP: {spec.name} keeps a majority of the high-VIF cluster (median {median_keep}/5)")
    assert median_keep <= 2, (
        f"{spec.name} kept {median_keep}/5 of the high-VIF cluster (per-seed {keeps}); a multicollinearity-aware selector should keep a representative (<=2)"
    )


@pytest.mark.parametrize("spec", spec_params())
def test_recovers_linear_combo_signal(spec):
    """The selected subset must RECOVER the y=sign(x1+x2) signal: downstream LogReg AUC within 0.05 of the
    x1+x2 baseline (~0.991). x3 alone reaches only ~0.68, so a selector that collapses onto the singular
    surrogate fails this. The recovery-GAP selector (MRMR -> {x3}) is xfailed to the correct bar.
    """
    X, y = make_multicollinear_pollution(seed=0)
    base = _baseline_auc(X, y)
    names, _ = _fit_or_report(spec, X, y)
    auc = _subset_auc(names, X, y)
    print(f"[recovery] {spec.name}: subset AUC={auc:.3f} vs baseline={base:.3f} kept={list(map(str, names))}")

    if spec.name in _RECOVERY_GAP:
        pytest.xfail(reason=f"FS GAP: {spec.name} collapses onto the rank-deficient surrogate and loses the x1+x2 signal (AUC {auc:.3f})")
    assert np.isfinite(auc), f"{spec.name} selected an empty/untransformable subset"
    assert auc >= base - 0.05, f"{spec.name} subset AUC {auc:.3f} fell >0.05 below the x1+x2 baseline {base:.3f}; it did not recover the linear-combo signal"


@pytest.mark.parametrize("spec", spec_params())
def test_reduces_multicollinearity(spec):
    """Post-selection max-VIF must be bounded BELOW the polluted-full value (the selector reduced the
    multicollinearity): a finite VIF below a generous 20.0 ceiling AND strictly less than the full-frame
    VIF (which is infinite under the kept singular block). Selectors that keep a rank-deficient subset or
    the full cluster (post-VIF stays unbounded/high) are xfailed to the correct bound.
    """
    X, y = make_multicollinear_pollution(seed=0)
    _names, cols = _fit_or_report(spec, X, y)
    post_vif = _max_vif(X[cols]) if cols else 1.0
    print(f"[vif-reduce] {spec.name}: post max-VIF={post_vif} on {cols}")

    if spec.name in _VIF_REDUCE_GAP:
        pytest.xfail(reason=f"FS GAP: {spec.name} leaves a rank-deficient / high-VIF subset (post max-VIF {post_vif})")
    assert np.isfinite(post_vif), f"{spec.name} kept a rank-deficient subset (singular Gram, max-VIF inf); it did not break the collinearity"
    assert post_vif < 20.0, (
        f"{spec.name} post-selection max-VIF {post_vif:.1f} exceeds the 20.0 ceiling; the high-VIF cluster / collinear pair was not reduced to a representative"
    )


# --------------------------------------------------------------------------- positive control


def test_group_aware_rfecv_is_the_clean_positive_control():
    """The production-default GroupAware(RFECV) (cluster_reduce=ON via the registry) is the positive
    control that proves the contracts are SATISFIABLE on this fixture: it drops the whole cluster AND the
    singular x3, keeping {x1, x2} -- 0/5 cluster, max-VIF 1.0, AUC ~0.991. If THIS regresses, the
    cluster-reduce path itself broke (not just a per-selector gap).
    """
    spec = SELECTOR_SPECS["GroupAware(RFECV)"]
    X, y = make_multicollinear_pollution(seed=0)
    base = _baseline_auc(X, y)
    names, cols = _fit_or_report(spec, X, y)
    kept_cluster = _cluster_kept(names)
    post_vif = _max_vif(X[cols]) if cols else 1.0
    auc = _subset_auc(names, X, y)
    print(f"[positive-control] GroupAware(RFECV): kept={list(map(str, names))} cluster={kept_cluster}/5 maxVIF={post_vif} AUC={auc:.3f}")

    assert kept_cluster <= 1, f"positive control kept {kept_cluster}/5 of the high-VIF cluster"
    assert np.isfinite(post_vif) and post_vif < 5.0, f"positive control post max-VIF {post_vif} not bounded"
    assert auc >= base - 0.05, f"positive control AUC {auc:.3f} below baseline {base:.3f}"
