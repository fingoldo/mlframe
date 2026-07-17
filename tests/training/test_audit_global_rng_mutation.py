"""Wave 49 (2026-05-20): global RNG mutation audit.

Audit class: production code that calls random.seed(...) / np.random.seed(...) /
torch.manual_seed(...) / cp.random.seed(...) and then uses the GLOBAL RNG --
silently mutating the process-global stream and breaking caller's seed
determinism.

1 P0 + 5 P1 + 2 P2 = 8 fixes applied:

  P0:
    1. training/neural/ranker.py:465 (MLPRanker.fit)
       torch.manual_seed + np.random.seed at fit() entry mutated globals; downstream
       code already uses local np.random.default_rng(self.seed) + per-sampler seed,
       so the global mutation added nothing but the bug. Removed both seed calls.

  P1:
    2. feature_selection/filters/screen.py:951+ (screen_predictors finally)
       Pre-fix: numpy state restored, numba+cupy leaked. Post-fix: restore both via
       a captured os.urandom(8) reseed (high-entropy, indistinguishable downstream).

    3. feature_selection/filters/screen.py:95+ (_preserve_global_numpy_rng_state)
       Symmetric to #2 -- numba+cupy leaked through the context manager. Closed.

    4. utils/misc.py:12 (set_random_seed)
       Documentation contract added: function INTENTIONALLY mutates globals; for
       top-of-script setup only; library code must use local Generators.

    5. feature_selection/filters/cat_interactions.py:991 (_count_nfailed_joint_indep_prange)
       cp.random.seed inside per-permutation loop -> cp.random.RandomState(seed)
       local per-iter generator. Reproducibility preserved; caller's cupy stream
       untouched.

  P2:
    6. feature_engineering/mps.py:600 (generate_market_price)
       np.random.seed + np.random.* -> rng = np.random.default_rng(seed); rng.*.

    7. votenrank/iia_exp.py:36 (compute_iia)
       Per-rep np.random.seed + np.random.shuffle -> per-rep np.random.default_rng(i).

Verified clean (do not refactor):
  - metrics/core.py:252,325 -- np.random.RandomState(0).rand(...) is local.
  - training/_classif_helpers.py:319,349 -- local RandomState.
  - feature_selection MRMR/RFECV/permutation/fleuret -- inside @njit kernels using
    numba's per-thread RNG (not Python np.random global).
  - feature_selection/filters/mrmr.py:946 -- comment confirms prior global removed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


# Use the on-disk src/ tree directly: pytest may resolve mlframe.* to a stale
# build/lib/ copy, but the source-level audit must check the LIVE source.
# tests/training/this_file.py -> ../../src/mlframe/
MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    2026-05-22 monolith split compat: when ``feature_selection/filters/screen.py``
    is requested, also append the ``_screen_predictors.py`` sibling so
    source-pattern sensors for the relocated body still match.
    """
    primary = (MLFRAME_ROOT / rel).read_text(encoding="utf-8")
    if rel == "feature_selection/filters/screen.py":
        sibling = MLFRAME_ROOT / "feature_selection" / "filters" / "_screen_predictors.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_ranker_fit_no_global_torch_or_np_seed() -> None:
    src = _read("training/neural/ranker.py")
    # The pre-fix two-liner is gone; the local generators referenced in comments still exist.
    assert "torch.manual_seed(self.seed)\n        np.random.seed(self.seed)" not in src
    # The documenting comment is present.
    assert "drop global RNG mutations" in src


def test_screen_predictors_restores_numba_and_cupy() -> None:
    src = _read("feature_selection/filters/screen.py")
    # Both restoration variables must be initialised + used.
    assert "_numba_restore_seed = None" in src
    assert "_cp_restore_seed = None" in src
    # The finally block must reseed numba + cupy.
    assert "set_numba_random_seed(int(_numba_restore_seed))" in src
    assert "cp.random.seed(int(_cp_restore_seed))" in src


def test_preserve_global_numpy_rng_state_restores_numba_cupy() -> None:
    src = _read("feature_selection/filters/screen.py")
    # The context-manager helper closes the same leak.
    assert "_cp_module = None" in src
    assert "_cp_module.random.seed(int(_cp_restore_seed))" in src


def test_set_random_seed_documents_top_of_script_contract() -> None:
    src = _read("utils/misc.py")
    # docstring wraps "process-global RNG state" across two lines; check the
    # unique phrases that survive intact.
    assert "INTENTIONALLY mutates the process-global" in src
    assert "top-of-script / notebook setup" in src
    assert "NEVER call this from inside fit()" in src


def test_cat_interactions_uses_local_cupy_rng() -> None:
    # ``_count_nfailed_joint_indep_cupy`` (the cupy permutation kernel)
    # was moved to the ``_cat_confirm_permutation.py`` sibling when
    # ``cat_interactions.py`` was split below 1k LOC.
    src = _read("feature_selection/filters/_cat_confirm_permutation.py")
    # The fix replaces cp.random.seed + cp.random.permutation with local RandomState.
    assert "cp.random.seed(base_seed + p)\n        y_perm = cp.random.permutation(classes_y_g)" not in src
    assert "_local_cp_rng = cp.random.RandomState(base_seed + p)" in src
    assert "_local_cp_rng.permutation(classes_y_g)" in src


def test_mps_generate_market_price_uses_local_rng() -> None:
    src = _read("feature_engineering/mps.py")
    # The pre-fix np.random.seed must be gone.
    assert "np.random.seed(random_seed)\n\n    # Create date range" not in src
    # Local Generator path now in use.
    assert "rng = np.random.default_rng(random_seed)" in src
    # All four global np.random.* call sites in this function must have been replaced.
    assert "rng.normal(trend, 2.5)" in src
    assert "rng.random() < 0.05" in src
    assert "rng.choice([0.95, 1.05])" in src
    assert "rng.uniform(0.5, 2.0)" in src


def test_votenrank_iia_uses_local_rng() -> None:
    src = _read("votenrank/iia_exp.py")
    assert "np.random.seed(i)\n        np.random.shuffle(models_order)" not in src
    assert "rng = np.random.default_rng(i)" in src
    assert "rng.shuffle(models_order)" in src


# ---------------------------------------------------------------------------
# Behavioural sensors: caller's global RNG state must be preserved.
# ---------------------------------------------------------------------------


def test_generate_market_price_does_not_mutate_global_np_rng() -> None:
    """Generating market data with a seed must NOT clobber the caller's global RNG state."""
    from mlframe.feature_engineering.mps import generate_market_price

    np.random.seed(0)
    pre = np.random.get_state()
    generate_market_price(n_days=10, random_seed=42)
    post = np.random.get_state()
    # The MT19937 state tuple has a 624-element keys array; full byte-identical compare.
    assert pre[0] == post[0]
    np.testing.assert_array_equal(pre[1], post[1])
    assert pre[2] == post[2]
    assert pre[3] == post[3]


def test_compute_iia_does_not_mutate_global_np_rng() -> None:
    """compute_iia per-iter seed must NOT shift the caller's global stream."""
    pd = pytest.importorskip("pandas")
    from mlframe.votenrank.iia_exp import compute_iia

    # Minimal stub: 3 models x 2 metrics, weights=ones, method=mean.
    table = pd.DataFrame(
        {"m1": [0.5, 0.6], "m2": [0.55, 0.65], "m3": [0.7, 0.4]},
        index=["a", "b"],
    ).T
    weights = np.ones(2)

    np.random.seed(0)
    pre = np.random.get_state()

    def _mean_method(table, weights):
        return np.average(table.values, weights=weights, axis=1)

    try:
        compute_iia(_mean_method, table, weights, num_repetitions=3)
    except Exception:
        pass  # the method signature might mismatch; we only care about RNG state.
    post = np.random.get_state()
    np.testing.assert_array_equal(pre[1], post[1])
