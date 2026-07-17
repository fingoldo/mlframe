"""Regression: ``_mrmr_instance_state_size_bytes`` walks EVERY instance attribute, not a stale allowlist.

Pre-fix the estimator walked a hardcoded attribute tuple (``mi_scores_``, ``_selectors_``,
``_engineered_features_``, ``ranking_``, ``support_``, ``selected_features_``). Audited (grepped the
whole ``feature_selection/filters`` tree): MRMR never assigns ``mi_scores_``, ``_selectors_``, or
``ranking_`` ANYWHERE, and ``selected_features_`` is assigned only by OTHER selector classes (ace.py,
boruta_shap, hybrid_selector.py, shap_proxied_fs, rfecv), never MRMR. So on a REALISTIC fitted MRMR
instance -- whose actual large fit-time state lives in differently-named attributes such as
``_engineered_continuous_``, ``fe_rejection_ledger_``, ``mrmr_gains_`` -- the old estimator returned
~0, making the ``fit_cache_max_mb`` LRU eviction loop a no-op (it never saw the cache as "over budget").

This module pins: (a) the OLD attribute-list approach genuinely returns 0 on a realistic instance shape
(documents the bug was real), (b) the NEW generic ``vars(instance)`` walk correctly picks up ndarray /
dict-of-ndarray / list-of-ndarray state under ANY attribute name, and (c) the estimate is a plausible
match to the true summed ``.nbytes`` on a large known fixture.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import _mrmr_instance_state_size_bytes

# The exact pre-fix allowlist, reproduced here so the regression test does not depend on the (now
# removed) old implementation still existing anywhere in the source tree.
_OLD_ALLOWLIST = ("mi_scores_", "_selectors_", "_engineered_features_", "ranking_", "support_", "selected_features_")


def _legacy_estimator(instance) -> int:
    total = 0
    for _attr in _OLD_ALLOWLIST:
        _v = getattr(instance, _attr, None)
        if _v is None:
            continue
        _nb = getattr(_v, "nbytes", None)
        if isinstance(_nb, int):
            total += _nb
            continue
        if isinstance(_v, dict):
            for _vv in _v.values():
                _vvnb = getattr(_vv, "nbytes", None)
                if isinstance(_vvnb, int):
                    total += _vvnb
        elif isinstance(_v, (list, tuple)):
            for _item in _v:
                _inb = getattr(_item, "nbytes", None)
                if isinstance(_inb, int):
                    total += _inb
    return total


class _RealisticFittedMRMR:
    """Mimics the state a real MRMR.fit() actually leaves on the instance -- large arrays under names
    the old allowlist never covered, plus a small ``support_`` (the only allowlisted name MRMR really
    assigns, but tiny -- a boolean mask over the input feature count)."""

    def __init__(self, n_features: int = 50, n_rows: int = 20_000, n_engineered: int = 30):
        # MRMR does assign support_, but it is a tiny boolean mask -- not the dominant footprint.
        self.support_ = np.zeros(n_features, dtype=bool)
        # The REAL large fit-time state: per-engineered-column full-length continuous float64 arrays.
        self._engineered_continuous_ = {f"eng_{i}": np.zeros(n_rows, dtype=np.float64) for i in range(n_engineered)}
        # Other realistic dict-of-array / list-of-array provenance state.
        self.mrmr_gains_ = {f"col_{i}": np.zeros(8, dtype=np.float64) for i in range(n_features)}
        self.fe_rejection_ledger_ = [np.zeros(4, dtype=np.float64) for _ in range(10)]


def test_old_allowlist_approach_returns_near_zero_on_realistic_instance():
    """Documents the bug was real: the pre-fix estimator misses the dominant state entirely."""
    inst = _RealisticFittedMRMR(n_features=50, n_rows=20_000, n_engineered=30)
    legacy_bytes = _legacy_estimator(inst)
    # Only support_ (50 bools = 50 bytes) is visible to the old allowlist.
    assert legacy_bytes <= 64, f"expected the legacy allowlist to see ~nothing on a realistic instance; got {legacy_bytes}"


def test_generic_walk_finds_the_dominant_engineered_continuous_state():
    inst = _RealisticFittedMRMR(n_features=50, n_rows=20_000, n_engineered=30)
    new_bytes = _mrmr_instance_state_size_bytes(inst)
    # _engineered_continuous_ alone: 30 * 20_000 * 8 bytes = 4_800_000 bytes.
    expected_eng_cont = 30 * 20_000 * 8
    assert new_bytes >= expected_eng_cont, f"generic walk must count _engineered_continuous_ ({expected_eng_cont} bytes); got {new_bytes}"


def test_generic_walk_estimate_is_a_close_match_to_true_nbytes_sum():
    """Construct an instance with a fully known set of ndarray attributes and assert the estimate is
    within a sane range of the TRUE summed .nbytes (not just "non-zero")."""

    class _KnownInstance:
        def __init__(self):
            self.arr_a = np.zeros(10_000, dtype=np.float64)  # 80_000 bytes
            self.arr_b = np.zeros((100, 50), dtype=np.float32)  # 20_000 bytes
            self.dict_of_arrays = {f"k{i}": np.zeros(1_000, dtype=np.float64) for i in range(5)}  # 5*8000=40_000
            self.list_of_arrays = [np.zeros(500, dtype=np.int64) for _ in range(3)]  # 3*4000=12_000
            self.scalar_attr = 42
            self.string_attr = "not an array"
            self.none_attr = None

    inst = _KnownInstance()
    true_total = inst.arr_a.nbytes + inst.arr_b.nbytes + sum(v.nbytes for v in inst.dict_of_arrays.values()) + sum(v.nbytes for v in inst.list_of_arrays)
    estimate = _mrmr_instance_state_size_bytes(inst)
    assert estimate == true_total, f"expected exact match on a fully-known ndarray-only fixture: {estimate} != {true_total}"


def test_generic_walk_handles_empty_instance():
    class _Empty:
        pass

    assert _mrmr_instance_state_size_bytes(_Empty()) == 0


def test_generic_walk_never_raises_on_pathological_attributes():
    """Cyclic self-reference / non-array garbage attributes must not crash the estimator (best-effort)."""

    class _Pathological:
        def __init__(self):
            self.self_ref = self
            self.weird_dict = {"a": object(), "b": np.zeros(10)}
            self.weird_list = [object(), np.zeros(5)]

    inst = _Pathological()
    n = _mrmr_instance_state_size_bytes(inst)
    assert n >= np.zeros(10).nbytes + np.zeros(5).nbytes
