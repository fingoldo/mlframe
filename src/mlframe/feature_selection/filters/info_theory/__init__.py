"""Information-theoretic primitives: entropy, mutual information, conditional MI.

All functions are ``@njit``-compiled and operate on integer-encoded arrays produced upstream by :mod:`.discretization`.

Contents
--------
* ``merge_vars``    -- collapse multiple ordinal-encoded variables into a single 1-D class array (used as the histogram building block).
* ``entropy``       -- Shannon entropy ``-sum(p * log p)``.
* ``mi``            -- mutual information ``I(X; Y) = H(X) + H(Y) - H(X, Y)`` computed via entropy decomposition.
* ``conditional_mi`` -- ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)`` with a pluggable entropy cache. Each cache branch owns its own
  ``key_z`` / ``key_xz`` / ``key_yz`` / ``key_xyz`` local; ``test_info_theory_cache.py`` enumerates all four ``(can_use_x_cache, can_use_y_cache)`` combos and
  pins down exactly which keys land in the cache.
* ``compute_mi_from_classes`` -- mutual information directly from two pre-computed class vectors and their marginals (used by the permutation loop where
  ``classes_y`` is shuffled in place).

This package is the carved form of the former ``info_theory.py`` module; the public surface is unchanged and every name below resolves from
``mlframe.feature_selection.filters.info_theory`` exactly as before.
"""
from __future__ import annotations

from ._class_encoding import joint_entropy_2var, joint_freqs_2var, merge_vars
from ._entropy_kernels import (
    conditional_mi,
    conditional_symmetric_uncertainty,
    entropy,
    entropy_miller_madow,
    mi,
    mi_miller_madow,
    mi_miller_madow_correct,
    symmetric_uncertainty,
)
from ._state_and_dispatch import (
    _BUR_STATE,
    _JMIM_STATE,
    _SU_STATE,
    cmi_or_csu,
    get_bur_lambda,
    get_cmi_perm_stop,
    get_group_mi,
    get_pid_synergy_bonus,
    get_relaxmrmr_alpha,
    mi_or_su,
    set_bur_lambda,
    set_cmi_perm_stop,
    set_group_mi,
    set_jmim_aggregator,
    set_mi_miller_madow,
    set_pid_synergy_bonus,
    set_relaxmrmr_alpha,
    set_su_normalization,
    use_group_mi,
    use_jmim_aggregator,
    use_mi_miller_madow,
    use_su_normalization,
)
from ._class_mi_kernels import (
    compute_mi_from_classes,
    compute_mi_mm_from_classes,
    compute_relevance_score,
    compute_su_from_classes,
    mi_or_su_from_classes,
)
from ._batch_kernels import (
    _relevance_from_dense,
    batch_mi_with_noise_gate,
    batch_mi_with_noise_gate_v2,
    batch_pair_mi_prange,
    batch_triple_mi_prange,
    select_batch_mi_kernel,
)

__all__ = [
    "merge_vars",
    "joint_freqs_2var",
    "joint_entropy_2var",
    "entropy",
    "entropy_miller_madow",
    "mi_miller_madow_correct",
    "mi",
    "symmetric_uncertainty",
    "conditional_symmetric_uncertainty",
    "conditional_mi",
    "use_su_normalization",
    "set_su_normalization",
    "use_jmim_aggregator",
    "set_jmim_aggregator",
    "use_mi_miller_madow",
    "set_mi_miller_madow",
    "get_group_mi",
    "set_group_mi",
    "use_group_mi",
    "compute_mi_mm_from_classes",
    "mi_miller_madow",
    "get_bur_lambda",
    "set_bur_lambda",
    "get_relaxmrmr_alpha",
    "set_relaxmrmr_alpha",
    "get_pid_synergy_bonus",
    "set_pid_synergy_bonus",
    "get_cmi_perm_stop",
    "set_cmi_perm_stop",
    "mi_or_su",
    "cmi_or_csu",
    "compute_mi_from_classes",
    "compute_su_from_classes",
    "compute_relevance_score",
    "mi_or_su_from_classes",
    "batch_pair_mi_prange",
    "batch_triple_mi_prange",
    "batch_mi_with_noise_gate",
    "batch_mi_with_noise_gate_v2",
    "select_batch_mi_kernel",
]
