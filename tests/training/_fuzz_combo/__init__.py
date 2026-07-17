"""Combo enumerator + results log for train_mlframe_models_suite fuzzing.

Design principles:
  * Deterministic: identical master_seed → identical combo list on any host.
  * Dedup-canonical: combos are canonicalized before hashing so
    semantically-equivalent combos (e.g. align_polars_dicts=True with
    pandas input) collapse to one.
  * Pairwise-covering: the greedy sampler guarantees every
    (axis_i=value_i, axis_j=value_j) pair is exercised at least once.
  * xfail-aware: combos hitting known bugs are auto-marked xfail via a
    declarative rule table — single source of truth shared with tracked
    tests elsewhere in the suite.

Canonicalisation contract (READ THIS BEFORE EDITING ``canonical_key`` /
``_canonical_*``).

Canonicalisation deduplicates SEMANTICALLY-EQUIVALENT combos. It does
NOT silence flaky combos. The two are easy to confuse and the latter is
strictly forbidden.

  Legitimate canon (keep): ``imbalance="balanced"`` collapses regardless
  of the imbalance-mode flag, because at 50/50 the mode produces
  bit-identical data. The two combos really are the same combo, and
  hashing them as one is a memory / time win with no coverage cost.

  ILLEGITIMATE canon (DO NOT WRITE, FIX PROD INSTEAD): zeroing
  ``text_col_count`` for combos that hit a CB hang, forcing
  ``inject_degenerate_cols=False`` for combos that crash CB's cat-feature
  auto-detect, forcing ``remove_constant_columns=True`` for combos that
  break the polars-ds robust scaler. These are real production bugs
  that real users would hit. Hiding them in canon means the fuzz suite
  STAYS GREEN while production stays broken — exactly the inverse of
  what this harness is for.

  If you catch yourself writing a canon rule whose justification
  references a CrashID / fuzz cXXXX / "the X path hangs" — STOP. Find
  the prod fix. Once prod is fixed the canon is unnecessary; until prod
  is fixed, the failure is doing its job by surfacing the bug.

Concrete example, 2026-04-27: the original ``_canonical_text_col_count``
zeroed text columns when CB + small-n + heavy NaN injection landed on
inner-CV folds smaller than CB's default ``occurrence_lower_bound=50``.
The fix was a real production change in
``training/helpers.compute_cb_text_processing`` that scales the floor
proportionally to the fit-time row count (called from
``trainer._train_model_with_fallback`` and ``feature_selection/wrappers.py``
RFECV inner-fold). After the fix, the canon was retired.

See ``CLAUDE.md`` (project root) for the full anti-masking checklist
covering canon, runtime ``*_eff`` rewrites in ``test_fuzz_suite.py``,
``pytest.mark.xfail`` rules, and "0-row defensive guards" in
production code.

Results log: every fuzz run appends one JSONL row per combo to
``tests/training/_fuzz_results.jsonl`` capturing combo key, outcome
(pass/fail/xfail/skip), and — on failure — the exception class and a
one-line summary. That file is the audit trail used by human / agent
follow-ups to decide what to fix next.
"""

from .axes import MODELS, AXES
from .combo import FuzzCombo
from .xfail import KNOWN_XFAIL_RULES, xfail_reason
from .enumerator import (
    enumerate_combos,
    enumerate_combos_3way,
    _build_combo,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _sample_axes,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _powerset_nonempty,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _combo_is_runnable,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _LTR_NATIVE_RANKERS,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _all_axis_pairs,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _combo_pairs,  # noqa: F401 -- re-exported for direct import by test_fuzz_combo_cross_axis.py and others
    _all_axis_triples,  # noqa: F401 -- re-exported for direct import by test_fuzz_3way_suite.py
    _combo_triples,  # noqa: F401 -- re-exported for direct import by test_fuzz_3way_suite.py
)
from .builders import (
    build_cat_fe_config_from_flat,
    build_mrmr_kwargs_from_flat,
    build_mrmr_kwargs,
    build_mlp_kwargs_from_flat,
    build_mlp_kwargs,
    build_shap_proxied_fs_kwargs_from_flat,
    build_shap_proxied_fs_kwargs,
    build_composite_discovery_config_from_flat,
    build_composite_discovery_config,
    build_slice_stable_es_config_from_flat,
    build_slice_stable_es_config,
)
from .results_log import RESULTS_LOG, log_combo_outcome, read_fail_summary
from .perf_mode import apply_perf_mode
from .frame_builder import build_frame_for_combo

__all__ = [
    "MODELS",
    "AXES",
    "FuzzCombo",
    "KNOWN_XFAIL_RULES",
    "xfail_reason",
    "enumerate_combos",
    "enumerate_combos_3way",
    "build_cat_fe_config_from_flat",
    "build_mrmr_kwargs_from_flat",
    "build_mrmr_kwargs",
    "build_mlp_kwargs_from_flat",
    "build_mlp_kwargs",
    "build_shap_proxied_fs_kwargs_from_flat",
    "build_shap_proxied_fs_kwargs",
    "build_composite_discovery_config_from_flat",
    "build_composite_discovery_config",
    "build_slice_stable_es_config_from_flat",
    "build_slice_stable_es_config",
    "RESULTS_LOG",
    "log_combo_outcome",
    "read_fail_summary",
    "apply_perf_mode",
    "build_frame_for_combo",
]
