"""Meta-test: ctx-derived kwargs MUST be forwarded to inner-function call sites.

Complements `test_setup_configuration_propagation.py` (which guards the
**constructor layer**: public-API kwarg -> ctx slot). This guards the
**call-site layer**: ctx slot -> inner-function kwarg at the orchestrator call.

The bug class this catches: an inner function (e.g. ``score_ensemble``,
``compute_oof_holdout_predictions``, ``RecurrentClassifierWrapper.fit``,
``compute_dummy_baselines``) ACCEPTS a kwarg like ``group_ids`` / ``sample_weight`` /
``time_ordering`` -- but the orchestrator that has the corresponding ctx slot
forgets to pass it. The inner function silently uses its default (None ->
"no group awareness", "i.i.d. rows", "random shuffle"). Behavioural tests don't
catch this: the model still trains, metrics still look plausible. Per-function
unit tests don't catch this either: they call the inner fn with explicit kwargs.
The gap is at the boundary.

Validated this session against 5 real bugs (wave 12 commit fb52f4c).
"""

from __future__ import annotations

import pathlib

import pytest


import mlframe as _mlframe

_SRC_ROOT = pathlib.Path(_mlframe.__file__).resolve().parent / "training"


# 2026-05-21 monolith split: ``_train_one_target`` body lives in
# ``_phase_train_one_target_body.py``; source-pattern sensors that grep the
# parent file must also read the body sibling. Resolves the core/ dir from
# the installed package so it works regardless of where pytest is invoked.
def _read_phase_train_one_target_combined():
    """Concatenates the split _phase_train_one_target module and its body sibling for source-pattern greps."""
    import pathlib
    import mlframe as _mlframe

    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    return (_core / "_phase_train_one_target.py").read_text(encoding="utf-8") + "\n" + (_core / "_phase_train_one_target_body.py").read_text(encoding="utf-8")


def _read(rel: str) -> str:
    """Read a source file relative to _SRC_ROOT.

    Monolith-split compat: when the requested file is
    ``core/_phase_train_one_target.py``, append every sibling that the
    body file delegates to (ensembling tail, polars fastpath, pre-screen
    gate) so source-pattern sensors that pre-date the splits still match.
    """
    primary = (_SRC_ROOT / rel).read_text(encoding="utf-8")
    if rel == "core/_phase_train_one_target.py":
        for sib in (
            "_phase_train_one_target_body.py",
            "_phase_train_one_target_ensembling.py",
            "_phase_train_one_target_polars_fastpath.py",
            "_phase_train_one_target_pre_screen.py",
            "_phase_train_one_target_model_setup.py",
        ):
            _sib_path = _SRC_ROOT / "core" / sib
            if _sib_path.exists():
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    # Wave 100 monolith split: the cross-target ensemble body that holds
    # the OOF holdout call moved to ``_phase_composite_post_xt_ensemble.py``.
    # Sensor markers still pinned to ``_phase_composite_post.py`` need to
    # see the sibling too.
    if rel == "core/_phase_composite_post.py":
        for sib in (
            "_phase_composite_post_xt_ensemble.py",
            "_phase_composite_post_lag_predict.py",
            "_phase_composite_wrapping.py",
        ):
            _sib_path = _SRC_ROOT / "core" / sib
            if _sib_path.exists():
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
            else:
                # Monolith-split compat: the sibling became a subpackage (``X.py`` -> ``X/``); read __init__ + submodules.
                _sib_pkg = _SRC_ROOT / "core" / sib[:-3]
                if _sib_pkg.is_dir():
                    for _p in sorted(_sib_pkg.glob("*.py")):
                        primary = primary + "\n" + _p.read_text(encoding="utf-8")
    return primary


# ----- contract table -----------------------------------------------------
# Each entry: caller-file (relative to training/), expected propagation marker
# (a substring that MUST appear in the file's source as evidence that the
# ctx-kwarg threading is in place). Source-level guard because we don't have a
# behavioural fixture for every (LTR, weighted, time-series, recurrent)
# combination, but each marker is uniquely tied to the fix shipped in wave 12.
#
# When a future refactor reverts one of these contracts, the matching test
# fails with a message naming the boundary and the kwarg that went missing.

_CONTRACTS = [
    # #1: score_ensemble in the simple per-target ensemble path must thread
    # group_ids + sample_weight.
    ("core/_phase_train_one_target.py", 'group_ids=getattr(ctx, "group_ids", None)', "wave 12 #1: simple-path score_ensemble missing group_ids from ctx"),
    ("core/_phase_train_one_target.py", "sample_weight=_ens_sample_weight", "wave 12 #1: simple-path score_ensemble missing sample_weight from ctx"),
    # #2: run_dummy_baselines signature + caller must pass group_ids.
    (
        "core/_phase_dummy_baselines.py",
        "group_ids=None",  # signature param
        "wave 12 #2: run_dummy_baselines must accept group_ids in signature",
    ),
    ("core/_phase_dummy_baselines.py", "group_ids_train=_gid_train", "wave 12 #2: compute_dummy_baselines call missing group_ids_train"),
    ("core/_phase_train_one_target.py", 'group_ids=getattr(ctx, "group_ids", None)', "wave 12 #2: run_dummy_baselines caller missing group_ids=ctx.group_ids"),
    # #3: recurrent-rerun score_ensemble must thread group_ids + sample_weight.
    ("core/_phase_recurrent.py", '"group_ids": getattr(ctx, "group_ids", None)', "wave 12 #3: recurrent-rerun score_ensemble missing group_ids"),
    ("core/_phase_recurrent.py", '"sample_weight": _sw_for_target', "wave 12 #3: recurrent-rerun score_ensemble missing sample_weight"),
    # #4: RecurrentClassifier.fit must thread sample_weight.
    ("core/_phase_recurrent.py", '_fit_kwargs["sample_weight"] = _sw_for_target', "wave 12 #4: recurrent .fit missing sample_weight propagation"),
    # #5: compute_oof_holdout_predictions must thread time_ordering + sample_weight.
    ("core/_phase_composite_post.py", "time_ordering=_time_ordering", "wave 12 #5: OOF holdout call missing time_ordering=ctx.timestamps[idx]"),
    ("core/_phase_composite_post.py", "sample_weight=_sw_for_oof", "wave 12 #5: OOF holdout call missing sample_weight=ctx.sample_weights"),
]


@pytest.mark.parametrize("rel,marker,why", _CONTRACTS, ids=lambda v: v if isinstance(v, str) else None)
def test_ctx_kwarg_propagates_to_inner_call_site(rel, marker, why):
    """Source-level guard: the marker substring must appear in the named file.

    Justified per [[feedback_behavioral_tests]] carve-out: behavioural tests
    cannot distinguish "model trained" from "model trained with group_ids
    silently dropped". The source-level check matches the contract this wave
    established; if a future refactor removes the marker, the test name +
    failure message points directly at the boundary that lost its kwarg.
    """
    src = _read(rel)
    assert marker in src, (
        f"propagation regression -- {why}.\n"
        f"Expected marker substring NOT found in src/mlframe/training/{rel}:\n"
        f"  marker: {marker!r}\n"
        f"Re-add the ctx kwarg propagation OR update this contract entry "
        f"AND verify the new propagation is equivalent (e.g. a different "
        f"helper name accessing the same ctx slot)."
    )


def test_meta_test_self_check():
    """Sanity: the contract list isn't empty, otherwise the parametrize gives
    false confidence (zero tests run = zero failures)."""
    assert len(_CONTRACTS) >= 10, f"contract table looks short ({len(_CONTRACTS)} entries); wave 12 had 10. Did the constants accidentally get cleared?"
