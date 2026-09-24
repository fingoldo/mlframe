"""Wrappers under src/mlframe/training forward the options of what they wrap (py_ci_shared.kwarg_forwarding).

The discovery variants once dropped ``time_ordering`` / ``val_df`` / ``val_y`` on their way to ``fit`` (DSC-14), and the
per-group delegate ran without the rerank groups and hint strengths the suite injects (DSC-27). On a tree with those shapes
restored the scanners report exactly them; here they must report nothing beyond the reasoned allow table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_D = "src/mlframe/training/composite/discovery/"
_T = "src/mlframe/training/"

# Variants whose base lives in another module (the discovery methods are bound onto the class from their own modules).
DELEGATES = {
    _D + "_stacked.py::fit_stacked": _D + "_fit.py::fit",
    _D + "_stacked.py::fit_stacked_on_residual": _D + "_fit.py::fit",
    _D + "_stability_check.py::fit_with_stability_check": _D + "_fit.py::fit",
}

# Deliberate omissions, keyed as the scanner keys them, each with the reason.
ALLOWED = {
    f"available_not_passed:{_D}_stacked.py::fit_stacked->{_D}_fit.py::fit:val_df":
        "pass 2 adds OOF columns to the train frame only; the val frame lacks them, so pass 2 cannot be scored on it",
    f"available_not_passed:{_D}_stacked.py::fit_stacked->{_D}_fit.py::fit:val_y": "goes with val_df above",
    f"available_not_passed:{_D}_stacked.py::fit_stacked_on_residual->{_D}_fit.py::fit:val_df":
        "pass 2 fits the residual target, so the raw-target val split does not apply",
    f"available_not_passed:{_D}_stacked.py::fit_stacked_on_residual->{_D}_fit.py::fit:val_y": "goes with val_df above",
    f"available_not_passed:{_T}feature_drift_report.py::_compute_drift_invariant->{_T}feature_drift_report.py::compute_categorical_drift_psi:feature_names":
        "the call sits in the feature_names-is-None branch; the restricted case passes the categorical subset above it",
    f"available_not_passed:{_T}pipeline/_entity_time_composite_fe.py::replay_entity_time_composite_fe->{_T}pipeline/_entity_time_composite_fe.py::apply_entity_time_composite_fe:metadata":
        "a predict-time replay reads fit-time state from metadata; passing it would let the replay overwrite that state",
    f"available_not_passed:{_T}pipeline/_event_proximity_decay_composite_fe.py::replay_event_proximity_decay_composite_fe->{_T}pipeline/_event_proximity_decay_composite_fe.py::apply_event_proximity_decay_composite_fe:metadata":
        "same predict-time replay contract as the entity-time replay",
    f"available_not_passed:{_T}pipeline/_ma_crossover_composite_fe.py::replay_ma_crossover_composite_fe->{_T}pipeline/_ma_crossover_composite_fe.py::apply_ma_crossover_composite_fe:metadata":
        "same predict-time replay contract as the entity-time replay",
    f"available_not_passed:{_T}pipeline/_pipeline_helpers.py::_prepare_test_split->{_T}pipeline/_pipeline_helpers.py::_passthrough_cols_fit_transform:target":
        "the test split is only transformed (fit=False); the target has no role in a transform",
}


def _files():
    """Every source file under src/mlframe/training except the benchmarks."""
    return sorted(p for p in (REPO_ROOT / "src" / "mlframe" / "training").rglob("*.py") if "_benchmarks" not in p.parts)


def test_wrappers_forward_what_they_wrap():
    """No dropped variant option, no in-scope argument left out and no delegate state lost, beyond the allow table."""
    kwf = pytest.importorskip("py_ci_shared.kwarg_forwarding")
    files = _files()
    found = kwf.find_dropped_variant_params(files, REPO_ROOT, delegates=DELEGATES)
    found += kwf.find_available_but_not_passed(files, REPO_ROOT, delegates=DELEGATES)
    found += kwf.find_delegate_state_loss(files, REPO_ROOT)
    unexpected = [repr(f) for f in found if f.key not in ALLOWED]
    assert not unexpected, "forward the argument, or allow it here with the reason: " + "; ".join(unexpected)
    stale = sorted(set(ALLOWED) - {f.key for f in found})
    assert not stale, f"allow-table entries that no longer occur: {stale}"
    assert all(str(r).strip() for r in ALLOWED.values())
