"""Four handlers changed the answer and said so only at debug level -- or not at all.

  * The marginal-MI re-add probe returns `True` on failure, re-injecting a screening-confirmed but
    statistically untested raw column. The permissive policy is defensible; reverting the gate to its
    pre-fix behaviour on a debug line per candidate is not, because that gate exists precisely because
    coarse-binning plug-in MI upward-biases pure-noise columns.
  * The subsumption discriminator's OUTER handler blanket-excluded every candidate raw from the re-attach
    set on one exception, while the inner per-candidate handler retains on error -- opposite polarities,
    both at debug, with no way to tell from the logs which had fired.
  * `_route_basis` returned the hardcoded string "hermite" on any exception, freezing the WRONG basis into
    the persisted recipe. `transform()` replays that recipe, so train-time and serve-time features diverge
    for that leg, and the code's own comment described the defect before logging it at debug.
  * The GPU CMI kernel's handler logged the literal string "suppressed: %s" -- naming neither the kernel,
    the fallback, nor the shape -- so a cupy import error, a kernel-shape miss, GPU contention and a real
    numeric regression were indistinguishable afterwards. The CPU recomputation keeps the answer correct,
    which is exactly why a real regression would never be noticed.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"

SITES = {
    "_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py": "mrmr_readd_significance_probe_failed",
    "_mi_greedy_cmi_fe.py": "cmi_gpu_kernel_fallback",
}


def _handlers(path: pathlib.Path) -> list:
    """Every `except` clause in the module, as AST nodes."""
    return [n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.ExceptHandler)]


def _literals(path: pathlib.Path) -> set:
    """Every string literal in the module, from its parsed AST.

    Not a substring search over the raw text: that matches a phrase sitting in a COMMENT just as happily as
    one in an emitted message, so "this warning is produced" and "this warning is described in a note above
    the handler" become indistinguishable -- and several of these sites carry exactly such a note.
    """
    return {n.value for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _logs_at(handler: ast.ExceptHandler, names: set) -> bool:
    """True when the handler body calls one of `names` (a logging call at or above warning)."""
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in names:
            return True
        if isinstance(fn, ast.Name) and fn.id in names:
            return True
    return False


class TestThePermissiveReAddIsAudible:
    """A systematic estimator failure must not be one debug line per candidate."""

    def test_the_handler_warns(self):
        """It returns True -- re-adding an untested column -- so the caller has to be able to see it."""
        path = SRC / "_mrmr_fit_impl" / "_friend_graph_and_redundancy" / "_group1.py"
        returning_true = [
            h for h in _handlers(path) if any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value is True for n in ast.walk(h))
        ]
        assert returning_true, "the permissive re-add handler was not found; this test needs updating"
        assert all(_logs_at(h, {"warning", "error", "log_throttle"}) for h in returning_true)

    def test_it_is_throttled(self):
        """One line per fit, not per candidate."""
        key = SITES["_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py"]
        assert key in _literals(SRC / "_mrmr_fit_impl" / "_friend_graph_and_redundancy" / "_group1.py"), f"the throttle key {key!r} is not an emitted literal, so the warning is not throttled per fit"


class TestTheBlanketExclusionIsGone:
    """One exception must not remove an entire candidate set from the support."""

    PATH = SRC / "_mrmr_fit_impl" / "_assign_support_tail.py"

    def test_no_handler_bulk_updates_the_exclusion_set(self):
        """`_rr_excl_names.update(_rr_cand_subsumed)` inside an except is the bulk drop."""
        for h in _handlers(self.PATH):
            for node in ast.walk(h):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                    root = node.func.value
                    assert not (isinstance(root, ast.Name) and root.id == "_rr_excl_names"), "the blanket exclusion is back"

    def test_the_outer_handler_warns(self):
        """Its polarity disagreed with the inner one and both were at debug."""
        literals = _literals(self.PATH)
        assert any("RETAINING all" in s for s in literals), "the outer handler no longer says it retains the whole candidate set"
        assert any(_logs_at(h, {"warning", "error", "log_throttle"}) for h in _handlers(self.PATH)), "no handler in this module logs above debug"


class TestAMislabelledBasisIsAnnounced:
    """Freezing the wrong basis into a replayed recipe is a train/serve skew."""

    PATH = SRC / "_orthogonal_adaptive_arity_fe.py"

    def test_the_route_fallback_warns(self):
        """A per-column event, so there is no spam risk in warning."""
        handlers = [h for h in _handlers(self.PATH) if any(isinstance(n, ast.Return) and getattr(n.value, "value", None) == "hermite" for n in ast.walk(h))]
        assert handlers, "the basis-routing fallback was not found; this test needs updating"
        assert all(_logs_at(h, {"warning", "error"}) for h in handlers)

    def test_the_except_is_narrowed(self):
        """A bug inside `basis_route_by_moments` must propagate, not be laundered into a default."""
        handlers = [h for h in _handlers(self.PATH) if any(isinstance(n, ast.Return) and getattr(n.value, "value", None) == "hermite" for n in ast.walk(h))]
        for h in handlers:
            assert h.type is not None, "the basis-routing fallback catches bare Exception"
            names = [n.id for n in ast.walk(h.type) if isinstance(n, ast.Name)]
            assert "Exception" not in names, f"the basis-routing fallback still catches Exception: {names}"


class TestTheGpuFallbackNamesItself:
    """ "suppressed: %s" identified nothing."""

    PATH = SRC / "_mi_greedy_cmi_fe.py"

    def test_the_uninformative_message_is_gone(self):
        """The exact string, which named neither kernel nor fallback nor shape."""
        assert "suppressed: %s" not in _literals(self.PATH), "the placeholder message is back; it names no cause and no consequence"

    def test_the_fallback_warns_with_a_throttle_key(self):
        """Correctness is preserved by the CPU recomputation, so the cost is the only visible signal."""
        literals = _literals(self.PATH)
        assert "cmi_gpu_kernel_fallback" in literals, "the throttle key is gone, so the fallback warns once per candidate instead of once per fit"
        assert any("recomputing this CMI on the CPU path" in s for s in literals), "the message no longer says what the fallback actually does"

    @pytest.mark.parametrize("rel", sorted(SITES))
    def test_the_throttle_key_is_distinct_per_site(self, rel):
        """Two handlers sharing a key would silence each other."""
        assert SITES[rel] in _literals(SRC / rel), f"{rel} no longer carries the throttle key {SITES[rel]!r}"


def test_all_four_modules_still_import():
    """The narrowed excepts and new logging calls must not break module load."""
    import importlib

    for mod in (
        "mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._group1",
        "mlframe.feature_selection.filters._mrmr_fit_impl._assign_support_tail",
        "mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe",
        "mlframe.feature_selection.filters._mi_greedy_cmi_fe",
    ):
        assert importlib.import_module(mod) is not None
