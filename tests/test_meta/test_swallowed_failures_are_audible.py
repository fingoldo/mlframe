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
import importlib
import logging
import pathlib

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"

SITES = {
    "_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py": "mrmr_readd_significance_probe_failed",
    "_mi_greedy_cmi_fe.py": "cmi_gpu_kernel_fallback",
}


def _handlers(path: pathlib.Path) -> list:
    """Every `except` clause in the module, as AST nodes."""
    return [n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.ExceptHandler)]


def _emits(path: pathlib.Path, phrase: str) -> bool:
    """True when ``phrase`` appears in a string literal the module can actually emit.

    A predicate rather than a set the caller then tests membership on: the assertion is about what the module
    SAYS, and phrasing it as `phrase in <something derived from read_text>` is the shape that cannot tell an
    emitted message from a comment -- which is the very confusion these helpers exist to remove.
    """
    return any(phrase in lit for lit in _literals(path))


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
        returning_true = [h for h in _handlers(path) if any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value is True for n in ast.walk(h))]
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
        handlers = _handlers(self.PATH)
        assert handlers, "no except handlers found in the support-assignment tail; this test needs updating"
        bulk_updates = [
            node
            for h in handlers
            for node in ast.walk(h)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_rr_excl_names"
        ]
        assert not bulk_updates, "the blanket exclusion is back"

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

    @staticmethod
    def _run_recipes(monkeypatch, frame_columns):
        """Build recipes for one winning pair column whose legs are ``a`` and ``b``, with ``frame_columns`` present in the source frame."""
        import pandas as pd

        from mlframe.feature_selection.filters import _orthogonal_adaptive_arity_fe as mod

        X = pd.DataFrame({c: np.linspace(0.0, 1.0, 12) ** 2 for c in frame_columns})
        X_aug = X.assign(**{"a*b__He2_T1": 0.0})
        monkeypatch.setattr(mod, "hybrid_orth_mi_adaptive_arity_fe", lambda *args, **kwargs: (X_aug, pd.DataFrame(), pd.DataFrame()))
        return mod, mod.hybrid_orth_mi_adaptive_arity_fe_with_recipes(X, np.arange(12) % 2, basis="auto")

    def test_the_except_is_narrowed(self, monkeypatch, caplog):
        """A column that cannot be read is announced and frozen as 'hermite'; a bug inside `basis_route_by_moments` propagates instead of becoming a default."""
        with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe"):
            _, result = self._run_recipes(monkeypatch, ["z"])
        recipes = result[-1]
        assert len(recipes) == 1
        routed = [m for m in caplog.messages if m.startswith("_route_basis: could not route column")]
        assert len(routed) == 2, f"expected one warning per unroutable leg, got {routed}"
        assert "'a' (KeyError" in routed[0] and "freezing 'hermite' into the recipe" in routed[0]

        def router_bug(values):
            """A defect inside the router itself."""
            raise RuntimeError("router bug")

        from mlframe.feature_selection.filters import _orthogonal_adaptive_arity_fe as mod

        monkeypatch.setattr(mod, "basis_route_by_moments", router_bug)
        with pytest.raises(RuntimeError, match="router bug"):
            self._run_recipes(monkeypatch, ["a", "b"])


def _failing_gpu_cmi_run(monkeypatch, caplog, repeats):
    """Call the CMI wrapper ``repeats`` times with the GPU path forced on and raising; return ``(results, cpu_reference, warning messages)``."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as mod
    from mlframe.utils.log_throttle import reset_throttle_counts

    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 200)
    y = (x + rng.integers(0, 2, 200)) % 4

    def _boom(*args, **kwargs):
        """Stand in for the cupy kernel and fail."""
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(mod, "_cmi_gpu_enabled", lambda **kwargs: False)
    cpu_reference = mod._cmi_from_binned(x, y, None)
    monkeypatch.setattr(mod, "_cmi_gpu_enabled", lambda **kwargs: True)
    monkeypatch.setattr(mod, "_cmi_from_binned_cupy", _boom)
    reset_throttle_counts("cmi_gpu_kernel_fallback")
    caplog.set_level(logging.WARNING, logger=mod.logger.name)
    results = [mod._cmi_from_binned(x, y, None) for _ in range(repeats)]
    return results, cpu_reference, [r.getMessage() for r in caplog.records]


class TestTheGpuFallbackNamesItself:
    """ "suppressed: %s" identified nothing."""

    def test_the_fallback_names_the_kernel_the_cause_and_the_consequence(self, monkeypatch, caplog):
        """A failing GPU kernel still returns the CPU answer, and the one warning says which kernel failed, why, at what shape and what happened next."""
        results, cpu_reference, messages = _failing_gpu_cmi_run(monkeypatch, caplog, repeats=1)
        assert results == [cpu_reference]
        assert len(messages) == 1
        assert "_cmi_from_binned_cupy failed (RuntimeError: kernel exploded) at n=200" in messages[0]
        assert "recomputing this CMI on the CPU path" in messages[0]
        assert not any(m.startswith("suppressed:") for m in messages), "the placeholder message is back; it names no cause and no consequence"

    def test_the_fallback_warns_with_a_throttle_key(self, monkeypatch, caplog):
        """Correctness is preserved by the CPU recomputation, so the cost is the only visible signal: it is throttled per key, not per candidate."""
        throttle = importlib.import_module("mlframe.utils.log_throttle")

        results, cpu_reference, messages = _failing_gpu_cmi_run(monkeypatch, caplog, repeats=12)
        assert results == [cpu_reference] * 12
        assert throttle._counts["cmi_gpu_kernel_fallback"] == 12
        failures = [m for m in messages if "_cmi_from_binned_cupy failed" in m]
        assert len(failures) == 5, f"expected the warning throttled to 5 occurrences, got {len(failures)}"
        assert len(messages) == 6
        assert "cmi_gpu_kernel_fallback: further occurrences suppressed" in messages[-1]

    def test_the_throttle_key_is_distinct_per_site_and_used_by_the_cmi_site(self, monkeypatch, caplog):
        """The registry keys differ between the two sites, and the CMI fallback really counts under its own key."""
        throttle = importlib.import_module("mlframe.utils.log_throttle")

        assert len(set(SITES.values())) == len(SITES), "two handlers share a throttle key and would silence each other"
        _failing_gpu_cmi_run(monkeypatch, caplog, repeats=1)
        assert throttle._counts[SITES["_mi_greedy_cmi_fe.py"]] == 1

    @pytest.mark.parametrize("rel", ["_mrmr_fit_impl/_friend_graph_and_redundancy/_group1.py"])
    def test_the_throttle_key_is_distinct_per_site(self, rel):
        """Two handlers sharing a key would silence each other."""
        assert SITES[rel] in _literals(SRC / rel), f"{rel} no longer carries the throttle key {SITES[rel]!r}"


def test_all_four_modules_still_import():
    """The narrowed excepts and new logging calls must not break module load."""
    for mod in (
        "mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._group1",
        "mlframe.feature_selection.filters._mrmr_fit_impl._assign_support_tail",
        "mlframe.feature_selection.filters._orthogonal_adaptive_arity_fe",
        "mlframe.feature_selection.filters._mi_greedy_cmi_fe",
    ):
        assert importlib.import_module(mod) is not None
