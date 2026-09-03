"""A dataset build attributed to ``sklearn.model_selection._validation:1319`` names the machinery, not the caller.

A production run built five 1.96M-row LightGBM datasets -- about three minutes -- on a fit whose model list was
``['cb']``. Every one of them was logged against sklearn's CV internals, so the log could not say which mlframe
module asked for them, and the only way to find out was to go get a stack trace by hand.

Two changes: the call site now names the owning ``mlframe.*`` frame alongside the mechanism, and every build is
counted per owner into a rollup printed at the end of the suite -- including the builds whose individual line is
demoted to DEBUG inside internal fit loops, which is exactly where unattributed time hides.
"""

from __future__ import annotations

import pytest

from mlframe.training._dataset_build_stats import (
    dataset_build_snapshot,
    format_dataset_build_stats,
    record_dataset_build,
    reset_dataset_build_stats,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is process-wide, so every test starts and leaves it empty."""
    reset_dataset_build_stats()
    yield
    reset_dataset_build_stats()


class TestTheRollup:
    """What the operator reads at the end of a run."""

    def test_builds_by_the_same_owner_are_summed(self):
        """Five scattered lines become one row with a total -- the number worth acting on."""
        for _ in range(5):
            record_dataset_build("lightgbm.Dataset", "mlframe.training.composite.dual_direction:162", 1_960_963, 12.0)
        row = dataset_build_snapshot()[0]
        assert row["count"] == 5
        assert row["rows_total"] == 5 * 1_960_963
        assert row["seconds"] == pytest.approx(60.0)

    def test_owners_are_kept_apart(self):
        """Attribution is the entire point; merging owners would undo it."""
        record_dataset_build("lightgbm.Dataset", "a:1", 100, 1.0)
        record_dataset_build("lightgbm.Dataset", "b:2", 100, 1.0)
        assert len(dataset_build_snapshot()) == 2

    def test_the_same_owner_with_two_libraries_stays_separate(self):
        """A module that builds both a Pool and a Dataset has two different costs to look at."""
        record_dataset_build("catboost.Pool", "a:1", 100, 1.0)
        record_dataset_build("lightgbm.Dataset", "a:1", 100, 1.0)
        assert len(dataset_build_snapshot()) == 2

    def test_rows_drive_the_order(self):
        """Row count is what makes a build expensive, so the biggest spender leads."""
        record_dataset_build("lightgbm.Dataset", "small:1", 10, 5.0)
        record_dataset_build("lightgbm.Dataset", "big:2", 1_000_000, 0.1)
        assert dataset_build_snapshot()[0]["owner"].startswith("big:2")

    def test_the_largest_single_build_is_kept(self):
        """A total of 2M rows means something different as one build or as two thousand."""
        record_dataset_build("lightgbm.Dataset", "a:1", 1_000_000, 1.0)
        record_dataset_build("lightgbm.Dataset", "a:1", 5, 1.0)
        assert dataset_build_snapshot()[0]["rows_max"] == 1_000_000

    def test_reset_clears_everything(self):
        """One suite's table must describe one suite's builds."""
        record_dataset_build("lightgbm.Dataset", "a:1", 10, 1.0)
        reset_dataset_build_stats()
        assert dataset_build_snapshot() == []


class TestTheTable:
    """Readable straight out of the log."""

    def test_empty_says_so(self):
        """A header over nothing reads as "no builds were expensive"."""
        assert "none recorded" in format_dataset_build_stats(dataset_build_snapshot())

    def test_it_names_the_owner_and_the_totals(self):
        """The two facts that make the row actionable: who, and how much."""
        record_dataset_build("lightgbm.Dataset", "mlframe.training.composite.dual_direction:162", 1_960_963, 12.0)
        text = format_dataset_build_stats(dataset_build_snapshot())
        assert "dual_direction:162" in text
        assert "1,960,963" in text


class TestTheCallSiteNamesTheOwner:
    """The stack walk, exercised over a synthetic chain shaped like the production one."""

    def _callsite_through(self, *module_names: str) -> str:
        """What the walk reports when the stack is, outermost first, ``module_names``.

        Each frame is a function exec'd into globals carrying the module name we want, which is exactly what
        the walk reads (``frame.f_globals["__name__"]``).
        """
        from mlframe.training._dataset_build_stats import infer_build_callsite

        holder: dict = {}

        def _leaf():
            """Innermost frame: stands in for the library constructor asking where it was called from."""
            # skip_frames=2 lands the walk on the OUTERMOST-inserted synthetic frame rather than on this
            # helper, mirroring production where the innermost non-library frame is the dispatcher itself.
            holder["site"] = infer_build_callsite(skip_frames=2)

        fn = _leaf
        for name in reversed(module_names):
            g: dict = {"__name__": name}
            # exec is the point: the walk reads ``frame.f_globals["__name__"]``, and this is how a frame
            # carrying a chosen module name gets created. The compiled source is a fixed literal.
            exec(compile("def _f(_next):" + chr(10) + "    return _next()" + chr(10), "<" + name + ">", "exec"), g)  # nosec B102
            fn = (lambda _inner, _g=g: (lambda: _g["_f"](_inner)))(fn)
        fn()
        return holder["site"]

    def test_the_mlframe_owner_is_named_behind_a_sklearn_dispatcher(self):
        """The production shape: mlframe calls sklearn, sklearn builds the dataset."""
        site = self._callsite_through("mlframe.training.composite.dual_direction", "sklearn.model_selection._validation")
        assert "mlframe.training.composite.dual_direction" in site

    def test_the_mechanism_is_still_shown(self):
        """Knowing it came through sklearn CV is what explains WHY there are five of them."""
        site = self._callsite_through("mlframe.training.composite.dual_direction", "sklearn.model_selection._validation")
        assert "sklearn.model_selection._validation" in site

    def test_a_direct_mlframe_call_is_reported_once(self):
        """With no dispatcher in between there is nothing to disambiguate."""
        site = self._callsite_through("mlframe.training.trainer")
        assert "<-" not in site
        assert "mlframe.training.trainer" in site

    def test_a_deep_dispatch_chain_still_finds_the_owner(self):
        """The old walk gave up after 8 frames, which is shallower than sklearn CV plus joblib."""
        chain = ["mlframe.training.trainer"] + ["joblib.parallel"] * 6 + ["sklearn.model_selection._validation"]
        assert "mlframe.training.trainer" in self._callsite_through(*chain)

    def test_a_stack_with_no_mlframe_frame_still_reports_something(self):
        """A third-party caller is legitimate; the line must not go blank."""
        site = self._callsite_through("some_user_notebook")
        assert site not in ("", "?")
