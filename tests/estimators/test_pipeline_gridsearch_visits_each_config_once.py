"""The pipeline grid search ran every configuration once per assignment ORDER, and threw its results away.

Two defects in one function. The recursion branched on EVERY still-unassigned block at each level instead of
fixing one, so a complete assignment was reached once for each of the `k!` orders of its blocks: `k! * m^k` full
cross-validation runs for `m^k` distinct configurations. Four blocks is 24x redundant CV, and `cv_func` is the
entire cost of the function.

And the recursive call forwarded neither `cv_results` nor `output_dir`. Every leaf built a fresh accumulator and
dumped it to the system temp dir under a path derived only from `title`, so the caller's dict came back empty,
the caller's `output_dir` stayed empty, and each leaf overwrote the previous one. A finished sweep left behind a
file containing exactly one configuration.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mlframe.estimators.pipelines import optimize_pipeline_by_gridsearch

BLOCKS = {"fs": [True, False], "hpt": [True, False], "od": [True, False]}
X = np.zeros((4, 2))
Y = np.zeros(4)


@pytest.fixture
def recorder():
    """A `cv_func` that records the exact configuration it was asked to evaluate."""
    seen = []

    def cv_func(X, Y, title, **constants):
        """Record and return a trivial result payload."""
        seen.append(tuple(sorted(constants.items())))
        return {"results": {"cv_results": {}}, "config": constants}

    cv_func.seen = seen
    return cv_func


def _run(cv_func, tmp_path, blocks=None, **kw):
    """One sweep, dumping into an isolated directory."""
    return optimize_pipeline_by_gridsearch(X, Y, title="t", cv_func=cv_func, possible_pipeline_blocks=blocks if blocks is not None else BLOCKS, output_dir=str(tmp_path), **kw)


class TestEachConfigurationIsEvaluatedOnce:
    """The combinatorial defect."""

    def test_the_leaf_count_is_m_to_the_k(self, recorder, tmp_path):
        """Three binary blocks: 8 configurations, not 3! * 8 = 48."""
        _run(recorder, tmp_path)
        assert len(recorder.seen) == 8, f"{len(recorder.seen)} CV runs for 8 configurations"

    def test_no_configuration_is_evaluated_twice(self, recorder, tmp_path):
        """The duplicates were identical down to the paramset hash, so they were pure waste."""
        assert len(set(recorder.seen)) == len(recorder.seen)

    def test_the_coverage_is_complete(self, recorder, tmp_path):
        """Fixing one block per level must not drop any assignment."""
        _run(recorder, tmp_path)
        expected = {tuple(sorted({"fs": a, "hpt": b, "od": c}.items())) for a in (True, False) for b in (True, False) for c in (True, False)}
        assert set(recorder.seen) == expected

    def test_a_pinned_block_is_still_evaluated(self, recorder, tmp_path):
        """When every remaining block is pinned by a constant the old loop fell through and evaluated nothing."""
        _run(recorder, tmp_path, blocks={"fs": [True, False]}, constants={"fs": True})
        assert recorder.seen == [(("fs", True),)]

    def test_the_growth_is_exponential_not_factorial(self, recorder, tmp_path):
        """Four blocks: 16, where the old form ran 4! * 16 = 384."""
        _run(recorder, tmp_path, blocks={**BLOCKS, "cal": [True, False]})
        assert len(recorder.seen) == 16


class TestTheResultsSurvive:
    """The accumulation defect."""

    def test_the_returned_dict_holds_every_configuration(self, recorder, tmp_path):
        """The docstring promises a summary across the sweep; it used to return None."""
        res = _run(recorder, tmp_path)
        assert len(res["t"]) == 8

    def test_the_callers_own_dict_is_filled(self, recorder, tmp_path):
        """The caller passes an accumulator in; it came back empty."""
        acc: dict = {}
        _run(recorder, tmp_path, cv_results=acc)
        assert len(acc.get("t", {})) == 8

    def test_the_dump_lands_in_the_requested_directory(self, recorder, tmp_path):
        """`output_dir` was dropped at the first recursion, so every dump went to the system temp dir."""
        _run(recorder, tmp_path)
        assert list(pathlib.Path(tmp_path).glob("cv_results-*.dump")), "nothing was written to the requested output_dir"

    def test_the_dump_holds_the_whole_sweep(self, recorder, tmp_path):
        """Leaves shared one title-derived path, so the last leaf's single-entry file was the only artifact."""
        import joblib

        _run(recorder, tmp_path)
        dumped = joblib.load(next(pathlib.Path(tmp_path).glob("cv_results-*.dump")))  # nosec B301 - written by the call above
        assert len(dumped["t"]) == 8, f"the dump holds {len(dumped['t'])} of 8 configurations"

    def test_the_dump_has_its_integrity_sidecar(self, recorder, tmp_path):
        """`replay_cv_results` refuses a dump with no sidecar."""
        _run(recorder, tmp_path)
        assert list(pathlib.Path(tmp_path).glob("cv_results-*.dump.sha256")) or list(pathlib.Path(tmp_path).glob("*.sha256"))
