"""Selection rows and report rows never overlap, verdicts read test, and train-time charts read train.

The honest holdout both dropped specs and reported the survivors' gain on the same rows (the winner's curse it existed to
remove), the composite-vs-raw verdict was decided on the val split discovery selected on, and the discovery chart drew test
rows into a train-time diagnostic. The consumers note their reads in ``composite._row_roles``; these checks read the ledger.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import _row_roles
from mlframe.training.composite._row_roles import note_rows


@contextlib.contextmanager
def recording():
    """Enable the ledger for the block and yield the reads it records (cleared on entry)."""
    prev = _row_roles._FORCED
    _row_roles._FORCED = True
    _row_roles._LOG.clear()
    try:
        yield _row_roles._LOG
    finally:
        _row_roles._FORCED = prev


def _reads(log, row_set, role):
    """Every recorded read of ``row_set`` in ``role``."""
    return [r for r in log if r.row_set == row_set and r.role == role]


@pytest.fixture(scope="module")
def discovery_reads():
    """The ledger of one grouped discovery fit with an honest holdout large enough to halve."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    n = 3000
    g = np.repeat(np.arange(30), n // 30)
    base = rng.uniform(1.0, 10.0, n) + 0.3 * g
    df = pd.DataFrame({"b": base, "x": rng.normal(size=n), "g": g, "y": 2.0 * base + rng.normal(0.0, 0.5, n)})
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, base_candidates=["b"], transforms=["linear_residual", "diff"],
                                         eps_mi_gain=-10.0, group_column="g", honest_holdout_frac=0.3)
    with recording() as log:
        CompositeTargetDiscovery(cfg).fit(df, "y", ["b", "x"], np.arange(n))
        reads = list(log)
    return reads


def test_the_honest_holdout_never_selects_on_the_rows_it_reports(discovery_reads):
    """Every row a gate or ranking read from the honest holdout is absent from the rows the reported gain came from."""
    select = _reads(discovery_reads, "honest_holdout", "select")
    report = _reads(discovery_reads, "honest_holdout", "report")
    assert select and report, f"the fixture must exercise both halves; recorded {[(r.consumer, r.role) for r in discovery_reads]}"
    selected = set(np.concatenate([r.rows for r in select if r.rows is not None]).tolist())
    reported = set(np.concatenate([r.rows for r in report if r.rows is not None]).tolist())
    assert selected and reported and not (selected & reported), f"{len(selected & reported)} honest-holdout rows both selected and reported"


def test_verdicts_read_only_the_test_split():
    """The composite-vs-raw verdict notes a test read, and no verdict consumer reads another split."""
    from mlframe.training.core._phase_composite_post_summary import format_composite_vs_raw_block

    with recording() as log:
        format_composite_vs_raw_block(models={}, metadata={}, best_metrics={}, composite_to_raw={})
        verdicts = [r for r in log if r.role == "verdict"]
    assert verdicts and all(r.row_set == "test" for r in verdicts), [(r.consumer, r.row_set) for r in verdicts]


def test_discovery_charts_plot_only_train_rows(tmp_path):
    """The discovery chart notes a train read covering exactly the train rows it was given."""
    from mlframe.training.core._phase_composite_discovery_helpers import _render_composite_discovery_diagnostics

    y = np.arange(50, dtype=float)
    with recording() as log:
        _render_composite_discovery_diagnostics(data_dir=tmp_path, raw_target_name="y", y_full=y, t_by_spec={}, specs_export=[], train_idx=np.arange(30))
        plots = [r for r in log if r.role == "plot"]
    assert plots and all(r.row_set == "train" for r in plots)
    assert np.array_equal(plots[0].rows, np.arange(30))


def test_the_ledger_is_off_unless_enabled_and_rejects_unknown_roles(monkeypatch):
    """Outside :func:`recording` (and without the env var) nothing is recorded; a misspelt role fails loudly."""
    monkeypatch.delenv("MLFRAME_ROW_ROLE_LEDGER", raising=False)
    _row_roles._LOG.clear()
    note_rows("test", "verdict", "outside")
    assert not _row_roles._LOG
    with recording(), pytest.raises(ValueError):
        note_rows("test", "verdit", "typo")


def _fit_report_collisions(log) -> dict[str, int]:
    """Per decision (consumer), how many rows of one row set it both fitted on and reported on."""
    out = {}
    for c in {r.consumer for r in log}:
        for s in {r.row_set for r in log if r.consumer == c}:
            fit = [r.rows for r in log if r.consumer == c and r.row_set == s and r.role == "fit" and r.rows is not None]
            rep = [r.rows for r in log if r.consumer == c and r.row_set == s and r.role == "report" and r.rows is not None]
            if fit and rep:
                out[c] = int(np.intersect1d(np.concatenate(fit), np.concatenate(rep)).size)
    return out


def _stack_gate_log(n: int):
    """The ledger of the xt-ensemble fallback gate scoring an nnls stack on an ``n``-row OOF matrix."""
    from mlframe.training.composite import CompositeCrossTargetEnsemble
    from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import gate_stack_rmse

    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    P = np.column_stack([y + rng.normal(0.0, s, n) for s in (0.5, 0.8, 1.2)])
    comps = [object(), object(), object()]
    with recording() as log:
        ens = CompositeCrossTargetEnsemble.from_nnls_stack(component_models=comps, component_names=["a", "b", "c"], component_predictions=P, y_train=y)
        rmse = gate_stack_rmse(CompositeCrossTargetEnsemble, "nnls_stack", comps, ["a", "b", "c"], P, y, P @ np.asarray(ens.weights))
        reads = list(log)
    assert np.isfinite(rmse)
    return reads, n


def test_the_xt_stack_gate_never_scores_weights_on_their_own_fit_rows():
    """Each fold of the stack gate reports on OOF rows its weights were not fitted on, and every OOF row is reported once (EST-05)."""
    reads, n = _stack_gate_log(400)
    collisions = _fit_report_collisions(reads)
    assert collisions and set(collisions.values()) == {0}, collisions
    reported = np.concatenate([r.rows for r in reads if r.role == "report"])
    assert np.array_equal(np.sort(reported), np.arange(n))


def test_the_canary_sees_the_in_sample_stack_gate():
    """Too few rows to cross-fit: the gate falls back to scoring the weights on their own fit rows, and the contract sees it."""
    reads, n = _stack_gate_log(12)
    assert _fit_report_collisions(reads) == {"xt_stack_gate[in-sample]": n}
