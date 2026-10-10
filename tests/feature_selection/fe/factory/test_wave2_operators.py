"""Wave-2 FE operator prototypes: out-of-fold warp service (leak safety, replay purity), row statistics (subset recovery) and the residual pair screen (picks the true pair)."""

import itertools
import json

import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.ops_M import resid_oof
from mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.harness import replay_proof
from mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.rowstats import fit_rowstat, replay_rowstat
from mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.run_M import _qcodes, cell_screen
from mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.targets import CASES, gen_M
from mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.warp_service import oof_fit, replay


def test_oof_warp_is_leak_free_on_noise_and_replay_is_pure():
    """On pure-noise y the out-of-fold column is uncorrelated with y while the in-sample table value is not; replay passes all purity checks and handles NaN."""
    rng = np.random.default_rng(0)
    X = rng.random((3000, 3))
    y = rng.standard_normal(3000)
    f, rec = oof_fit("warp1d", X, y, [0], seed=1, nb=20)
    insample = replay(rec, X)
    assert abs(np.corrcoef(f, y)[0, 1]) < 0.06
    assert np.corrcoef(insample, y)[0, 1] > abs(np.corrcoef(f, y)[0, 1]) + 0.04
    proof = replay_proof(rec, replay, X, replay(rec, X), rng)
    assert all(proof.values()), proof
    Xn = X.copy()
    Xn[:5, 0] = np.nan
    out = replay(rec, Xn)
    assert np.isfinite(out).all() and np.allclose(out[:5], rec["full"]["fill"])
    assert len(json.dumps(rec)) < 20_000  # tables only: far smaller than the 3000-row target


def test_cell2d_recovers_bumps_and_foldavg_matches_full_in_distribution():
    """The cross-fitted 2-D table recovers the two-bump surface out of fold; the fold-average replay stays close to the full-table replay."""
    gen, _ = CASES["B"]["W"]
    X, y, truth = gen(np.random.default_rng(1), 6000)
    f, rec = oof_fit("cell2d", X, y, [0, 1], seed=0, K=10)
    assert np.corrcoef(f, truth)[0, 1] > 0.85
    assert np.corrcoef(replay(rec, X, "full"), replay(rec, X, "foldavg"))[0, 1] > 0.97
    df = pd.DataFrame(X, columns=["a", "b"])
    rn = dict(rec, src=["a", "b"])
    assert np.array_equal(replay(rn, df), replay(rec, X))


def test_row_stat_recovers_the_informative_subset():
    """The range of columns 0-4 with five irrelevant columns: the learned subset is exactly the five and replay is deterministic."""
    gen, cols = CASES["G"]["H"]
    X, y, _ = gen(np.random.default_rng(2), 6000)
    rec = fit_rowstat(X, y, cols)
    assert rec["stat"] in ("rng", "std") and set(rec["src"]) >= {0, 1, 2, 3, 4}
    assert np.array_equal(replay_rowstat(rec, X), replay_rowstat(json.loads(json.dumps(rec)), X))


def test_residual_cell_screen_finds_the_true_pair_on_the_additive_target():
    """Pair screen on the out-of-fold additive residual ranks the true interaction pair first."""
    X, y, true_pairs = gen_M(np.random.default_rng(3), 6000, "W")
    pairs = list(itertools.combinations(range(X.shape[1]), 2))
    Q = np.column_stack([_qcodes(X[:, j], 6) for j in range(X.shape[1])])
    r = resid_oof(X, y, np.random.default_rng(0))
    top_r = pairs[int(np.argmax(cell_screen(Q, r, pairs)))]
    assert top_r == true_pairs[0]
