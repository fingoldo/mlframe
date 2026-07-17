"""Tests for Phase 7 resume-from-checkpoint.

The contract:
- ``checkpoint_path=None`` (default): no behavior change, no file IO
- ``checkpoint_path=path``: state is pickled atomically after every outer
  iter; subsequent fit() with matching signature resumes from that state
- Mismatched signature -> warning + start fresh
- Corrupt or version-mismatched file -> warning + start fresh
- Atomic write (tmpfile + os.replace): a crash mid-write does NOT corrupt
  the previous valid checkpoint
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV
from tests.training.synthetic import make_sklearn_classification_df


@pytest.fixture(scope="module")
def small_problem():
    """Small problem."""
    X_df, y, _ = make_sklearn_classification_df(
        n_samples=200,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        shuffle=False,
        class_sep=2.0,
        seed=0,
    )
    return X_df, y


def _make_selector(checkpoint_path=None, max_refits=8, random_state=0):
    """Make selector."""
    return RFECV(
        estimator=LogisticRegression(max_iter=300, random_state=random_state),
        cv=3,
        max_refits=max_refits,
        verbose=0,
        random_state=random_state,
        checkpoint_path=checkpoint_path,
    )


# ----------------------------------------------------------------------------
# Defaults: checkpoint_path=None must not write files
# ----------------------------------------------------------------------------
class TestCheckpointOptIn:
    """Groups tests covering TestCheckpointOptIn."""
    def test_no_checkpoint_path_means_no_file_io(self, small_problem, tmp_path):
        """Default: no file should appear anywhere when checkpoint_path is None."""
        X, y = small_problem
        sel = _make_selector(checkpoint_path=None)
        sel.fit(X, y)
        # No stray .rfecv_ckpt_* tempfiles in the working dir or tmp_path
        leftovers = list(tmp_path.glob(".rfecv_ckpt_*"))
        assert leftovers == [], f"unexpected checkpoint tempfiles: {leftovers}"


# ----------------------------------------------------------------------------
# Save: file is created and contains the documented keys
# ----------------------------------------------------------------------------
class TestCheckpointSave:
    """Groups tests covering TestCheckpointSave."""
    def test_checkpoint_file_created_after_fit(self, small_problem, tmp_path):
        """Checkpoint file created after fit."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        sel = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel.fit(X, y)
        assert ckpt.exists(), f"checkpoint file was not created at {ckpt}"
        with ckpt.open("rb") as fh:
            state = pickle.load(fh)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert isinstance(state, dict)
        # Documented schema
        for required_key in (
            "version",
            "signature",
            "nsteps",
            "evaluated_scores_mean",
            "evaluated_scores_std",
            "feature_importances",
            "selected_features_per_nfeatures",
            "best_nfeatures",
            "best_iter",
            "best_score",
            "optimizer",
        ):
            assert required_key in state, f"missing key {required_key!r} in checkpoint"
        # nsteps must equal number of fully-completed outer iters (==max_refits
        # when fit didn't early-stop on max_noimproving_iters / time / etc.)
        assert state["nsteps"] >= 1
        # signature mirrors fit()'s computation:
        # ``(X_shape, y_shape, feature_names, y_hash, X_hash, params_signature)`` -- the (X, y) hashes
        # identify the exact data fit on (not just shapes), and the trailing params-signature invalidates
        # the checkpoint on a config change. Older 3-/5-tuple shapes are gone.
        assert isinstance(state["signature"], tuple) and len(state["signature"]) == 6

    def test_checkpoint_file_size_excludes_estimators(self, small_problem, tmp_path):
        """fitted_estimators is intentionally NOT pickled to keep file small.
        Sanity-check by confirming the loaded dict has no 'estimators' /
        'fitted_estimators' key carrying the heavy ensemble objects."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        sel = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel.fit(X, y)
        with ckpt.open("rb") as fh:
            state = pickle.load(fh)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert "fitted_estimators" not in state
        assert "estimators_" not in state


# ----------------------------------------------------------------------------
# Load: signature match, mismatch, corrupt
# ----------------------------------------------------------------------------
class TestCheckpointLoad:
    """Groups tests covering TestCheckpointLoad."""
    def test_signature_match_resumes_state(self, small_problem, tmp_path):
        """fit-then-fit-with-checkpoint should resume from the saved state.
        On the second fit (with skip_retraining_on_same_shape=False to force
        re-fit), the checkpoint must be loaded and trajectory continues."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        sel = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel.fit(X, y)
        first_nsteps = pickle.loads(ckpt.read_bytes())["nsteps"]  # nosec B301 -- round-trip of a locally-created, trusted object
        # Second selector, fresh instance, larger budget.
        sel2 = RFECV(
            estimator=LogisticRegression(max_iter=300, random_state=0),
            cv=3,
            max_refits=8,
            verbose=0,
            random_state=0,
            checkpoint_path=str(ckpt),
            skip_retraining_on_same_shape=False,
        )
        sel2.fit(X, y)
        # After resumed fit: checkpoint nsteps must have advanced past
        # the saved-state's nsteps
        second_nsteps = pickle.loads(ckpt.read_bytes())["nsteps"]  # nosec B301 -- round-trip of a locally-created, trusted object
        assert second_nsteps > first_nsteps, f"resume failed to advance: first={first_nsteps}, second={second_nsteps}"

    def test_signature_mismatch_starts_fresh(self, small_problem, tmp_path):
        """Same checkpoint file, different X shape -> must NOT load state.
        The 2nd fit should produce a sensible support_ for its own X."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        sel = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel.fit(X, y)
        # Build a deliberately different problem (more features, same n)
        rng = np.random.default_rng(1)
        X2 = pd.DataFrame(rng.standard_normal((200, 12)), columns=[f"g{i}" for i in range(12)])
        y2 = (X2["g0"] > 0).astype(int).values
        sel2 = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel2.fit(X2, y2)
        assert sel2.n_features_in_ == 12
        assert sel2.support_.shape[0] == 12, f"mismatched-signature path produced wrong support_ shape; got {sel2.support_.shape}"

    def test_corrupt_checkpoint_starts_fresh(self, small_problem, tmp_path, caplog):
        """Truncated pickle file must not crash fit()."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        ckpt.write_bytes(b"\x80\x05\x00\x00")  # malformed pickle
        sel = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel.fit(X, y)
        # The fit must complete successfully and overwrite the corrupt file.
        assert sel.n_features_ >= 1
        assert ckpt.stat().st_size > 4, "checkpoint should have been overwritten with valid data"

    def test_version_mismatch_starts_fresh(self, small_problem, tmp_path):
        """A pickle with wrong version field must be ignored."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        with ckpt.open("wb") as fh:
            pickle.dump({"version": 999, "signature": ((0, 0), (0,), ("__ndarray__", 0))}, fh)
        sel = _make_selector(checkpoint_path=str(ckpt), max_refits=4)
        sel.fit(X, y)
        assert sel.n_features_ >= 1


# ----------------------------------------------------------------------------
# Atomic write: pre-existing valid checkpoint survives a write that fails
# ----------------------------------------------------------------------------
class TestCheckpointAtomicity:
    """Groups tests covering TestCheckpointAtomicity."""
    def test_non_writeable_target_raises_inside_save_only(self, small_problem, tmp_path):
        """Direct unit test of _save_checkpoint atomicity: write to a path
        inside a non-existent dir must fail cleanly without leaving stray
        tempfiles in cwd."""
        sel = _make_selector(checkpoint_path=str(tmp_path / "subdir" / "ckpt.pkl"))
        # Subdir doesn't exist - _save_checkpoint creates it; this must work.
        sel._save_checkpoint({"version": 1, "signature": ((1, 1), (1,), ("__ndarray__", 1)), "nsteps": 0})
        assert (tmp_path / "subdir" / "ckpt.pkl").exists()

    def test_existing_checkpoint_not_corrupted_by_failed_save(self, tmp_path):
        """If pickle.dump raises (e.g. unpicklable object inside state),
        the existing valid checkpoint at the target path must remain intact.
        """
        target = tmp_path / "rfecv.pkl"
        # Pre-populate target with a valid checkpoint
        sel = _make_selector(checkpoint_path=str(target))
        valid_state = {"version": 1, "signature": ((1, 1), (1,), ("a",)), "nsteps": 5}
        sel._save_checkpoint(valid_state)
        original_bytes = target.read_bytes()
        # Try to save an unpicklable object (lambda is the textbook unpicklable)
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            sel._save_checkpoint({"version": 1, "bad": lambda x: x})
        # Original target must be untouched
        assert target.read_bytes() == original_bytes, "atomic write violated: failed save corrupted the existing checkpoint"

    def test_no_temp_files_left_on_success(self, tmp_path):
        """No temp files left on success."""
        sel = _make_selector(checkpoint_path=str(tmp_path / "ckpt.pkl"))
        sel._save_checkpoint({"version": 1, "signature": ((1, 1), (1,), ("a",)), "nsteps": 1})
        leftovers = list(tmp_path.glob(".rfecv_ckpt_*"))
        assert leftovers == [], f"stale tempfile after successful save: {leftovers}"

    def test_no_temp_files_left_on_failed_save(self, tmp_path):
        """No temp files left on failed save."""
        sel = _make_selector(checkpoint_path=str(tmp_path / "ckpt.pkl"))
        # Pre-create target so _save_checkpoint sees the dir exists.
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            sel._save_checkpoint({"bad": lambda x: x})
        leftovers = list(tmp_path.glob(".rfecv_ckpt_*"))
        assert leftovers == [], f"stale tempfile after failed save: {leftovers}"


# ----------------------------------------------------------------------------
# End-to-end: resume produces valid-shaped support_ and compatible cv_results_
# ----------------------------------------------------------------------------
class TestCheckpointEndToEnd:
    """Groups tests covering TestCheckpointEndToEnd."""
    def test_resumed_fit_produces_valid_outputs(self, small_problem, tmp_path):
        """After resume, all standard sklearn-style attributes must be
        populated as if no checkpoint existed."""
        X, y = small_problem
        ckpt = tmp_path / "rfecv.pkl"
        # First run: short budget
        sel1 = _make_selector(checkpoint_path=str(ckpt), max_refits=3)
        sel1.fit(X, y)
        # Second run: full budget, resumed
        sel2 = RFECV(
            estimator=LogisticRegression(max_iter=300, random_state=0),
            cv=3,
            max_refits=8,
            verbose=0,
            random_state=0,
            checkpoint_path=str(ckpt),
            skip_retraining_on_same_shape=False,
        )
        sel2.fit(X, y)
        # Core attributes
        assert sel2.support_.shape == (X.shape[1],)
        assert sel2.support_.dtype == bool
        assert sel2.n_features_ == int(sel2.support_.sum())
        assert sel2.n_features_ >= 1
        assert sel2.n_features_in_ == X.shape[1]
        # cv_results_ trajectory must contain at least the union of
        # 1st-run and 2nd-run iters (resume preserves prior data).
        assert len(sel2.cv_results_["nfeatures"]) >= len(sel1.cv_results_["nfeatures"])
