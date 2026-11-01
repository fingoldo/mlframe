"""``replay_cv_results`` does not run a pickle gadget inside a rewritten dump (DESER-06).

The sidecar is an integrity check, not authenticity: someone who can rewrite the dump can rewrite its sidecar too. The payload
therefore still has to pass the restricted joblib unpickler. A dump whose pickle calls ``os.mkdir`` must not create the marker.
"""

import os
import pickle

import joblib
import pytest

from mlframe.estimators.pipelines import replay_cv_results
from mlframe.utils.safe_pickle import write_sidecar


class _Gadget:
    """Unpickling this object calls ``os.mkdir(marker)``: a harmless stand-in for a code-execution gadget."""

    def __init__(self, marker):
        """Remember the path the gadget will create when unpickled."""
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


def test_replay_does_not_execute_a_gadget_in_the_dump(tmp_path):
    """A dump with a valid sidecar but a gadget payload is refused, and the gadget never runs."""
    marker = tmp_path / "gadget_ran"
    dump = tmp_path / "cv_results-gadget.dump"
    joblib.dump({"title": {"hash": _Gadget(str(marker))}}, str(dump), compress=9)
    write_sidecar(str(dump))
    with pytest.raises(pickle.UnpicklingError, match="denylist"):
        replay_cv_results(str(dump), trusted_root=str(tmp_path))
    assert not marker.exists(), "the gadget ran during replay"
