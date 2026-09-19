"""LightGBM 4.5/4.6 made every numpy fit + numpy predict print a false scikit-learn warning.

"X does not have valid feature names, but LGBMRegressor was fitted with feature names" appeared in suite logs right
after the composite-discovery baseline diagnostics: the achievable-ceiling precheck fits its tiny LightGBM probe on a
numpy matrix and predicts on one, yet LightGBM 4.5/4.6 report the auto names ``Column_i`` through ``feature_names_in_``.
mlframe backports LightGBM 4.7's fix; the genuine warning (fitted on a DataFrame, predicted on numpy) must survive.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe._lightgbm_compat import needs_backport, patch_lgbm_model_class

_FEATURE_NAMES_MSG = "valid feature names"


def test_fresh_interpreter_numpy_fit_predict_is_silent_but_frame_mismatch_still_warns():
    """The import hook patches lightgbm whenever it is imported after mlframe, in a clean process."""
    pytest.importorskip("lightgbm")
    code = textwrap.dedent(
        """
        import warnings
        import numpy as np, pandas as pd
        import mlframe
        import lightgbm as lgb

        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 3)); y = X[:, 0] + rng.normal(size=400)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(X, y).predict(X)
        print("numpy_warned", any("valid feature names" in str(w.message) for w in rec))

        df = pd.DataFrame(X, columns=["a", "b", "c"])
        m = lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(df, y)
        print("names", list(m.feature_names_in_))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            m.predict(X)
        print("frame_then_numpy_warned", any("valid feature names" in str(w.message) for w in rec))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)
    assert lines.get("numpy_warned") == "False", out.stdout
    assert lines.get("names") == "['a', 'b', 'c']", out.stdout
    assert lines.get("frame_then_numpy_warned") == "True", out.stdout


class _Lgbm46Like:
    """Mimics LightGBM 4.6: ``feature_names_in_`` returns the stored names, auto ones included."""

    def __init__(self, names):
        self.feature_name_ = names

    @property
    def feature_names_in_(self):
        """Stored feature names, auto-generated ones included, as LightGBM 4.6 returns them."""
        return np.array(self.feature_name_)

    @feature_names_in_.deleter
    def feature_names_in_(self):
        """No-op deleter mirroring the one LightGBM defines."""
        pass


def test_patch_hides_auto_names_and_keeps_real_ones():
    """The backport itself, independent of the installed LightGBM version."""
    cls = type("Model", (_Lgbm46Like,), {"feature_names_in_": _Lgbm46Like.__dict__["feature_names_in_"]})
    assert patch_lgbm_model_class(cls) is True
    assert patch_lgbm_model_class(cls) is False  # idempotent
    assert getattr(cls(["Column_0", "Column_1"]), "feature_names_in_", None) is None
    assert list(cls(["a", "b"]).feature_names_in_) == ["a", "b"]
    # Out-of-order or partial auto-looking names were supplied by the user: kept.
    assert list(cls(["Column_1", "Column_0"]).feature_names_in_) == ["Column_1", "Column_0"]
    del cls(["a"]).feature_names_in_  # the deleter LightGBM relies on is preserved


@pytest.mark.parametrize("version,expected", [("4.4.0", False), ("4.5.0", True), ("4.6.0", True), ("4.6.0.99", True),
                                              ("4.7.0", False), ("5.0.0", False), ("garbage", False)])
def test_backport_only_targets_affected_releases(version, expected):
    """The backport activates only for the LightGBM releases that report auto names (4.5.x-4.6.x)."""
    assert needs_backport(version) is expected


def test_achievable_ceiling_precheck_emits_no_feature_names_warning():
    """The suite path where the warning was seen: the precheck's numpy LightGBM probe."""
    pytest.importorskip("lightgbm")
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core._achievable_ceiling import run_achievable_ceiling_precheck

    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame({"a": rng.lognormal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = np.exp(0.5 * df["a"].to_numpy()) + 3 * df["b"].to_numpy() + rng.normal(size=n)
    df.loc[rng.random(n) < 0.3, "a"] = np.nan
    df["t"] = y
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        verdict = run_achievable_ceiling_precheck(config=CompositeTargetDiscoveryConfig(), df=df, target_col="t", feature_cols=["a", "b", "c"], y_train=y)
    assert verdict is not None
    assert not [w for w in rec if _FEATURE_NAMES_MSG in str(w.message)]
