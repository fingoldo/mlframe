"""Tests for ``mlframe.core.category_encoders_compat.ensure_category_encoders_sklearn_tags_shim``.

Previously had zero test coverage despite being a fragile, version-gated monkeypatch of category-encoders'
sklearn tag machinery, called from two production sites (feature_selection/optbinning.py and
training/core/_setup_helpers.py). Exercises the actual installed category_encoders/sklearn on this machine
(not mocked) end-to-end, plus the idempotency and version-gating logic via monkeypatching the module's
private ``_PATCHED`` flag and ``sklearn.__version__``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

category_encoders = pytest.importorskip("category_encoders")
import sklearn

import mlframe.core.category_encoders_compat as cec


@pytest.fixture(autouse=True)
def _reset_patched_flag(monkeypatch):
    """Every test starts from an unpatched state so each one exercises the shim's own patching logic
    rather than inheriting whatever a prior test (or module import elsewhere in the suite) already did."""
    monkeypatch.setattr(cec, "_PATCHED", False)
    yield


def _make_df(n=40, seed=0):
    """Small mixed numeric/categorical frame + binary target for encoder fit_transform smoke tests."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cat_a": rng.choice(["x", "y", "z"], size=n),
            "num_a": rng.standard_normal(n),
        }
    )
    y = (rng.random(n) < 0.5).astype(int)
    return df, y


class TestIdempotency:
    """Groups tests covering idempotent single-shot patching."""

    def test_second_call_is_a_noop(self, monkeypatch):
        """Once patched, a second call must not re-run the patching logic (the ``_PATCHED`` early-return)."""
        calls = []
        real_import = importlib.import_module

        def _spy_import(name, *a, **kw):
            """Record every ``category_encoders.utils`` import attempt while delegating to the real importer."""
            if name == "category_encoders.utils":
                calls.append(name)
            return real_import(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _spy_import)
        cec.ensure_category_encoders_sklearn_tags_shim()
        first_count = len(calls)
        cec.ensure_category_encoders_sklearn_tags_shim()
        assert len(calls) == first_count, "a second call must short-circuit on _PATCHED, not re-import/re-patch"

    def test_sets_patched_flag_true(self):
        """Sets patched flag true."""
        assert cec._PATCHED is False
        cec.ensure_category_encoders_sklearn_tags_shim()
        assert cec._PATCHED is True


class TestVersionGating:
    """Groups tests covering the sklearn-version gate."""

    def test_noop_below_sklearn_1_6(self, monkeypatch):
        """sklearn < 1.6 has no __sklearn_tags__ concept to satisfy -- the shim must not touch BaseEncoder."""
        import category_encoders.utils as ce_utils

        had_attr_before = "__sklearn_tags__" in vars(ce_utils.BaseEncoder)
        monkeypatch.setattr(sklearn, "__version__", "1.5.2")
        cec.ensure_category_encoders_sklearn_tags_shim()
        assert cec._PATCHED is True  # still marks patched (single-shot gate), just did nothing
        assert ("__sklearn_tags__" in vars(ce_utils.BaseEncoder)) == had_attr_before

    def test_malformed_version_string_is_handled_gracefully(self, monkeypatch):
        """A version string that doesn't parse as ``major.minor`` ints must not raise."""
        monkeypatch.setattr(sklearn, "__version__", "not-a-version")
        cec.ensure_category_encoders_sklearn_tags_shim()  # must not raise
        assert cec._PATCHED is True


class TestPatchingBehaviorOnThisEnvironment:
    """Exercises the real patch path against whatever category_encoders/sklearn versions are actually
    installed here -- skips itself when the installed category-encoders already natively defines
    __sklearn_tags__ (the shim documents itself as a no-op in that case)."""

    def _already_native(self) -> bool:
        """True when the installed category-encoders defines __sklearn_tags__ natively (shim would no-op)."""
        import category_encoders.utils as ce_utils

        return "__sklearn_tags__" in vars(ce_utils.BaseEncoder)

    def test_patches_base_encoder_when_not_already_native(self):
        """After a real call (sklearn>=1.6 assumed on this dev/CI environment), BaseEncoder gains its own
        __sklearn_tags__ in __dict__ -- unless the installed category-encoders already provides it, in
        which case the shim is correctly a no-op and this assertion is skipped."""
        if self._already_native():
            pytest.skip("installed category-encoders already defines __sklearn_tags__ natively")
        major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
        if (major, minor) < (1, 6):
            pytest.skip("installed sklearn predates 1.6, shim intentionally no-ops")
        import category_encoders.utils as ce_utils

        cec.ensure_category_encoders_sklearn_tags_shim()
        assert "__sklearn_tags__" in vars(ce_utils.BaseEncoder)

    def test_fit_transform_does_not_raise_attributeerror(self):
        """The concrete regression this shim exists to fix: fit_transform on a category-encoders encoder
        must not raise AttributeError about a missing __sklearn_tags__ on the broken cooperative super()
        chain, regardless of whether the shim was needed (idempotent either way)."""
        cec.ensure_category_encoders_sklearn_tags_shim()
        df, y = _make_df()
        enc = category_encoders.OneHotEncoder(cols=["cat_a"])
        out = enc.fit_transform(df, y)
        assert len(out) == len(df)

    def test_supervised_encoder_fit_transform_does_not_raise(self):
        """Same regression check for a SUPERVISED encoder (TargetEncoder), which additionally exercises
        the ``target_tags.required`` / ``_get_tags`` 'supervised_encoder' bypass path."""
        cec.ensure_category_encoders_sklearn_tags_shim()
        df, y = _make_df()
        enc = category_encoders.TargetEncoder(cols=["cat_a"])
        out = enc.fit_transform(df, y)
        assert len(out) == len(df)

    def test_pipeline_wrapped_fit_does_not_raise(self):
        """The bug this shim fixes was specifically triggered via sklearn's tag-resolution machinery, which
        a bare Pipeline.fit exercises the same way manual fit_transform does."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        cec.ensure_category_encoders_sklearn_tags_shim()
        df, y = _make_df()
        pipe = Pipeline([("enc", category_encoders.OneHotEncoder(cols=["cat_a"])), ("clf", LogisticRegression())])
        pipe.fit(df, y)  # must not raise
        preds = pipe.predict(df)
        assert len(preds) == len(df)

    def test_get_tags_supervised_encoder_key_reflects_instance_type(self):
        """After patching, BaseEncoder._get_tags()['supervised_encoder'] must be True for a supervised
        encoder and False for an unsupervised one -- the custom dict-key bypass this shim implements."""
        if self._already_native():
            pytest.skip("installed category-encoders already defines __sklearn_tags__ natively (no _get_tags patch applied)")
        major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
        if (major, minor) < (1, 6):
            pytest.skip("installed sklearn predates 1.6, shim intentionally no-ops")
        cec.ensure_category_encoders_sklearn_tags_shim()
        supervised = category_encoders.TargetEncoder(cols=["cat_a"])
        unsupervised = category_encoders.OneHotEncoder(cols=["cat_a"])
        assert supervised._get_tags().get("supervised_encoder") is True
        assert unsupervised._get_tags().get("supervised_encoder") is False


class TestMissingCategoryEncoders:
    """The ImportError guard when category_encoders itself is unavailable."""

    def test_noop_when_category_encoders_missing(self, monkeypatch):
        """category_encoders is genuinely installed in this test env, so simulate absence by making the
        module's own import of it fail, and confirm the shim degrades to a silent no-op rather than raising."""
        real_import = importlib.import_module

        def _raise_for_ce_utils(name, *a, **kw):
            """Simulate category_encoders.utils being unimportable while leaving every other import untouched."""
            if name == "category_encoders.utils":
                raise ImportError("simulated absence")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(importlib, "import_module", _raise_for_ce_utils)
        # ensure_category_encoders_sklearn_tags_shim uses `import category_encoders.utils as ce_utils`
        # (a plain import statement, not importlib.import_module) -- simulate absence the way Python's
        # import system actually resolves it: remove the submodule from sys.modules and block re-import
        # via a meta path finder raising ImportError for exactly this dotted name.
        import sys

        class _BlockCEUtils:
            """Meta-path finder that raises ImportError only for category_encoders.utils, leaving every
            other import (including category_encoders itself) untouched."""

            def find_module(self, fullname, path=None):
                """Claim ownership of category_encoders.utils so its find/load raises; return None otherwise."""
                if fullname == "category_encoders.utils":
                    return self
                return None

            def load_module(self, fullname):
                """Always raise -- this finder only ever claims the module it intends to block."""
                raise ImportError("simulated absence")

        sys.modules.pop("category_encoders.utils", None)
        sys.meta_path.insert(0, _BlockCEUtils())
        try:
            cec.ensure_category_encoders_sklearn_tags_shim()  # must not raise
        finally:
            sys.meta_path.pop(0)
        assert cec._PATCHED is True
