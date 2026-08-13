"""Compat shim for category-encoders releases that predate ``__sklearn_tags__``.

category-encoders only started defining ``__sklearn_tags__`` natively in 2.8.0 (confirmed by
inspecting the 2.7.0 vs 2.8.0 wheels directly) -- but category-encoders>=2.8.0 itself requires
Python>=3.10, so on Python<3.10 the only installable releases (<2.8.0) lack it. Under sklearn>=1.6,
whose tag-resolution machinery (``sklearn.utils.get_tags`` / an estimator's own ``_get_tags``) walks
the MRO looking for ``__sklearn_tags__``, this surfaces as ``AttributeError: 'super' object has no
attribute '__sklearn_tags__'`` on the very first ``fit``/``fit_transform`` call -- confirmed live in
CI on a Python 3.9 leg (sklearn==1.6.1 + category-encoders==2.6.4), and reproduced exactly (same
traceback) in an isolated Python 3.9.25 + scikit-learn==1.6.1 + category-encoders==2.6.4 venv while
diagnosing this fix.

Root cause (NOT simply "BaseEncoder is missing a method" -- verified by walking the actual MRO):
category-encoders' own multiple-inheritance class hierarchy places ``sklearn.base.BaseEstimator``
BEFORE ``category_encoders.utils.SupervisedTransformerMixin``/``TransformerWithTargetMixin`` in the
MRO (sklearn's own convention is the mixin first, ``BaseEstimator`` last -- this is exactly the
ordering bug sklearn's own maintainers document at
https://github.com/scikit-learn/scikit-learn/issues/30479). Those two category-encoders mixins each
define their OWN legacy ``_more_tags()`` without a matching ``__sklearn_tags__()``, which makes
sklearn's ``_find_tags_provider`` fall back to the crash-prone legacy MRO-walk path instead of the
modern one (which has a documented try/except workaround for exactly this ordering bug -- but only
takes effect when every legacy-tag-defining class ALSO defines ``__sklearn_tags__``). Once
``TransformerMixin.__sklearn_tags__`` (further down that same broken MRO) is invoked directly by the
legacy walk and calls its own ``super().__sklearn_tags__()``, there is nothing after it in the
instance's real MRO (``_SetOutputMixin`` / ``object``) to answer -- hence the AttributeError.

Fix, verified end-to-end in the isolated repro venv against ``TargetEncoder``, ``CatBoostEncoder``
(supervised) and ``OneHotEncoder`` (unsupervised), both via direct ``fit_transform`` and wrapped in a
real ``sklearn.pipeline.Pipeline``:
  1. Scan ``category_encoders.utils`` for every class defining its own legacy ``_more_tags``/
     ``_get_tags`` without ``__sklearn_tags__`` (currently ``SupervisedTransformerMixin`` and
     ``TransformerWithTargetMixin``, but this is discovered dynamically -- older/newer category-
     encoders releases in the <2.8.0 range may have different sibling classes with the same shape of
     gap) and give each one an explicit ``__sklearn_tags__`` that does NOT rely on the broken
     cooperative ``super()`` chain -- it calls ``sklearn.base.BaseEstimator.__sklearn_tags__(self)``
     directly, sets ``transformer_tags``, and marks ``target_tags.required`` based on whether ``self``
     is an instance of one of the discovered "supervised" mixins.
  2. ``BaseEncoder`` itself also needs the SAME patch (not just the mixins): once step 3 below adds a
     ``_get_tags`` to ``BaseEncoder``'s own ``__dict__``, ``_find_tags_provider`` would otherwise see
     ``BaseEncoder`` as ANOTHER offending class (legacy tag method present, no ``__sklearn_tags__``)
     and flip back into the broken legacy path for it specifically.
  3. category-encoders' OWN internal code (``BaseEncoder.fit`` / ``_check_fit_inputs``) calls
     ``self._get_tags().get('supervised_encoder')`` -- a custom dict key category-encoders invented
     for its pre-1.6 legacy tags dict, which sklearn's own ``_to_old_tags()`` conversion (used once the
     modern ``__sklearn_tags__`` path is active) has no way to know about, so it silently comes back
     ``None``/falsy even after steps 1-2 fix the crash, breaking ``lab_encoder_`` setup for supervised
     encoders. Bypassed entirely: ``BaseEncoder._get_tags`` is replaced with a small isinstance-based
     dict built directly from the same "supervised mixins" set discovered in step 1, sidestepping
     sklearn's tag machinery for this internal use rather than trying to make ``_to_old_tags()`` carry
     a category-encoders-specific key it was never designed to know about.

Idempotent; a no-op once the installed category-encoders already defines ``__sklearn_tags__``
natively (>=2.8.0) or sklearn itself predates 1.6 (no ``__sklearn_tags__`` concept to satisfy).
"""

from __future__ import annotations

_PATCHED = False


def ensure_category_encoders_sklearn_tags_shim() -> None:
    """Patch category-encoders' tag machinery in if the installed release predates 2.8.0's native
    fix. Call this once, right after ``import category_encoders``, before any encoder is fit."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import category_encoders.utils as ce_utils
    except ImportError:
        return
    # Check the class's OWN __dict__, not hasattr() -- inheriting __sklearn_tags__ from
    # sklearn.BaseEstimator (which every category-encoders release has always done) is not the same
    # as defining it directly, and hasattr() would report True either way, silently no-op'ing this
    # shim on exactly the releases that need it.
    if "__sklearn_tags__" in vars(ce_utils.BaseEncoder):
        return

    try:
        import sklearn
    except ImportError:
        return
    try:
        _major, _minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    except (ValueError, IndexError):
        return
    if (_major, _minor) < (1, 6):
        return

    import inspect

    from sklearn.base import BaseEstimator, TransformerTags

    supervised_mixins = tuple(
        obj
        for _name, obj in vars(ce_utils).items()
        if inspect.isclass(obj) and ("_more_tags" in vars(obj) or "_get_tags" in vars(obj)) and "__sklearn_tags__" not in vars(obj)
    )

    def _sklearn_tags_impl(self):
        """Bypasses the broken cooperative super() chain entirely (see module docstring)."""
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.transformer_tags = TransformerTags()
        tags.target_tags.required = isinstance(self, supervised_mixins)
        return tags

    for _klass in supervised_mixins:
        _klass.__sklearn_tags__ = _sklearn_tags_impl
    ce_utils.BaseEncoder.__sklearn_tags__ = _sklearn_tags_impl

    def _get_tags(self):
        """Bypasses sklearn's tag machinery for category-encoders' own internal
        'supervised_encoder' lookup, which sklearn's _to_old_tags() conversion has no way to carry."""
        return {"supervised_encoder": isinstance(self, supervised_mixins)}

    ce_utils.BaseEncoder._get_tags = _get_tags
