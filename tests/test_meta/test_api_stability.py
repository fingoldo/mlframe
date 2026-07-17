"""Meta-test — the public surface of ``mlframe.training`` (the
re-exported names plus their callable signatures) is captured in a
JSON snapshot.  Renames / removals fail the test until the snapshot is
explicitly refreshed.  Additions are silent (the API can grow without
breaking downstream code).

Why a snapshot test:

  Downstream consumers — notebooks, deployment scripts, sister
  projects — pin against the public names exported from
  ``mlframe.training``.  An unintentional rename (``train_mlframe_models_suite``
  → ``train_mlframe_models``) silently breaks every caller that imports
  by name.  Pull-request diffs against the snapshot make the rename
  visible to a reviewer who would otherwise scroll past one line of
  ``__init__.py``.

How to refresh after an intentional rename:

  python -m pytest tests/test_meta/test_api_stability.py --refresh-api-snapshot

  Then commit the resulting ``tests/test_meta/_api_snapshot.json``.

The snapshot captures:

  * Every name in ``mlframe.training.__all__`` (and / or the
    ``_LAZY_IMPORTS`` map) at the time of capture.
  * For each callable: its signature as a string (parameter names + kinds
    + defaults — but NOT typed annotations, which churn with refactors
    that don't change semantics).
  * For each class: the class name and its MRO (catches an accidental
    base-class change that breaks ``isinstance(obj, X)``).
"""

from __future__ import annotations

import inspect
import orjson
from pathlib import Path

import pytest

import mlframe.training as training_pkg
from pyutilz.dev.meta_test_utils import capture_signature

# Snapshot lives next to the test so it travels with the suite.
_SNAPSHOT_PATH = Path(__file__).resolve().parent / "_api_snapshot.json"


def pytest_addoption(parser):  # noqa: D401  (pytest hook)
    """Register ``--refresh-api-snapshot`` so the user can capture a
    fresh snapshot after an intentional API change."""
    # Late-binding: pytest's addoption is registered via a conftest
    # plugin file. Defining this here keeps the helper colocated, but
    # pytest only calls ``pytest_addoption`` from conftest. The
    # ``--refresh-api-snapshot`` flag is consumed via a sys.argv check
    # below to avoid the conftest dance.
    pass


def _refresh_requested() -> bool:
    """Probe sys.argv for the refresh flag (avoids the full conftest
    plugin dance — this single test doesn't justify it)."""
    import sys

    return "--refresh-api-snapshot" in sys.argv


def _public_names() -> list[str]:
    lazy = getattr(training_pkg, "_LAZY_IMPORTS", {})
    explicit = getattr(training_pkg, "__all__", [])
    names = sorted(set(lazy.keys()) | set(explicit))
    # Also fold in everything currently bound on the module that doesn't
    # start with _ (catches non-lazy direct re-exports).
    for n in dir(training_pkg):
        if n.startswith("_"):
            continue
        names.append(n)
    return sorted(set(names))


# Signature capture moved to ``pyutilz.dev.meta_test_utils`` —
# ``capture_signature`` imported above.


def _capture_class(obj: type) -> dict:
    # Normalize MRO entries that vary with stdlib / dependency minor versions.
    # 1. ``enum.StrEnum`` is stdlib on Python 3.11+ (with ``enum.ReprEnum`` in
    #    the MRO), polyfilled inside ``mlframe.training._configs_base`` on
    #    Python 3.10 (no ``ReprEnum`` in the MRO). Collapse all
    #    ``StrEnum``/``IntEnum``/``Enum`` bases to canonical ``enum.<Name>``
    #    and DROP the ``ReprEnum`` entry entirely.
    # 2. sklearn shuffles its internal ``_HTMLDocumentationLinkMixin`` /
    #    ``ReprHTMLMixin`` etc. between minor releases (``_repr_html.base`` vs
    #    ``_estimator_html_repr``). DROP every sklearn-internal mixin
    #    (``sklearn.*._*Mixin`` / ``_*Requester``) from the MRO so the snapshot
    #    survives the dep-pin matrix. mlframe doesn't depend on sklearn's
    #    private mixin layout; only the public base (``BaseEstimator``,
    #    ``ClassifierMixin``, ``RegressorMixin``, ``TransformerMixin``,
    #    ``MetaEstimatorMixin``) matters for the API contract.
    _SKIP_NAMES = {"ReprEnum"}
    _CANONICAL_ENUMS = {"StrEnum", "IntEnum", "Enum", "IntFlag", "Flag"}
    _PUBLIC_SKLEARN_MIXINS = {
        "BaseEstimator",
        "ClassifierMixin",
        "RegressorMixin",
        "TransformerMixin",
        "MetaEstimatorMixin",
        "ClusterMixin",
        "BiclusterMixin",
        "OutlierMixin",
        "DensityMixin",
        "MultiOutputMixin",
        "OneToOneFeatureMixin",
    }
    _canon = []
    _seen_canon = set()
    for c in obj.__mro__[1:]:
        if c.__name__ in _SKIP_NAMES:
            continue
        # sklearn internal: any class whose module starts with sklearn.utils.
        # or whose name starts with an underscore. sklearn moves these freely
        # between minor releases (``_repr_html.base.ReprHTMLMixin`` ->
        # ``_estimator_html_repr._HTMLDocumentationLinkMixin`` between
        # sklearn 1.5 and 1.6 was the 2026-05-24 CI trigger).
        if c.__module__.startswith("sklearn.utils."):
            continue
        if c.__module__.startswith("sklearn.") and c.__name__.startswith("_"):
            continue
        if c.__name__ in _CANONICAL_ENUMS:
            key = f"enum.{c.__name__}"
        elif c.__module__.startswith("sklearn.") and c.__name__ in _PUBLIC_SKLEARN_MIXINS:
            # Canonical sklearn.base location (sklearn moves these between
            # sklearn.base and sklearn.utils.* between minor releases too).
            key = f"sklearn.base.{c.__name__}"
        else:
            key = c.__module__ + "." + c.__name__
        if key not in _seen_canon:
            _seen_canon.add(key)
            _canon.append(key)
    return {
        "kind": "class",
        "module": obj.__module__,
        "mro": _canon,
    }


def _build_snapshot() -> dict:
    snapshot: dict[str, dict] = {}
    for name in _public_names():
        try:
            obj = getattr(training_pkg, name)
        except (AttributeError, ImportError):
            # Lazy-import failures are NOT recorded — they're either
            # already broken or genuinely platform-conditional.
            continue
        if inspect.isclass(obj):
            snapshot[name] = _capture_class(obj)
        elif callable(obj):
            snapshot[name] = {
                "kind": "callable",
                "signature": capture_signature(obj),
                "module": getattr(obj, "__module__", "?"),
            }
        else:
            snapshot[name] = {
                "kind": "value",
                "type": type(obj).__name__,
            }
    return snapshot


def test_public_api_matches_snapshot():
    """Public surface must match the recorded snapshot byte-for-byte."""
    current = _build_snapshot()
    if _refresh_requested() or not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.write_bytes(
            orjson.dumps(current, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
        )
        pytest.skip(f"snapshot refreshed at {_SNAPSHOT_PATH.name} ({len(current)} symbols)")

    expected = orjson.loads(_SNAPSHOT_PATH.read_bytes())

    diffs: list[str] = []
    # Removed / renamed names — always fail.
    for name in expected:
        if name not in current:
            diffs.append(f"REMOVED: {name}")
    # Changed signatures / kinds.
    for name in expected:
        if name not in current:
            continue
        if expected[name] != current[name]:
            old = expected[name]
            new = current[name]
            diffs.append(f"CHANGED: {name}\n      was: {old}\n      now: {new}")
    # Additions — informational, not a failure.
    additions = [n for n in current if n not in expected]
    if additions:
        import sys

        sys.stderr.write(
            f"\n[test_public_api_matches_snapshot] {len(additions)} new "
            f"public symbol(s): {', '.join(additions[:20])}"
            + (" ..." if len(additions) > 20 else "")
            + "\n  (additions are non-breaking; refresh the snapshot when "
            f"you're ready to lock them in: pytest "
            f"{Path(__file__).relative_to(Path.cwd()).as_posix()} "
            f"--refresh-api-snapshot)\n"
        )

    if diffs:
        pytest.fail(
            f"{len(diffs)} public-API change(s) detected against snapshot at "
            f"{_SNAPSHOT_PATH.name}. If intentional, refresh with "
            f"``pytest tests/test_meta/test_api_stability.py "
            f"--refresh-api-snapshot`` and commit the new snapshot:\n  "
            + "\n  ".join(diffs[:20])
            + (f"\n  ... and {len(diffs) - 20} more" if len(diffs) > 20 else "")
        )
