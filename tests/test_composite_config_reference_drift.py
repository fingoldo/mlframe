"""Doc-drift guard for the generated ``CompositeTargetDiscoveryConfig`` reference.

Re-runs ``scripts/gen_composite_config_reference.render_markdown()`` in memory and
asserts the committed ``docs/composite_config_reference.md`` matches byte-for-byte.
A new config field (or a changed comment / default / type) without regenerating
the doc therefore fails CI -- forcing the doc to stay in sync with the config.

To fix a failure: ``python scripts/gen_composite_config_reference.py`` then commit
the updated doc.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GEN_PATH = _REPO_ROOT / "scripts" / "gen_composite_config_reference.py"
_DOC_PATH = _REPO_ROOT / "docs" / "composite_config_reference.md"


def _load_generator():
    """Import the generator module from its file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "gen_composite_config_reference", _GEN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_module_importable():
    module = _load_generator()
    assert hasattr(module, "render_markdown")
    assert hasattr(module, "main")


def test_render_is_idempotent():
    """Two consecutive renders produce identical output (no nondeterminism)."""
    module = _load_generator()
    assert module.render_markdown() == module.render_markdown()


def test_render_covers_every_config_field():
    """Every declared config field appears as a row in the rendered doc."""
    module = _load_generator()
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    content = module.render_markdown()
    for name in CompositeTargetDiscoveryConfig.model_fields:
        assert f"| `{name}` |" in content, f"field {name!r} missing from generated doc"


def test_committed_doc_matches_generated():
    """The committed doc must equal the freshly generated content (no drift)."""
    module = _load_generator()
    assert _DOC_PATH.exists(), (
        f"{_DOC_PATH} is missing; run "
        "`python scripts/gen_composite_config_reference.py`"
    )
    # ``newline=""`` reads the file without translating CRLF so the comparison is
    # exact against the explicit-CRLF content the generator emits.
    committed = _DOC_PATH.read_text(encoding="utf-8", newline="")
    generated = module.render_markdown()
    assert committed == generated, (
        "docs/composite_config_reference.md is out of date; run "
        "`python scripts/gen_composite_config_reference.py` and commit the result."
    )


def test_dict_config_acceptance_documented():
    """The doc must explain that the suite accepts a plain dict config."""
    module = _load_generator()
    content = module.render_markdown()
    assert "dict" in content
    assert "_ensure_config" in content
    assert "composite_target_discovery_config" in content
