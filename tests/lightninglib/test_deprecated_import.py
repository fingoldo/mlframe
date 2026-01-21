"""
Test that the deprecated lightninglib import shows proper deprecation warning.

Run tests:
    pytest tests/lightninglib/test_deprecated_import.py -v
"""

import pytest
import warnings
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDeprecatedImport:
    """Tests for deprecated mlframe.lightninglib import."""

    def test_deprecated_import_shows_warning(self):
        """Test that importing from mlframe.lightninglib shows deprecation warning."""
        # Clear module cache if already imported
        modules_to_remove = [k for k in sys.modules.keys() if 'mlframe.lightninglib' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from mlframe.lightninglib import TorchDataset

            # Check that at least one deprecation warning was issued
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1, "Expected DeprecationWarning when importing from lightninglib"

            # Check warning message
            warning_messages = [str(x.message) for x in deprecation_warnings]
            assert any("deprecated" in msg.lower() for msg in warning_messages), \
                f"Expected 'deprecated' in warning message, got: {warning_messages}"

    def test_deprecated_import_still_works(self):
        """Test that deprecated import still provides functional classes."""
        # Clear module cache
        modules_to_remove = [k for k in sys.modules.keys() if 'mlframe.lightninglib' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mlframe.lightninglib import (
                TorchDataset,
                TorchDataModule,
                MLPTorchModel,
                generate_mlp,
            )

            # Verify classes are importable and usable
            assert TorchDataset is not None
            assert TorchDataModule is not None
            assert MLPTorchModel is not None
            assert callable(generate_mlp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
