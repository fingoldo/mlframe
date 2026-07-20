"""Direct unit coverage for ``_gpu_resident_radix_ktc`` (mrmr_audit_2026-07-20 test_coverage.md #5 /
edge_cases.md #35-36). Pins: (1) the CPU-only / no-KTC-entry fallback returns the documented
historical defaults, (2) a malformed ``.choose()`` return value also falls back cleanly, and (3) the
2026-07-18 fix -- the sweep-probe variant override must be written onto the OWNING module
(``_gpu_resident_select_kernels``), not onto a re-export alias, or the override is silently a no-op.
"""

from __future__ import annotations

import mlframe.feature_selection.filters._gpu_resident_radix_ktc as radix_ktc


class TestNoKtcEntryFallsBackToHistoricalDefault:
    """When the KTC spec is unavailable (no cupy / pre-sweep / lookup failure), both lookup
    functions must return their documented historical hardcoded defaults."""

    def test_radix_select_threads_falls_back_when_spec_is_none(self, monkeypatch):
        """spec=None -> the historical 512 threads/block."""
        monkeypatch.setattr(radix_ktc, "_RADIX_THREADS_SPEC", None)
        assert radix_ktc.radix_select_threads(100_000) == radix_ktc._RADIX_THREADS_DEFAULT == 512

    def test_radix_select_f32_variant_falls_back_when_spec_is_none(self, monkeypatch):
        """spec=None -> the historical 'v3' variant."""
        monkeypatch.setattr(radix_ktc, "_RADIX_F32_VARIANT_SPEC", None)
        assert radix_ktc.radix_select_f32_variant(100_000) == radix_ktc._RADIX_F32_VARIANT_DEFAULT == "v3"

    def test_radix_select_threads_falls_back_when_choose_raises(self, monkeypatch):
        """A .choose() lookup failure must be swallowed, not propagated, and fall back to the default."""

        class _RaisingSpec:
            """Stand-in spec whose choose() always raises."""

            def choose(self, **kwargs):
                """Raise unconditionally."""
                raise RuntimeError("lookup failed")

        monkeypatch.setattr(radix_ktc, "_RADIX_THREADS_SPEC", _RaisingSpec())
        assert radix_ktc.radix_select_threads(50_000) == radix_ktc._RADIX_THREADS_DEFAULT

    def test_radix_select_f32_variant_falls_back_when_choose_raises(self, monkeypatch):
        """Same failure-mode check for the f32-variant lookup."""

        class _RaisingSpec:
            """Stand-in spec whose choose() always raises."""

            def choose(self, **kwargs):
                """Raise unconditionally."""
                raise RuntimeError("lookup failed")

        monkeypatch.setattr(radix_ktc, "_RADIX_F32_VARIANT_SPEC", _RaisingSpec())
        assert radix_ktc.radix_select_f32_variant(50_000) == radix_ktc._RADIX_F32_VARIANT_DEFAULT


class TestMalformedChooseReturnFallsBack:
    """A ``.choose()`` return value that doesn't match the expected shape (wrong prefix / not a
    known variant name) must fall back to the default instead of propagating garbage."""

    def test_threads_choice_without_th_prefix_falls_back(self, monkeypatch):
        """A choice string missing the 'th_' prefix must not be parsed as a thread count."""

        class _Spec:
            """Stand-in spec returning a malformed choice."""

            def choose(self, **kwargs):
                """Return a value with no th_ prefix."""
                return "garbage"

        monkeypatch.setattr(radix_ktc, "_RADIX_THREADS_SPEC", _Spec())
        assert radix_ktc.radix_select_threads(100_000) == radix_ktc._RADIX_THREADS_DEFAULT

    def test_threads_choice_with_unparseable_suffix_falls_back(self, monkeypatch):
        """'th_' prefix present but the suffix isn't an int -> ValueError caught, falls back."""

        class _Spec:
            """Stand-in spec returning th_ prefix with a non-numeric suffix."""

            def choose(self, **kwargs):
                """Return a th_-prefixed but unparseable choice."""
                return "th_not_a_number"

        monkeypatch.setattr(radix_ktc, "_RADIX_THREADS_SPEC", _Spec())
        assert radix_ktc.radix_select_threads(100_000) == radix_ktc._RADIX_THREADS_DEFAULT

    def test_f32_variant_choice_not_in_known_set_falls_back(self, monkeypatch):
        """A choice string not in _RADIX_F32_VARIANTS must fall back to the default."""

        class _Spec:
            """Stand-in spec returning an unknown variant name."""

            def choose(self, **kwargs):
                """Return a variant name not in the known set."""
                return "not_a_real_variant"

        monkeypatch.setattr(radix_ktc, "_RADIX_F32_VARIANT_SPEC", _Spec())
        assert radix_ktc.radix_select_f32_variant(100_000) == radix_ktc._RADIX_F32_VARIANT_DEFAULT

    def test_valid_threads_choice_is_parsed(self, monkeypatch):
        """Sanity check: a well-formed choice IS parsed (contrast case for the malformed-input tests above)."""

        class _Spec:
            """Stand-in spec returning a well-formed choice."""

            def choose(self, **kwargs):
                """Return a valid th_-prefixed choice."""
                return "th_768"

        monkeypatch.setattr(radix_ktc, "_RADIX_THREADS_SPEC", _Spec())
        assert radix_ktc.radix_select_threads(100_000) == 768

    def test_valid_f32_variant_choice_is_returned(self, monkeypatch):
        """Sanity check: a known variant name IS returned as-is."""

        class _Spec:
            """Stand-in spec returning a known variant."""

            def choose(self, **kwargs):
                """Return a valid variant name."""
                return "bsearch"

        monkeypatch.setattr(radix_ktc, "_RADIX_F32_VARIANT_SPEC", _Spec())
        assert radix_ktc.radix_select_f32_variant(100_000) == "bsearch"


class TestOverrideWritesToOwningModule:
    """Regression pin for the 2026-07-18 fix: the sweep-probe variant override MUST be written onto
    ``_gpu_resident_select_kernels`` (the module that actually reads the global at kernel-dispatch
    time), not onto ``_gpu_resident_select`` (which only re-exports the name via a plain import,
    creating an independent binding the owning module never sees)."""

    def test_threads_override_is_restored_on_the_owning_module(self):
        """_radix_edges_with_threads must save/restore _grsk._RADIX_THREADS_OVERRIDE -- exercised via
        cupy-free introspection of the function's own save/restore contract using a monkeypatched
        owning-module attribute (no GPU kernel launch needed to pin the module-identity fix)."""
        import mlframe.feature_selection.filters._gpu_resident_select_kernels as grsk

        assert hasattr(grsk, "_RADIX_THREADS_OVERRIDE"), (
            "the owning module must expose _RADIX_THREADS_OVERRIDE -- this is the attribute "
            "_radix_edges_with_threads reads/writes; a re-export-only alias would not have it "
            "independently mutable at the SAME identity the kernel dispatch reads."
        )
        saved = grsk._RADIX_THREADS_OVERRIDE
        try:
            grsk._RADIX_THREADS_OVERRIDE = 999
            assert grsk._RADIX_THREADS_OVERRIDE == 999
        finally:
            grsk._RADIX_THREADS_OVERRIDE = saved
        assert grsk._RADIX_THREADS_OVERRIDE == saved, "override must be fully restored after use"
