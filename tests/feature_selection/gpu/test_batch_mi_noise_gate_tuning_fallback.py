"""Direct unit coverage for ``_batch_mi_noise_gate_tuning`` (mrmr_audit_2026-07-20 test_coverage.md
#8): the pre-sweep size-gated backend-choice heuristic and the KTC-lookup-failure fallback,
mirroring the already-tested sibling KTC modules' fallback contract."""

from __future__ import annotations

import mlframe.feature_selection.filters._batch_mi_noise_gate_tuning as tuning_mod


class TestFallbackChoiceSizeGating:
    """_batch_mi_noise_gate_fallback_choice: GPU only when BOTH n_rows and n_cols clear their
    respective minimums; CPU otherwise, regardless of GPU availability."""

    def test_below_both_thresholds_is_cpu(self, monkeypatch):
        """Below both thresholds is cpu."""
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", True)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", True)
        choice = tuning_mod._batch_mi_noise_gate_fallback_choice(n_rows=100, n_cols=10)
        assert choice == "cpu"

    def test_rows_clear_but_cols_below_is_cpu(self, monkeypatch):
        """Rows clear but cols below is cpu."""
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", True)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", True)
        choice = tuning_mod._batch_mi_noise_gate_fallback_choice(n_rows=tuning_mod.GPU_MIN_ROWS, n_cols=tuning_mod.GPU_MIN_COLS - 1)
        assert choice == "cpu"

    def test_cols_clear_but_rows_below_is_cpu(self, monkeypatch):
        """Cols clear but rows below is cpu."""
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", True)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", True)
        choice = tuning_mod._batch_mi_noise_gate_fallback_choice(n_rows=tuning_mod.GPU_MIN_ROWS - 1, n_cols=tuning_mod.GPU_MIN_COLS)
        assert choice == "cpu"

    def test_both_clear_and_cupy_available_prefers_cupy(self, monkeypatch):
        """Both clear and cupy available prefers cupy."""
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", True)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", True)
        choice = tuning_mod._batch_mi_noise_gate_fallback_choice(n_rows=tuning_mod.GPU_MIN_ROWS, n_cols=tuning_mod.GPU_MIN_COLS)
        assert choice == "cupy", "cupy must be preferred over cuda when both are available"

    def test_both_clear_no_cupy_but_cuda_available_uses_cuda(self, monkeypatch):
        """Both clear no cupy but cuda available uses cuda."""
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", False)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", True)
        choice = tuning_mod._batch_mi_noise_gate_fallback_choice(n_rows=tuning_mod.GPU_MIN_ROWS, n_cols=tuning_mod.GPU_MIN_COLS)
        assert choice == "cuda"

    def test_both_clear_but_no_gpu_backend_available_is_cpu(self, monkeypatch):
        """Both clear but no gpu backend available is cpu."""
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", False)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", False)
        choice = tuning_mod._batch_mi_noise_gate_fallback_choice(n_rows=tuning_mod.GPU_MIN_ROWS, n_cols=tuning_mod.GPU_MIN_COLS)
        assert choice == "cpu"


class TestBackendChoiceFallsBackOnKtcFailure:
    """_batch_mi_noise_gate_backend_choice must fall back to the size-gated heuristic when the KTC
    lookup itself fails (e.g. pyutilz's KernelTuningCache import/usage raises), never propagating."""

    def test_ktc_import_failure_falls_back_to_size_gated_choice(self, monkeypatch):
        """Simulate a KTC import failure by making the cache module unimportable via sys.modules poisoning."""
        import builtins

        real_import = builtins.__import__

        def _poisoned_import(name, *args, **kwargs):
            """Raise on the specific pyutilz KTC import path; delegate everything else."""
            if name == "pyutilz.performance.kernel_tuning.cache":
                raise ImportError("simulated KTC unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _poisoned_import)
        monkeypatch.setattr(tuning_mod, "_CUPY_AVAIL", False)
        monkeypatch.setattr(tuning_mod, "_CUDA_AVAIL", False)

        # Below-threshold shape -> heuristic says "cpu" regardless of the KTC failure.
        choice = tuning_mod._batch_mi_noise_gate_backend_choice(n_rows=10, n_cols=10)
        assert choice == "cpu"
