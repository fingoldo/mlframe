"""Gate tests for the Muon Triton Newton-Schulz path.

The dispatch decision must be empirical, not hardcoded from compute
capability: a low-end Ampere+/Ada laptop GPU has cc >= 8.0 yet loses to
cuBLAS, so the gate has to measure and keep such cards on eager.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural import _muon_optimizer as mopt
from mlframe.training.neural import _muon_triton_kernel as mtk


def test_env_force_parsing():
    """Env force parsing."""
    try:
        for v in ("0", "off", "false", "no", "OFF"):
            os.environ[mtk._TRITON_ENV_VAR] = v
            assert mtk._env_force() is False
        for v in ("1", "on", "true", "yes", "ON"):
            os.environ[mtk._TRITON_ENV_VAR] = v
            assert mtk._env_force() is True
        for v in ("", "auto", "garbage"):
            os.environ[mtk._TRITON_ENV_VAR] = v
            assert mtk._env_force() is None
    finally:
        os.environ.pop(mtk._TRITON_ENV_VAR, None)


def test_size_bucket_is_power_of_two_ceiling():
    """Size bucket is power of two ceiling."""
    assert mtk._size_bucket(256) == 256
    assert mtk._size_bucket(257) == 512
    assert mtk._size_bucket(1000) == 1024
    assert mtk._size_bucket(2048) == 2048


def test_cpu_tensor_never_uses_triton():
    # No CUDA tensor -> always eager, regardless of host.
    """Cpu tensor never uses triton."""
    assert mtk.maybe_newton_schulz_triton(torch.randn(512, 512), steps=2) is None


# Cache the one-shot "can this GPU actually EXECUTE the Triton kernel" probe. A CUDA error poisons the
# context for the rest of the process, so we run the probe at most once.
_TRITON_EXEC_PROBE: list = []


def _triton_can_execute() -> tuple[bool, str]:
    """Triton can execute."""
    if _TRITON_EXEC_PROBE:
        return _TRITON_EXEC_PROBE[0]
    fn = mtk.get_triton_ns_fn()
    result: tuple[bool, str]
    try:
        d = mtk._MIN_DIM_FOR_TRITON_NS
        fn(torch.randn(d, d, device="cuda"), 2)
        torch.cuda.synchronize()
        result = (True, "")
    except Exception as e:  # bf16 cublasGemmEx etc. on low-end Ada/Ampere laptop cards
        result = (False, f"{type(e).__name__}: {str(e)[:120]}")
    _TRITON_EXEC_PROBE.append(result)
    return result


def _require_ampere_gpu():
    """Require ampere gpu."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if torch.cuda.get_device_capability() < mtk._MIN_COMPUTE_CAPABILITY:
        pytest.skip("needs an Ampere+ (cc >= 8.0) GPU")
    if mtk.get_triton_ns_fn() is None:
        pytest.skip("Triton kernel did not compile on this host")
    # Compilation != execution: low-end Ampere+/Ada laptop GPUs (cc >= 8.0) compile the kernel but its
    # bf16 cublasGemmEx path raises CUBLAS_STATUS_INVALID_VALUE at runtime, so the gate correctly keeps
    # them on eager and maybe_newton_schulz_triton returns None. The "Triton wins -> produces output"
    # assertion is only meaningful on a card that can run the kernel; skip where it physically cannot.
    ok, why = _triton_can_execute()
    if not ok:
        pytest.skip(f"Triton kernel compiles but cannot execute on this GPU ({why})")


@pytest.mark.gpu
def test_gate_picks_eager_when_calibration_shows_loss(monkeypatch):
    """Gate picks eager when calibration shows loss."""
    _require_ampere_gpu()
    mtk._TRITON_VERDICT.clear()
    monkeypatch.delenv(mtk._TRITON_ENV_VAR, raising=False)
    # Pretend Triton is 2x slower on this device: gate must choose eager
    # even though compute capability passes the cheap pre-filter.
    monkeypatch.setattr(mtk, "_calibrate_triton_vs_eager", lambda *a, **k: 0.5)
    G = torch.randn(512, 512, device="cuda")
    assert mtk.maybe_newton_schulz_triton(G, steps=2) is None
    assert mtk._TRITON_VERDICT  # verdict was measured and cached


@pytest.mark.gpu
def test_gate_picks_triton_when_calibration_shows_win(monkeypatch):
    """Gate picks triton when calibration shows win."""
    _require_ampere_gpu()
    mtk._TRITON_VERDICT.clear()
    monkeypatch.delenv(mtk._TRITON_ENV_VAR, raising=False)
    monkeypatch.setattr(mtk, "_calibrate_triton_vs_eager", lambda *a, **k: 3.0)
    G = torch.randn(512, 512, device="cuda")
    out = mtk.maybe_newton_schulz_triton(G, steps=2)
    assert out is not None
    assert out.shape == G.shape


@pytest.mark.gpu
def test_env_force_off_skips_triton_without_calibrating(monkeypatch):
    """Env force off skips triton without calibrating."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    monkeypatch.setenv(mtk._TRITON_ENV_VAR, "0")
    mtk._TRITON_VERDICT.clear()
    # Calibration must NOT run when the user forced eager.
    monkeypatch.setattr(mtk, "_calibrate_triton_vs_eager", lambda *a, **k: pytest.fail("calibration ran despite force-off"))
    G = torch.randn(512, 512, device="cuda")
    assert mtk.maybe_newton_schulz_triton(G, steps=2) is None
    assert not mtk._TRITON_VERDICT


def test_dispatch_matches_eager_on_cpu():
    # On CPU the gate returns None, so the dispatcher must be byte-for-byte the eager reference.
    """Dispatch matches eager on cpu."""
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    assert torch.equal(mopt._newton_schulz_dispatch(G, steps=5), mopt._zeropower_via_newtonschulz5(G, steps=5))


def test_muon_step_routes_through_triton_gate(monkeypatch):
    # Pins the wiring: Muon.step must go through the Triton gate, not call the eager reference directly.
    """Muon step routes through triton gate."""
    hits = {"n": 0}
    real = mopt.maybe_newton_schulz_triton

    def spy(G, steps=4):
        """Spy."""
        hits["n"] += 1
        return real(G, steps=steps)  # None on CPU -> eager fallback inside the dispatcher

    monkeypatch.setattr(mopt, "maybe_newton_schulz_triton", spy)
    layer = torch.nn.Linear(8, 4, bias=False)  # Muon only handles 2D params
    opt = mopt.Muon(layer.parameters(), lr=0.02)
    layer(torch.randn(16, 8)).sum().backward()
    opt.step()
    assert hits["n"] >= 1
