"""Inference-acceleration mixin carved out of ``_flat_torch_module.MLPTorchModel``.

Holds the torch.compile / CUDA-graph predict fast paths plus the predict_step
dispatch. All caches set here (``_cuda_graph_predict_cache``,
``_compiled_predict_fn``, ``_compile_predict_failed``) are non-picklable runtime
state and are excluded by ``MLPTorchModel.__getstate__`` / re-initialised by
``__setstate__``; see the parent for the exclusion contract.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, cast

import torch

logger = logging.getLogger("mlframe.training.neural.flat")

# Mixed into MLPTorchModel(_PredictAccelMixin, _LossMixin, L.LightningModule) -- the real nn.Module
# base (and hence .training/.eval()/__call__) comes from LightningModule at runtime. This
# TYPE_CHECKING-only base lets mypy see those without changing the runtime MRO (plain `object` there).
if TYPE_CHECKING:
    _PredictAccelBase = torch.nn.Module
else:
    _PredictAccelBase = object

# Max number of CAPTURED CUDA graphs retained per model instance. Each entry pins ~2 device buffers
# (static in + static out) + the graph object, so an inference run over many distinct batch shapes (ragged
# tails across folds/targets/datasets) would otherwise grow VRAM without bound. Evicting the least-recently
# replayed graph keeps the hot full-batch graph resident. Override via env for a host with more/less VRAM.
_CUDA_GRAPH_PREDICT_CACHE_MAX = max(1, int(os.environ.get("MLFRAME_CUDA_GRAPH_PREDICT_CACHE_MAX", "16")))


class _PredictAccelMixin(_PredictAccelBase):
    """torch.compile + CUDA-graph predict fast paths and the predict_step dispatch."""

    # Provided by the composed class (MLPTorchModel); declared here so mypy can type-check
    # the self.network reads/writes in this mixin without a self-referential inference cycle.
    network: torch.nn.Module
    hparams: Any
    task_type: Any
    _cuda_graph_predict_cache: dict
    _compiled_predict_fn: Optional[Any]
    _compile_predict_failed: bool

    def _evict_cuda_graph_cache_if_needed(self) -> None:
        """Bound the captured-graph count, reclaiming VRAM from the least-recently-used graph.

        Only real captures (tuple values) hold device memory; ``False`` sentinels (permanent eager
        fallback for a shape that failed capture) are negligible and are NOT evicted (that would cause a
        capture-retry storm). Entries are ordered oldest-first (dict insertion order + LRU move-to-end on
        replay hits), so popping from the front drops the coldest graph.
        """
        cache = self._cuda_graph_predict_cache
        graph_keys = [k for k, v in cache.items() if v is not False]
        while len(graph_keys) > _CUDA_GRAPH_PREDICT_CACHE_MAX:
            old_key = graph_keys.pop(0)
            entry = cache.pop(old_key, None)
            if entry and entry is not False:
                # Drop the last references so the captured graph + its static in/out buffers free their VRAM.
                _g, _static_in, _static_out = entry
                del _g, _static_in, _static_out
            logger.debug("Evicted LRU CUDA-graph predict cache entry shape=%s to bound VRAM.", old_key[0])

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the network if enabled.

        Compiled models cannot be pickled in PyTorch 2.8 ("cannot pickle 'ConfigModuleInstance' object"),
        which breaks checkpoint saving. See https://github.com/pytorch/pytorch/issues/126154.

        Safety guards:
          * LSTM/GRU/RNN networks: TorchDynamo INTENTIONALLY graph-breaks
            on these (pytorch/pytorch#167275, #140845). Compiled is SLOWER
            than eager due to host-device syncs around the cuDNN call.
            Detect + skip + WARN; users who set compile_network globally
            should NOT silently regress recurrent fits.
          * MLFRAME_TORCH_COMPILE_DEBUG=1 env var: routes graph_break +
            recompile events through torch._logging.set_logs so the next
            perf investigation has visibility. Off by default (logs would
            spam STDERR otherwise).
        """
        if not self.hparams.compile_network:
            return

        if torch.__version__ < "2.0":
            logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")
            return

        # LSTM/GRU/RNN are explicitly anti-compile.
        _recurrent_types = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
        try:
            _has_recurrent = any(isinstance(m, _recurrent_types) for m in self.network.modules())
        except Exception:
            _has_recurrent = False
        if _has_recurrent:
            logger.warning(
                "torch.compile requested but network contains LSTM/GRU/RNN "
                "modules. TorchDynamo intentionally graph-breaks on these "
                "(pytorch/pytorch#167275, #140845); compiled is SLOWER than "
                "eager. Skipping compile for this network."
            )
            return

        # Opt-in debug logs for graph breaks + recompiles.
        if os.environ.get("MLFRAME_TORCH_COMPILE_DEBUG", "0") == "1":
            try:
                import torch._logging as _tlog
                _tlog.set_logs(graph_breaks=True, recompiles=True)
                logger.info(
                    "MLFRAME_TORCH_COMPILE_DEBUG=1: enabled torch._logging " "graph_breaks + recompiles. Re-run with --no-cov + capture " "STDERR to inspect."
                )
            except Exception as _dbg_err:
                logger.debug("torch._logging.set_logs failed: %s", _dbg_err)

        try:
            self.network = cast(torch.nn.Module, torch.compile(self.network, mode=self.hparams.compile_network))
            logger.info("Applied torch.compile with mode='%s'", self.hparams.compile_network)
        except Exception:
            logger.warning("Failed to apply torch.compile. Using uncompiled network.", exc_info=True)

    def _invalidate_predict_caches(self) -> None:
        """Clear the CUDA-graph cache + torch.compile predict cache.

        The cache holds captured kernels that reference the parameter
        tensors' storage addresses at capture time. Most weight-update
        paths (Lookahead .copy_, EMA WeightAveraging.update_parameters,
        SWA .copy_, load_state_dict default) PRESERVE storage so the
        captured graph keeps producing correct values. But ANY path that
        REPLACES a nn.Parameter object (load_state_dict(assign=True),
        LoRA adapter swap, dynamic head replacement, user-explicit
        param mutation) leaves the captured graph pointing at stale
        storage -- replay produces predictions from PRE-swap weights
        with no exception.

        Call this AFTER any code path that could replace params
        (on_train_end checkpoint reload, user calls to set weights, etc.)
        Idempotent.
        """
        if hasattr(self, "_cuda_graph_predict_cache"):
            self._cuda_graph_predict_cache.clear()
        if hasattr(self, "_compiled_predict_fn"):
            self._compiled_predict_fn = None
            self._compile_predict_failed = False

    def _maybe_compile_predict_forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """torch.compile(mode="reduce-overhead") fast path for the inference
        forward. Strictly more powerful than the manual CUDA-graph path --
        Inductor fuses elementwise chains (BN+act+Dropout) in addition to
        capturing kernels into CUDA graphs. Cost: 1-5s compile latency on
        first call.

        Gating:
          1. ``MLFRAME_TORCH_COMPILE_PREDICT=1`` env var (default off
             until validated on the target host's PyTorch + GPU combo)
          2. CUDA is available + the input tensor lives on CUDA
          3. No LSTM/GRU/RNN in the network (same anti-pattern as the
             compile guard + CUDA-graph gate)
          4. Prior compile attempt has not been cached as failed

        Returns None to signal "not applicable; caller should fall
        through to next path" (vs returning the eager output directly
        which would obscure the cache miss).
        """
        if os.environ.get("MLFRAME_TORCH_COMPILE_PREDICT", "0") != "1":
            return None
        if self._compile_predict_failed:
            return None
        if not torch.cuda.is_available():
            return None
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return None
        _net = getattr(self.network, "_orig_mod", self.network)
        try:
            _recurrent = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
            if any(isinstance(m, _recurrent) for m in _net.modules()):
                return None
        except Exception:
            return None

        if self._compiled_predict_fn is None:
            try:
                # mode="reduce-overhead" enables CUDA graph trees +
                # Inductor fusion. dynamic=False is correct for the
                # cached-shape regime; tail batches that differ in
                # batch size trigger a recompile inside the cache (one-
                # off ~1-2s) which is amortised across the rest of the
                # predict pass.
                self._compiled_predict_fn = torch.compile(
                    self.network, mode="reduce-overhead", dynamic=False,
                )
                logger.info(
                    "torch.compile(mode='reduce-overhead') applied "
                    "to predict forward; first call will pay 1-5s compile "
                    "latency, subsequent calls run on CUDA graph trees + "
                    "Inductor-fused kernels."
                )
            except Exception as _comp_err:
                self._compile_predict_failed = True
                logger.warning(
                    "torch.compile(reduce-overhead) setup failed " "(%s); falling back to eager predict + CUDA-graph " "path (if env-gated).",
                    _comp_err,
                )
                return None
        try:
            return cast(torch.Tensor, self._compiled_predict_fn(x))
        except Exception as _exec_err:
            self._compile_predict_failed = True
            self._compiled_predict_fn = None
            logger.warning(
                "compiled predict forward failed at execution " "(%s); permanently falling back to eager + CUDA-graph path.",
                _exec_err,
            )
            return None

    def _maybe_cuda_graph_forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """CUDA-graph fast path for the inference forward via the LOW-LEVEL
        torch.cuda.CUDAGraph() API.

        Non-destructive: each captured graph is an independent CUDAGraph
        object with its own static input + output buffers. self.network
        and self.network.forward are NEVER mutated, so other shapes can
        still run through eager forward without crashing.

        Per-shape capture flow (cache miss path):
          1. Warmup: 3 eager forwards on a side stream (NVIDIA best
             practice -- lets the caching allocator settle, primes
             cuDNN's algo selection, validates the network is capturable
             for this shape).
          2. Static buffer allocation: ``static_in = empty_like(x)``,
             initial copy ``static_in.copy_(x)``.
          3. Capture: ``with torch.cuda.graph(g): static_out = self.network(static_in)``.
             Records every kernel into ``g``. static_out is the output
             tensor bound to the captured graph's output slot.
          4. Cache: store (g, static_in, static_out) keyed by
             (shape, dtype, device).
        Replay (cache hit):
          1. ``static_in.copy_(x, non_blocking=True)``.
          2. ``g.replay()`` -- single CPU-side launch + GPU runs the
             captured kernel sequence against the new input.
          3. Clone ``static_out`` (next replay overwrites it).

        Gating chain (any False -> eager fallback, zero overhead):
          1. ``MLFRAME_CUDA_GRAPH_PREDICT`` env var:
               "1" / "true" / "on" / "yes" -> ON (opt-in)
               unset / "0" / "false" / "off" -> OFF (default-off after a
               cross-call determinism regression; default-on returned
               stale replay output when Lightning's predict loop reused
               the GPU allocator between successive ``_predict_raw`` calls)
          2. CUDA is available + the input tensor lives on CUDA
          3. No LSTM/GRU/RNN in the network (cuDNN control flow breaks
             capture; same anti-pattern as the torch.compile guard)
          4. Successful capture (any failure caches the False sentinel
             so we don't retry forever)

        Tail-batch behaviour: the LAST batch of a predict pass is
        usually smaller than the trained batch size (drop_last=False on
        the predict dataloader). Its shape misses the cache and triggers
        a fresh capture (~3-5 ms one-off). The capture is NON-destructive
        so the previously-captured graph for the full-size batch still
        works on the NEXT predict pass.

        An earlier make_graphed_callables attempt was REVERTED because that
        API MUTATES self.network.forward (replaces it with the graphed_fn
        specialised to the first capture's shape). The low-level CUDAGraph()
        API leaves the network untouched.
        """
        # Default-on caused predict() and predict_proba() to return divergent
        # values across successive _predict_raw calls (observed max diff ~1.0
        # in [0, 1] sigmoid output, 43% of test rows). Root cause: the captured
        # graph's static buffers can read stale GPU memory after Lightning's
        # predict loop releases its intermediates between calls -- the replay
        # returns the cached output rather than the recomputation over the new
        # input copied into _static_in. Until the cross-call invalidation is
        # solved, default OFF; users with a validated host can opt back in via
        # MLFRAME_CUDA_GRAPH_PREDICT=1.
        _env = os.environ.get("MLFRAME_CUDA_GRAPH_PREDICT", "0").lower()
        if _env in ("0", "false", "off", "no", ""):
            return cast(torch.Tensor, self(x))
        if not torch.cuda.is_available():
            return cast(torch.Tensor, self(x))
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return cast(torch.Tensor, self(x))
        # Skip if the underlying network contains recurrent modules
        # (same anti-pattern as the torch.compile guard).
        _net = getattr(self.network, "_orig_mod", self.network)
        try:
            _recurrent = (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
            if any(isinstance(m, _recurrent) for m in _net.modules()):
                return cast(torch.Tensor, self(x))
        except Exception:
            return cast(torch.Tensor, self(x))

        _key = (tuple(x.shape), x.dtype, x.device)
        _cached = self._cuda_graph_predict_cache.get(_key)
        if _cached is False:
            # Previous capture attempt failed for this shape; permanent
            # eager fallback to avoid retry storms.
            return cast(torch.Tensor, self(x))
        if _cached is not None:
            # Defensive: replay can fail if model state changed
            # (parameters swapped by an EMA callback between predict
            # calls, weight load via load_state_dict, etc.). Wrap in
            # try/except + evict + fall back to eager.
            try:
                _g, _static_in, _static_out = _cached
                _static_in.copy_(x, non_blocking=True)
                _g.replay()
                # Block until the captured kernel sequence finishes
                # writing _static_out before .clone() reads it. Without
                # this sync the replay was racing the host read on
                # subsequent _predict_raw calls, returning stale values.
                torch.cuda.synchronize()
                # Mark most-recently-used (move to the end of the dict) so the LRU-cap evicts cold shapes,
                # not this hot one.
                self._cuda_graph_predict_cache[_key] = self._cuda_graph_predict_cache.pop(_key)
                return cast(torch.Tensor, _static_out.clone())
            except Exception as _replay_err:
                logger.warning(
                    "CUDA-graph replay failed for shape=%s (%s); " "evicting cache entry + falling back to eager.",
                    tuple(x.shape),
                    _replay_err,
                )
                self._cuda_graph_predict_cache.pop(_key, None)
                return cast(torch.Tensor, self(x))

        # First time seeing this shape on this device + dtype. Try a
        # capture via the LOW-LEVEL CUDAGraph() API (non-destructive).
        try:
            # 3 warmup forwards on a side stream. wait_stream chains so
            # the side stream sees the caller's prior work; current
            # stream then waits on the side stream so subsequent ops
            # see the warmup's effects.
            _side_stream = torch.cuda.Stream()
            _side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_side_stream):
                for _ in range(3):
                    _ = self.network(x.clone())
            torch.cuda.current_stream().wait_stream(_side_stream)
            torch.cuda.synchronize()

            # Static input + output buffers. Capture binds them into
            # the graph's input/output slots; replay just copies new x
            # into static_in and the captured kernel sequence writes
            # static_out.
            _static_in = torch.empty_like(x)
            _static_in.copy_(x)
            _g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_g):
                _static_out = self.network(_static_in)

            self._cuda_graph_predict_cache[_key] = (_g, _static_in, _static_out)
            self._evict_cuda_graph_cache_if_needed()
            logger.info(
                "CUDA-graph captured for predict shape=%s dtype=%s.",
                tuple(x.shape), x.dtype,
            )
            # CRITICAL: after capture, ``_static_out`` is bound to the graph's
            # output slot but the kernels have only been RECORDED -- the buffer
            # is uninitialised memory (observed: zeros). Returning it directly
            # on the first call yields garbage predictions for the FIRST batch,
            # dropping aggregate R^2 from 0.998 to 0.659 on a 360-row test split
            # (n=64 per batch, 6 batches). All replays after the first are
            # correct, which is why this bug masquerades as "first-batch random
            # failure" and only surfaces in metrics. Fix: do one replay AFTER
            # capture so _static_out has the actual computed values for this
            # batch, AND the cache is primed for subsequent same-shape calls.
            _g.replay()
            return cast(torch.Tensor, _static_out.clone())
        except Exception as _graph_err:
            self._cuda_graph_predict_cache[_key] = False
            logger.warning(
                "CUDA-graph capture failed for predict shape=%s " "(%s); falling back to eager forward for this shape.",
                tuple(x.shape),
                _graph_err,
            )
            return cast(torch.Tensor, self(x))

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Handle prediction for both (x, y) and x-only batches.

        Returns raw model output (logits for classification, values for regression).
        Softmax/argmax conversion is handled in the estimator's predict methods.
        """
        if self.training:
            logger.warning(f"Model was in training mode during predict_step at batch {batch_idx}. Switching to eval mode.")
            self.eval()

        # Accept both training/testing format (x, y) and prediction format (x only).
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        # torch.inference_mode() replaces torch.no_grad() to be
        # torch.compile-friendly. torch.no_grad() still graph-breaks under
        # TorchDynamo in some PyTorch 2.x versions (the "data-dependent context
        # manager" pattern); torch.inference_mode() is the modern equivalent
        # with cleaner graph capture semantics. The observable behaviour is
        # identical (no grad tracking) for standard tensor ops; inference_mode
        # additionally blocks version-counter mutation which catches a class of
        # user bugs cheaply.
        #
        # Two-tier accelerated predict. Order:
        #   1. torch.compile(mode='reduce-overhead') [most powerful: CUDA
        #      graphs + Inductor fusion] -- gated by MLFRAME_TORCH_COMPILE_PREDICT=1
        #   2. manual CUDA-graph capture [graphs only, no fusion] -- gated by
        #      MLFRAME_CUDA_GRAPH_PREDICT=1
        #   3. Eager fallback (always available)
        # Each gate returns None / falls through when not applicable, so the
        # default behaviour with no env vars set is byte-identical to eager.
        with torch.inference_mode():
            logits = self._maybe_compile_predict_forward(x)
            if logits is None:
                logits = self._maybe_cuda_graph_forward(x)
            # Explicit eager fallback as the documented third tier.
            # ``_maybe_cuda_graph_forward`` normally returns a tensor (it does
            # its OWN eager fallback internally), so this branch is defensive --
            # but it makes the three-tier contract (compile -> graph -> eager)
            # REAL rather than implicit, so any future path that returns None
            # can't crash the downstream ``logits.dim()`` dispatch.
            if logits is None:
                logits = self(x)

        # task_type='regression' returns raw values regardless of shape --
        # including (N, K) multi-target regression where the prior
        # ``logits.shape[1] > 1`` softmax branch would have silently mangled
        # the outputs. Check this FIRST so it short-circuits before any
        # classification-flavoured transform.
        if self.task_type == "regression":
            return logits
        # task_type='multilabel' returns per-label sigmoid (each output independent binary in [0, 1]);
        # task_type='binary' returns sigmoid of 1-output logit -> P(y=1) shape (N, 1);
        # default multi-class K>1 path returns softmax rows that sum to 1.
        if logits.dim() == 2 and logits.shape[1] > 1:
            if self.task_type == "multilabel":
                return torch.sigmoid(logits)
            return torch.softmax(logits, dim=1)
        if self.task_type == "binary":
            # Binary 1-output sigmoid head: return P(y=1) in shape (N, 1).
            # The classifier wrapper stacks [1-p, p] for the (N, 2) contract.
            return torch.sigmoid(logits)
        else:
            # Regression (no task_type tag -- legacy path): return raw values.
            return logits
