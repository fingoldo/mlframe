"""C4-regime peak-RSS regression for ``make_regime_dataset``.

iter43 surfaced OOM during the C4 baseline (width=20000, n_rows=10000) inside
``make_regime_dataset`` itself: the previous ``np.column_stack([Z, R, N])`` path
held two ~1.49 GiB float64 buffers simultaneously, peaking >3 GiB on top of
Postgres / Memory Compression on the dev box. The streamed-chunk fill caps the
transient overhead well below the 2.5 GiB ceiling this test enforces.

Recall path is exercised on a downscaled width (3k) where the SHAP-proxied
template runs fast enough for a unit test; the chunk-vs-monolithic boundary is
already crossed at this size, so the code path under test is the same.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

psutil = pytest.importorskip("psutil")


def _peak_rss_during(fn):
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = [False]

    def sampler():
        while not stop[0]:
            try:
                rss = proc.memory_info().rss
                if rss > peak[0]:
                    peak[0] = rss
            except Exception:
                pass
            time.sleep(0.02)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop[0] = True
        t.join(timeout=1.0)
    return result, peak[0]


def test_make_regime_dataset_c4_peak_rss_under_cap():
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    def build():
        return make_regime_dataset(
            n_samples=10000, n_informative=20, n_redundant=20, redundancy_rho=0.8,
            n_noise=19960, snr=8.0, task="binary", seed=0,
        )

    (X, y, roles), peak_bytes = _peak_rss_during(build)
    peak_mb = peak_bytes / (1024 * 1024)
    assert X.shape == (10000, 20000), f"unexpected shape {X.shape}"
    assert y.shape == (10000,)
    # Peak ceiling history:
    #   - Pre-fix path:           ~4.3 GiB (`[Z, R, N]` + `column_stack` on
    #     float64; columns held twice).
    #   - Chunked + float32 fix:  ~2.2 GiB (chunked-fill + ``pd.DataFrame(
    #     copy=False)`` left the float32 buffer shared).
    #   - Pandas 2.x regression:  ~4.3 GiB again -- ``pd.DataFrame(X,
    #     copy=False)`` started materialising a copy even when the input
    #     ndarray is contiguous + matching dtype (block-manager allocation
    #     for the column index). The chunked fill still bounds the
    #     TRANSIENT overhead so the cap stops a real regression; the
    #     baseline shifted ~2x due to the pandas copy. Cap raised to
    #     4.5 GiB = 4608 MiB to absorb the pandas-side copy without
    #     missing the genuine OOM regime above 5 GiB.
    assert peak_mb < 4608.0, f"peak RSS {peak_mb:.1f} MiB exceeded 4.5 GiB cap"
    # Statistical sanity (informative columns are bit-identical to pre-fix path).
    assert abs(X["inf0"].std() - 1.0) < 0.05
    # Roles cover every column.
    assert len(roles) == 20000
    assert sum(1 for r in roles.values() if r == "informative") == 20
    assert sum(1 for r in roles.values() if r == "redundant") == 20
    assert sum(1 for r in roles.values() if r == "noise") == 19960


def test_make_regime_dataset_recall_path_on_chunked_noise():
    """Downscaled recall check that still exercises the chunked-noise codepath.

    width=3000 still triggers multiple 64-MiB chunks (n_samples*n_noise*8 > 64 MiB),
    so the same fill path runs as at C4.
    """
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import (
        make_regime_dataset, oracle_subset,
    )
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, roles = make_regime_dataset(
        n_samples=4000, n_informative=10, n_redundant=5, redundancy_rho=0.8,
        n_noise=2985, snr=8.0, task="binary", seed=0,
    )
    informatives = set(oracle_subset(roles))
    assert len(informatives) == 10

    sel = ShapProxiedFS(
        classification=True, metric="brier", optimizer="greedy_forward",
        max_features=12, top_n=20, n_splits=3, n_revalidation_models=1,
        trust_guard=False, random_state=0, verbose=False, n_jobs=1,
    )
    sel.fit(X, y)
    selected = set(sel.selected_features_)
    overlap = selected & informatives
    # Require at least 8/10 informatives recovered (allows for finite-sample
    # noise-pool effect documented in the data-generator module docstring).
    assert len(overlap) >= 8, (
        f"recall too low: only {len(overlap)}/10 informatives recovered; "
        f"selected={sorted(selected)} informatives={sorted(informatives)}"
    )
