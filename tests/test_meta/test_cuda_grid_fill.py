"""Launch grids of the shared-memory count kernels fill the device (static model, no GPU).

Two real launches under-filled a 6-SM card and no gate saw it: the u8 joint histogram ran 14 blocks at 100000 rows x 25 columns (achieved occupancy 0.25, 338 us against 122 us once the grid was sized) and the
family null kernel ran 10 blocks at 100 columns on a wider card. This gate models the grid each launcher builds, for the multiprocessor counts of this host (6) and a mid-range card (24), over the shapes of a
default fit, and requires at least ``2 * SMs`` blocks, or every block the shape can supply (a 1024-row floor per histogram block, one warp per column tile).
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters import _cmi_null_strata_gpu as strata
from mlframe.feature_selection.filters import _family_maxt_cuda_bits as fam
from mlframe.feature_selection.filters import _fe_batched_mi_cmi_hist as cmi

_SMS = (6, 24)
_BLOCKS_PER_SM = 2


def _need(sms: int, supply: int) -> int:
    """Blocks the launch must reach: two per multiprocessor, or all the shape can supply."""
    return min(_BLOCKS_PER_SM * sms, supply)


@pytest.mark.parametrize("sms", _SMS)
@pytest.mark.parametrize("n", [30_000, 100_000, 1_000_000])
@pytest.mark.parametrize("k", [8, 25, 100, 400])
def test_the_shared_histogram_grid_fills_the_device(n, k, sms, monkeypatch):
    """Row blocks x column tiles of the u8 joint histogram reach two blocks per SM at 10 x 2 cells (the cells of a binary target over ten levels)."""
    monkeypatch.setattr(cmi, "_SM_COUNT", [sms])
    tile = cmi._hist_smem_tile(20, k, True)
    assert tile, "the 20-cell histogram must fit shared memory"
    col_tiles = -(-k // tile)
    rows = cmi._hist_rows_per_block(n, col_tiles)
    blocks = -(-n // rows) * col_tiles
    assert blocks >= _need(sms, -(-n // 1024) * col_tiles), (n, k, sms, rows, blocks)


@pytest.mark.parametrize("sms", _SMS)
@pytest.mark.parametrize("b,k", [(40, 25), (40, 100), (40, 1000), (99, 200), (20, 60)])
@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("perms_per_block", [1, 4])
def test_the_family_null_grid_fills_the_device(b, k, width, perms_per_block, sms):
    """Permutation blocks x column tiles of the family null kernel, after the block width is narrowed for the device, reach two blocks per SM when the columns allow."""
    other = -(-b // perms_per_block)
    w = fam.fill_block_width(width, k, other, sms)
    assert 32 <= w <= width and w % 32 == 0
    blocks = other * -(-k // w)
    assert blocks >= _need(sms, other * -(-k // 32)), (b, k, width, sms, w, blocks)


@pytest.mark.parametrize("sms", _SMS)
@pytest.mark.parametrize("n_strata", [strata.MIN_STRATA, 300, 1000, 5000, 40000])
@pytest.mark.parametrize("permutations", [25, 99])
def test_the_per_stratum_count_grid_fills_the_device(n_strata, permutations, sms):
    """One block per 128 strata per permutation: even the smallest stratum count the route accepts (``MIN_STRATA``) gives two blocks per SM at a default 25-permutation null."""
    blocks = -(-n_strata // strata._THREADS) * permutations
    assert blocks >= _need(sms, blocks), (n_strata, permutations, sms, blocks)
    assert blocks >= _BLOCKS_PER_SM * sms


def test_the_fill_width_only_narrows_and_keeps_warp_multiples():
    """The width never grows, never drops under 32 and stays a multiple of 32 (a launch shape only: each column is still one thread's work)."""
    for width in (32, 64, 128, 256, 1024):
        for columns in (1, 31, 32, 33, 100, 5000):
            for other in (1, 10, 100):
                for sms in (1, 6, 24, 100):
                    w = fam.fill_block_width(width, columns, other, sms)
                    assert 32 <= w <= max(width, 32) and w % 32 == 0
