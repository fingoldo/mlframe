"""The access inventory classifies column reads, row reads and loop-order scalar reads of a matrix parameter."""

from pathlib import Path

from mlframe.feature_selection._benchmarks.profiling.codes_matrix_access_inventory import inventory, raw_kernel_files, render

SRC = '''
import numpy as np
from numba import njit, prange


@njit
def column_reader(data, j):
    return data[:, j].sum()


@njit
def rows_inner(factors_data, a):
    s = 0
    for i in range(factors_data.shape[0]):
        s += factors_data[i, a]
    return s


@njit
def cols_inner(factors_data, i):
    s = 0
    for j in range(factors_data.shape[1]):
        s += factors_data[i, j]
    return s


@njit
def row_reader(data, i):
    return data[i, :].sum()


def vector_reader(codes, i):
    return codes[i]
'''


def test_classification_by_access_pattern(tmp_path: Path):
    """Each reader lands on the side its loop order serves; a single index is not counted as a matrix row."""
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    (tmp_path / "k.py").write_text('KERNEL = """extern "C" __global__ void f() {}"""\n', encoding="utf-8")
    recs = {r["function"]: r for r in inventory(tmp_path)}
    assert recs["column_reader"]["leans"] == "feature-major" and recs["column_reader"]["column"] == 1
    assert recs["rows_inner"]["leans"] == "feature-major" and recs["rows_inner"]["rows_inner"] == 1
    assert recs["cols_inner"]["leans"] == "row-major" and recs["cols_inner"]["cols_inner"] == 1
    assert recs["row_reader"]["leans"] == "row-major" and recs["row_reader"]["row"] == 1
    assert recs["vector_reader"]["leans"] == "neutral" and recs["vector_reader"]["other"] == 1
    assert all(r["kind"] == "njit" for f, r in recs.items() if f != "vector_reader") and recs["vector_reader"]["kind"] == "python"
    assert raw_kernel_files(tmp_path) == ["k.py"]
    text = render(list(recs.values()), ["k.py"])
    assert "column_reader" in text and "k.py" in text
