"""PositionalEncoding extends its table for a sequence longer than max_len instead of failing on a shape mismatch."""
import torch

from mlframe.training.neural._recurrent_arch import PositionalEncoding


def test_sequence_longer_than_the_table_is_encoded_with_the_same_formula():
    """max_len 16 and a 20-step input: the first 16 positions equal the stored table, the rest continue the sinusoid."""
    pe = PositionalEncoding(8, max_len=16, dropout=0.0).eval()
    out = pe(torch.zeros(2, 20, 8))
    assert out.shape == (2, 20, 8)
    assert torch.allclose(out[0, :16], pe.pe[0, :16])
    assert torch.allclose(out[0, 16:], PositionalEncoding._sinusoid_table(20, 8)[16:])
