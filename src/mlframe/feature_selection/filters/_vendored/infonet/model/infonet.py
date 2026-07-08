"""Vendored InfoNet model: a transformer-based amortized mutual-information estimator (encoder/decoder + attention query generator) that maps a batch of paired samples directly to an MI lower-bound estimate."""

from typing import Optional

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .query import Query_Gen_transformer
from .gauss_mild import GaussConv
from .util import mutual_information
#from .query import Query_Gen

class infonet(nn.Module):
    """Amortized MI estimator: encodes the input pair samples into latents, generates attention queries from the same input, decodes a smoothed density-ratio surface via a Gaussian-blur conv, then reduces it to a scalar MI lower bound."""

    def __init__(self, encoder: Encoder, decoder: Decoder, query_gen: Query_Gen_transformer, decoder_query_dim: int):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.query_gen = query_gen
        self.mild = GaussConv(size=15, nsig=3, channels=1)

        # self.query = nn.Parameter(torch.randn(1, decoder_query_dim, decoder_query_dim))

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ):
        """Encode ``inputs`` into latents, decode a query-conditioned output surface, smooth it with a small Gaussian kernel, and return the resulting MI lower-bound estimate via ``mutual_information``."""
        latents = self.encoder(inputs, input_mask)
        query = self.query_gen(inputs)

        outputs = self.decoder(x_q=query, latents=latents, query_mask=query_mask)
        # print(outputs.shape)
        # torch.save(outputs.cpu().numpy(), "lookuptable.pth")
        outputs = outputs.unsqueeze(1)
        outputs = self.mild(outputs)
        outputs = outputs.squeeze(1)
        # torch.save(outputs.cpu().numpy(), "lookuptable_smoothed.pth")
        # print("saved!!!!!!!!!!!!!!!1")
        mi_lb = mutual_information(inputs, outputs)

        return mi_lb
