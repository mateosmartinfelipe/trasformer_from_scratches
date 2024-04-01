import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .utils import InputEmbeddings, PositionalEncoding, ProjectionLayer


class Transformer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        n: int,
        d_model,
        d_ff: int,
        h: int,
        src_seq_length: int,
        trg_seq_length: int,
        src_voc_size: int,
        trg_voc_size: int,
        dropout: torch.float,
    ) -> None:
        super().__init__()
        self.src_input = InputEmbeddings(d_model, src_voc_size)
        self.trg_input = InputEmbeddings(d_model, trg_voc_size)
        self.pe_src = PositionalEncoding(d_model, src_seq_length, dropout)
        self.pe_trg = PositionalEncoding(d_model, trg_seq_length, dropout)
        self.encoder = Encoder(n, d_model, d_ff, h, dropout)
        self.decoder = Decoder(n, d_model, d_ff, h, dropout)
        self.proj = ProjectionLayer(d_model, trg_voc_size)

    def encode(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            src_mask (torch.Tensor): _description_
            trg_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.src_input(x)
        x = self.pe_src(x)
        return self.encoder(x, src_mask)

    def decode_proj(
        self,
        o: torch.Tensor,
        y: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            src_mask (torch.Tensor): _description_
            trg_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        y = self.trg_input(y)
        y = self.pe_trg(y)
        y = self.decoder(y, o, o, src_mask, trg_mask)
        y = self.proj(y)
        return y

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            src_mask (torch.Tensor): _description_
            trg_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        o = self.encode(x, src_mask)
        return self.decode_proj(o, y, src_mask, trg_mask)


def build_model(
    *,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    src_seq_length: int,
    trg_seq_length: int,
    src_voc_size: int,
    trg_voc_size: int,
    dropout: torch.float = 0.1
) -> Transformer:
    """_summary_

    Args:
        src_seq_length (int): _description_
        trg_seq_length (int): _description_
        src_voc_size (int): _description_
        trg_voc_size (int): _description_
        n (int, optional): _description_. Defaults to 6.
        d_model (int, optional): _description_. Defaults to 512.
        d_ff (int, optional): _description_. Defaults to 2048.
        h (int, optional): _description_. Defaults to 8.
        dropout (torch.float, optional): _description_. Defaults to 0.1.

    Returns:
        Transformer: _description_
    """
    return Transformer(
        n,
        d_model,
        d_ff,
        h,
        src_seq_length,
        trg_seq_length,
        src_voc_size,
        trg_voc_size,
        dropout,
    )
