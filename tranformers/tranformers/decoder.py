import torch
import torch.nn as nn

from .utils import (
    FeedForward,
    LayerNormalization,
    MultiHeadAttentionBlock,
    ResidualConnection,
)


class DecoderBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, d_ff: int, h: int, dropout: torch.float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_1 = ResidualConnection(dropout)
        self.residual_2 = ResidualConnection(dropout)
        self.residual_3 = ResidualConnection(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            q (torch.Tensor): _description_
            v (torch.Tensor): _description_
            trg_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.residual_1(x, lambda x: self.self_attention(x, x, x, trg_mask))
        x = self.residual_2(x, lambda x: self.cross_attention(x, k, v, src_mask))
        x = self.residual_3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, n: int, d_model: int, d_ff: int, h: int, dropout: torch.float
    ) -> None:
        super().__init__()
        self.decoder_block = nn.ModuleList(
            [DecoderBlock(d_model, d_ff, h, dropout) for _ in range(n)]
        )
        self.norm = LayerNormalization()

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            k (torch.Tensor): _description_
            v (torch.Tensor): _description_
            trg_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        for block in self.decoder_block:
            x = block(x, k, v, trg_mask, src_mask)

        return self.norm(x)
