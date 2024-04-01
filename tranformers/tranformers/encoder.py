import torch
import torch.nn as nn

from .utils import (
    FeedForward,
    LayerNormalization,
    MultiHeadAttentionBlock,
    ResidualConnection,
)


class EncoderBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, d_ff: int, h: int, dropout: torch.float) -> None:
        super().__init__()
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.multihead = MultiHeadAttentionBlock(d_model, h, dropout)
        self.residual_1 = ResidualConnection(dropout)
        self.residual_2 = ResidualConnection(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.residual_1(x, lambda x: self.multihead(x, x, x, src_mask))
        return self.residual_2(x, self.feed_forward)


class Encoder(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        n: int,
        d_model: int,
        d_ff: int,
        h: int,
        dropout: torch.float,
    ) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, d_ff, h, dropout) for _ in range(n)]
        )
        # This one is missed in the picture
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            src_mask (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        for block in self.encoder_blocks:
            x = block(x, src_mask)
        return self.norm(x)
