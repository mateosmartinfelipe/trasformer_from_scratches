import math

import torch
import torch.nn as nn

# TRANSLATION MODEL: ENGLISH TO ITALIAN task


class InputEmbeddings(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, voc_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, max_seq_length: int, dropout: torch.float) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)
        # (max_seq,range)
        self.pe = torch.zeros((max_seq_length, d_model))
        # (max-seq_length,1)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            2.0
            * torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / (d_model))
        )
        self.pe[:, 0::2] = torch.sin(pos * div)
        self.pe[:, 1::2] = torch.cos(pos * div)
        # ( 1, max_length_seq, d_model)
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer("pe", self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # x:  ( batch_size, seq_size, d_model )
        x += self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
