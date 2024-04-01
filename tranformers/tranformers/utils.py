import math
from typing import Any, Callable, Optional, Tuple

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


class LayerNormalization(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, epsilon: torch.float = 1e-9) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.beta + self.alpha * ((x - mean) / (std + self.epsilon))


class FeedForward(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, d_ff: int, dropout: torch.float) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # x: ( batch_size , seq_size , d_model) --. ( batch_size , seq_size , d_ff)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x: ( batch_size , seq_size , d_ff) --. ( batch_size , seq_size , d_model)
        x = self.layer_2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: torch.float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        dropout: Optional[nn.Dropout],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]
        # ( batch_size,h,seq_length,dk ) * ( batch_size,h,dk,seq_length ) --> ( batch_size,h,seq_length,seq_length)
        attention_scores = torch.einsum("bhsd,bhid -> bhsi", query, key) / math.sqrt(
            d_k
        )
        if mask:
            attention_scores.masked_fill(mask == 0, 1e-17)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout:
            attention_scores = dropout(attention_scores)
        # ( batch_size,h,seq_length,seq_length ) * ( batch_size,h,seq_length,dk ) --> ( batch_size,h,seq_length,dk)
        attention = torch.einsum("bhsi,bhil -> bhsl", attention_scores, value)
        return attention, attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ( batch_size , seq_size , d_model) --> (batch_size ,h, seq_size , d_k)
        query = torch.einsum(
            "bshk -> bhsk", self.w_q(q).view(q.shape[0], q.shape[1], self.h, self.d_k)
        )
        key = torch.einsum(
            "bshk -> bhsk", self.w_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k)
        )
        value = torch.einsum(
            "bshk -> bhsk", self.w_v(v).view(v.shape[0], v.shape[1], self.h, self.d_k)
        )
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # (batch,h,seq_size ,d_k) --> ( batch,seq_size,dim)
        x = (
            torch.einsum("bhsk -> bshk", x)
            .contiguous()
            .view(*x.shape[:2], self.d_k * self.h)
        )
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dropout: torch.float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(
        self, x: torch.Tensor, layer: Callable[[Any], torch.Tensor]
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            layer (Callable[[Any], torch.Tensor]): _description_

        Returns:
            torch.Tensor: _description_
        """
        return x + self.dropout(layer(self.norm(x)))


class ProjectionLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, voc_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(d_model, voc_size)

    def forward(self, x: int) -> torch.Tensor:
        """_summary_

        Args:
            x (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.log_softmax(self.layer(x), dim=-1)
