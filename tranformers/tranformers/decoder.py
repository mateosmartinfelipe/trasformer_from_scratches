import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        