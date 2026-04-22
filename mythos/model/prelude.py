"""Prelude encoder for MythosCore."""

from __future__ import annotations

import torch
from torch import nn

from mythos.model.block import TransformerBlock
from mythos.model.config import MythosConfig


class Prelude(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.prelude_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
