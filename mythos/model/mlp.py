"""Feed-forward layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mythos.model.config import MythosConfig


class SwiGLU(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
