"""Transformer blocks."""

from __future__ import annotations

import torch
from torch import nn

from mythos.model.attention import CausalSelfAttention
from mythos.model.config import MythosConfig
from mythos.model.mlp import SwiGLU
from mythos.model.norms import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
