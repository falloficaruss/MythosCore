"""Initial multi-latent attention interface.

The first implementation is intentionally small: reasoning slots attend to
semantic tokens, semantic tokens attend back to reasoning slots, and the module
returns updated streams. Later work can replace this with full MLA-style KV
compression without touching the recurrent core contract.
"""

from __future__ import annotations

import torch
from torch import nn

from mythos.model.attention import CrossAttention
from mythos.model.config import MythosConfig
from mythos.model.norms import RMSNorm


class MultiLatentAttention(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.reasoning_to_semantic_norm = RMSNorm(config.d_model, config.norm_eps)
        self.semantic_to_reasoning_norm = RMSNorm(config.d_model, config.norm_eps)
        self.reasoning_cross = CrossAttention(config)
        self.semantic_cross = CrossAttention(config)

    def forward(
        self,
        semantic: torch.Tensor,
        reasoning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reasoning = reasoning + self.reasoning_cross(
            self.reasoning_to_semantic_norm(reasoning),
            semantic,
        )
        semantic = semantic + self.semantic_cross(
            self.semantic_to_reasoning_norm(semantic),
            reasoning,
        )
        return semantic, reasoning
