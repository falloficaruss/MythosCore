"""Adaptive compute controller.

This module is intentionally not wired into the forward pass yet. Fixed-depth
recurrence is the first implementation target; ACT-style halting will use this
contract once recurrent behavior is stable.
"""

from __future__ import annotations

import torch
from torch import nn

from mythos.model.config import MythosConfig


class ComputeController(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, semantic: torch.Tensor, reasoning: torch.Tensor) -> torch.Tensor:
        semantic_summary = semantic.mean(dim=1)
        reasoning_summary = reasoning.mean(dim=1)
        features = torch.cat([semantic_summary, reasoning_summary], dim=-1)
        return torch.sigmoid(self.net(features)).squeeze(-1)
