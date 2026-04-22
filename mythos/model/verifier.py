"""Verifier head for recurrent-state diagnostics."""

from __future__ import annotations

import torch
from torch import nn

from mythos.model.config import MythosConfig


class VerifierHead(nn.Module):
    """Predicts a scalar confidence/consistency score from hidden state."""

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
        return self.net(features).squeeze(-1)
