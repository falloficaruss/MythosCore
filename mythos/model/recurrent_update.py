"""Stable recurrent state update."""

from __future__ import annotations

import math

import torch
from torch import nn

from mythos.model.config import MythosConfig


def _logit(prob: float) -> float:
    prob = min(max(prob, 1e-4), 1.0 - 1e-4)
    return math.log(prob / (1.0 - prob))


class StableRecurrentUpdate(nn.Module):
    """Gated recurrent injection: A(h) + B(e) + proposal."""

    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.state_gate_logit = nn.Parameter(
            torch.full((config.d_model,), _logit(config.recurrent_state_gate_init))
        )
        self.input_gate_logit = nn.Parameter(
            torch.full((config.d_model,), _logit(config.encoded_input_gate_init))
        )
        self.proposal_gate_logit = nn.Parameter(torch.full((config.d_model,), _logit(0.5)))
        self.input_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.proposal_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        previous: torch.Tensor,
        encoded_input: torch.Tensor,
        proposal: torch.Tensor,
    ) -> torch.Tensor:
        state_gate = torch.sigmoid(self.state_gate_logit)
        input_gate = torch.sigmoid(self.input_gate_logit)
        proposal_gate = torch.sigmoid(self.proposal_gate_logit)
        return (
            previous * state_gate
            + self.input_proj(encoded_input) * input_gate
            + self.proposal_proj(proposal) * proposal_gate
        )

    def gate_values(self) -> dict[str, float]:
        return {
            "state": float(torch.sigmoid(self.state_gate_logit).mean().detach().cpu()),
            "input": float(torch.sigmoid(self.input_gate_logit).mean().detach().cpu()),
            "proposal": float(torch.sigmoid(self.proposal_gate_logit).mean().detach().cpu()),
        }
