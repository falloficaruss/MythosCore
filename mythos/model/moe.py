"""Sparse Mixture-of-Experts layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mythos.model.config import MythosConfig
from mythos.model.mlp import SwiGLU


class TopKMoE(nn.Module):
    """Simple top-k MoE suitable for early research experiments."""

    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        if config.top_k > config.n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(config) for _ in range(config.n_experts))
        self.shared_expert = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        router_logits = self.router(x)
        route_probs = F.softmax(router_logits, dim=-1)
        top_values, top_indices = torch.topk(route_probs, self.top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        gathered = torch.gather(
            expert_outputs,
            dim=-2,
            index=top_indices.unsqueeze(-1).expand(*top_indices.shape, x.size(-1)),
        )
        routed = (gathered * top_values.unsqueeze(-1)).sum(dim=-2)
        output = routed + self.shared_expert(x)

        importance = route_probs.mean(dim=(0, 1))
        load = F.one_hot(top_indices, num_classes=self.n_experts).float().mean(dim=(0, 1, 2))
        aux_loss = self.n_experts * torch.sum(importance * load)
        stats = {
            "importance": importance.detach(),
            "load": load.detach(),
            "router_entropy": (-(route_probs * route_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().detach(),
        }
        return output, aux_loss, stats
