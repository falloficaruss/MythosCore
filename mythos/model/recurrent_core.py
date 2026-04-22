"""Recurrent core for MythosCore."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mythos.model.attention import CausalSelfAttention
from mythos.model.config import MythosConfig
from mythos.model.mlp import SwiGLU
from mythos.model.moe import TopKMoE
from mythos.model.multi_latent_attention import MultiLatentAttention
from mythos.model.norms import RMSNorm
from mythos.model.recurrent_update import StableRecurrentUpdate


@dataclass
class RecurrentDiagnostics:
    semantic_norms: list[float]
    reasoning_norms: list[float]
    update_gates: list[dict[str, float]]
    moe_aux_losses: list[torch.Tensor]
    moe_stats: list[dict[str, torch.Tensor]]


class RecurrentLayer(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.config = config
        self.depth_embedding = nn.Embedding(config.max_depth, config.d_model)
        self.semantic_norm = RMSNorm(config.d_model, config.norm_eps)
        self.semantic_attn = CausalSelfAttention(config)
        self.latent_attn = MultiLatentAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = TopKMoE(config) if config.use_moe else SwiGLU(config)
        self.semantic_update = StableRecurrentUpdate(config)
        self.reasoning_update = StableRecurrentUpdate(config)

    def forward(
        self,
        semantic: torch.Tensor,
        reasoning: torch.Tensor,
        encoded_input: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        depth_id = torch.full((semantic.size(0),), depth, device=semantic.device, dtype=torch.long)
        depth_bias = self.depth_embedding(depth_id).unsqueeze(1)

        semantic_work = semantic + depth_bias
        proposal = semantic_work + self.semantic_attn(self.semantic_norm(semantic_work))
        semantic_proposal, reasoning_proposal = self.latent_attn(proposal, reasoning + depth_bias)

        aux_loss = None
        moe_stats: dict[str, torch.Tensor] = {}
        ffn_input = self.ffn_norm(semantic_proposal)
        if self.config.use_moe:
            ffn_output, aux_loss, moe_stats = self.ffn(ffn_input)
        else:
            ffn_output = self.ffn(ffn_input)
        semantic_proposal = semantic_proposal + ffn_output

        semantic_next = self.semantic_update(semantic, encoded_input, semantic_proposal)
        reasoning_context = encoded_input.mean(dim=1, keepdim=True).expand(-1, reasoning.size(1), -1)
        reasoning_next = self.reasoning_update(reasoning, reasoning_context, reasoning_proposal)
        return semantic_next, reasoning_next, aux_loss, moe_stats


class RecurrentCore(nn.Module):
    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(RecurrentLayer(config) for _ in range(config.recurrent_layers))

    def forward(
        self,
        semantic: torch.Tensor,
        reasoning: torch.Tensor,
        encoded_input: torch.Tensor,
        *,
        depth: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentDiagnostics]:
        steps = depth or self.config.max_depth
        diagnostics = RecurrentDiagnostics([], [], [], [], [])
        for i in range(steps):
            depth_index = min(i, self.config.max_depth - 1)
            for layer in self.layers:
                semantic, reasoning, aux_loss, moe_stats = layer(
                    semantic,
                    reasoning,
                    encoded_input,
                    depth_index,
                )
                diagnostics.update_gates.append(layer.semantic_update.gate_values())
                if aux_loss is not None:
                    diagnostics.moe_aux_losses.append(aux_loss)
                    diagnostics.moe_stats.append(moe_stats)
            diagnostics.semantic_norms.append(float(semantic.norm(dim=-1).mean().detach().cpu()))
            diagnostics.reasoning_norms.append(float(reasoning.norm(dim=-1).mean().detach().cpu()))
        return semantic, reasoning, diagnostics
