"""Top-level MythosCore model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mythos.model.coda import Coda
from mythos.model.config import MythosConfig
from mythos.model.norms import RMSNorm
from mythos.model.prelude import Prelude
from mythos.model.recurrent_core import RecurrentCore, RecurrentDiagnostics


@dataclass
class MythosOutput:
    logits: torch.Tensor
    reasoning_state: torch.Tensor
    diagnostics: RecurrentDiagnostics
    aux_loss: torch.Tensor | None = None


class MythosCore(nn.Module):
    """Prelude -> recurrent core -> coda language model."""

    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.reasoning_seed = nn.Parameter(torch.randn(config.reasoning_slots, config.d_model) * 0.02)
        self.prelude = Prelude(config)
        self.recurrent_core = RecurrentCore(config)
        self.coda = Coda(config)
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, *, depth: int | None = None) -> MythosOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        encoded_input = self.prelude(x)
        semantic = encoded_input
        reasoning = self.reasoning_seed.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        semantic, reasoning, diagnostics = self.recurrent_core(
            semantic,
            reasoning,
            encoded_input,
            depth=depth,
        )
        decoded = self.coda(semantic)
        logits = self.lm_head(self.final_norm(decoded))

        aux_loss = None
        if diagnostics.moe_aux_losses:
            aux_loss = torch.stack(diagnostics.moe_aux_losses).mean()
        return MythosOutput(
            logits=logits,
            reasoning_state=reasoning,
            diagnostics=diagnostics,
            aux_loss=aux_loss,
        )
