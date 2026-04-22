"""Depth evaluation helpers."""

from __future__ import annotations

import torch

from mythos.model.mythos_core import MythosCore
from mythos.training.losses import next_token_loss


@torch.no_grad()
def loss_by_depth(model: MythosCore, input_ids: torch.Tensor, depths: list[int]) -> dict[int, float]:
    model.eval()
    results: dict[int, float] = {}
    for depth in depths:
        output = model(input_ids, depth=depth)
        results[depth] = float(next_token_loss(output.logits, input_ids).detach().cpu())
    return results
