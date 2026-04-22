"""Training losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard causal language-modeling loss."""

    shifted_logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        targets.view(-1),
    )
