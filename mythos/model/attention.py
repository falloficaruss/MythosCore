"""Attention modules used by MythosCore."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mythos.model.config import MythosConfig
from mythos.model.rope import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE."""

    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_repeats = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(
            seq_len,
            self.head_dim,
            base=self.config.rope_base,
            device=x.device,
            dtype=x.dtype,
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.kv_repeats != 1:
            k = k.repeat_interleave(self.kv_repeats, dim=1)
            v = v.repeat_interleave(self.kv_repeats, dim=1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        attn = attn.masked_fill(~causal_mask[None, None, :, :], torch.finfo(attn.dtype).min)
        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(y)


class CrossAttention(nn.Module):
    """Cross-attention from query states to context states."""

    def __init__(self, config: MythosConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, q_len, dim = query.shape
        k_len = context.size(1)
        q = self.q_proj(query).view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(attn, dim=-1)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, q_len, dim)
        return self.out_proj(y)
