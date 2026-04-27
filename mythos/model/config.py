"""Configuration objects for MythosCore models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MythosConfig:
    """Architecture configuration for the speculative recurrent model."""

    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1024
    prelude_layers: int = 2
    recurrent_layers: int = 1
    coda_layers: int = 2
    max_depth: int = 8
    reasoning_slots: int = 16
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    recurrent_state_gate_init: float = 0.85
    encoded_input_gate_init: float = 0.15
    use_moe: bool = False
    n_experts: int = 4
    top_k: int = 2
    mode: str = "fixed"
    act_epsilon: float = 0.01
    act_max_halting_steps: int = 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.reasoning_slots < 1:
            raise ValueError("reasoning_slots must be at least 1")
        if self.mode not in {"fixed", "adaptive"}:
            raise ValueError("mode must be 'fixed' or 'adaptive'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
