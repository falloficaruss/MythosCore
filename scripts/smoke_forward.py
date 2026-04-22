"""Run a deterministic MythosCore forward-pass smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mythos import MythosConfig, MythosCore


def load_config(path: Path) -> MythosConfig:
    with path.open("r", encoding="utf-8") as handle:
        return MythosConfig(**yaml.safe_load(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(7)
    config = load_config(args.config)
    model = MythosCore(config)
    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq_len))
    output = model(input_ids, depth=args.depth)

    print(f"logits={tuple(output.logits.shape)}")
    print(f"reasoning={tuple(output.reasoning_state.shape)}")
    print(f"semantic_norms={output.diagnostics.semantic_norms}")
    print(f"reasoning_norms={output.diagnostics.reasoning_norms}")
    if output.aux_loss is not None:
        print(f"aux_loss={float(output.aux_loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
