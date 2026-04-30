# MythosCore

<p align="center">
  <a href="https://pypi.org/project/mythos-core/">
    <img src="https://img.shields.io/pypi/v/mythos-core?color=success" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/mythos-core/">
    <img src="https://img.shields.io/pypi/pyversions/mythos-core" alt="Python">
  </a>
  <a href="https://github.com/anomalyco/mythoscore/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/mythos-core" alt="License">
  </a>
</p>

> Speculative recurrent-depth architecture — an open implementation exploring Mythos-style reasoning

MythosCore is an open research implementation of a **speculative recurrent-depth architecture** inspired by the closed-weight Mythos model by Anthropic. While Mythos represents one of the most powerful reasoning systems released to date, its weights remain private. This project attempts to reconstruct and explore the underlying architectural innovations.

## Why MythosCore?

The Mythos model by Anthropic demonstrated unprecedented reasoning capabilities through a novel recurrent depth architecture. By keeping weights closed, the community can only speculate about the architectural choices that enable such reasoning.

MythosCore is our attempt to:
- 📖 **Open the black box** — Make speculative architecture accessible for study
- 🔬 **Enable research** — Provide a testbed for architecture experiments
- 🚀 **Empower experimentation** — Let anyone train their own reasoning model

## Installation

Install from source:

```bash
git clone https://github.com/falloficaruss/MythosCore.git
cd MythosCore
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from mythos import MythosConfig, MythosCore

# Configure your model
config = MythosConfig(
    vocab_size=32000,
    max_seq_len=512,
    d_model=256,
    n_heads=8,
    n_kv_heads=4,
    d_ff=1024,
    prelude_layers=2,
    recurrent_layers=1,
    coda_layers=2,
    max_depth=8,
    reasoning_slots=16,
)

# Initialize the model
model = MythosCore(config)

# Forward pass with custom depth
input_ids = torch.randint(0, config.vocab_size, (1, 32))
output = model(input_ids, depth=4)

print(f"Logits shape: {output.logits.shape}")
print(f"Reasoning state shape: {output.reasoning_state.shape}")
print(f"Final depth: {output.diagnostics.final_depth}")
```

## Architecture Overview

MythosCore implements a **recurrent-depth language model** with three key components:

```
Input → Prelude → Recurrent Core (×N) → Coda → Output
              ↑
     Reasoning State ↔ Semantic State
```

### 1. Prelude Encoder
Encodes input tokens into semantic representations using standard transformer blocks.

### 2. Recurrent Core
The heart of the architecture — a **speculative depth mechanism** where:
- **Semantic state** carries token representations
- **Reasoning state** maintains latent reasoning slots
- Each depth step allows the model to iteratively refine its understanding

Two modes:
- **Fixed depth**: Computes exactly `N` steps
- **Adaptive depth**: Uses a halting controller to decide when to stop

### 3. Coda Decoder
Transforms semantic representations back to token logits for next-token prediction.

### Key Innovations

| Feature | Description |
|---------|-------------|
| **Latent Reasoning Slots** | Dedicated state vectors for reasoning, separate from semantic tokens |
| **Depth Modulation** | Both fixed and adaptive compute modes |
| **Stable Recurrence** | Gated updates to prevent gradient instability |
| **Mixture of Experts** | Optional MoE layers for expert routing |

## Training Your Own Model

```python
import torch
from torch.optim import AdamW
from mythos import MythosConfig, MythosCore
from mythos.training.losses import next_token_loss

# Configuration
config = MythosConfig(
    vocab_size=8000,
    max_seq_len=128,
    d_model=128,
    n_heads=4,
    n_kv_heads=2,
    d_ff=512,
    prelude_layers=1,
    recurrent_layers=1,
    coda_layers=1,
    max_depth=4,
    reasoning_slots=8,
)

model = MythosCore(config)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
model.train()
for step in range(1000):
    # Dummy training batch
    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    
    optimizer.zero_grad()
    output = model(input_ids, depth=3)
    
    # Next-token prediction loss
    loss = next_token_loss(output.logits, input_ids)
    
    # Add auxiliary losses (MoE balancing, etc.)
    if output.aux_loss is not None:
        loss = loss + 0.01 * output.aux_loss
    
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

## Predefined Configurations

Ready-to-use configs in `configs/`:

| Config | Use Case |
|--------|---------|
| `configs/tiny.yaml` | Quick experiments, low memory |
| `configs/debug.yaml` | Debugging, development |
| `configs/tiny_moe.yaml` | With Mixture of Experts |

Load a config:

```python
import yaml
from mythos import MythosConfig, MythosCore

with open("configs/tiny.yaml") as f:
    config_dict = yaml.safe_load(f)

config = MythosConfig(**config_dict)
model = MythosCore(config)
```

## Research Foundations

This implementation draws from multiple lines of academic research:

- **Universal Transformers** — Recurrent depth in transformers
- **Adaptive Computation Time** — Dynamic stopping criteria
- **Mixture of Experts** — Conditional computation routing
- **Latent Reasoning** — Continuous latent thought spaces
- **COCOnut** — Chain-of-thought in latent space

See `papers/README.md` for the full reference list.

## Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_forward_shapes.py
```

## Contributing

Open for contributions! Key areas:
- Architecture variants and experiments
- Training optimizations
- Evaluation benchmarks
- Documentation improvements

## Limitations

⚠️ **Important**: This is NOT an exact replica of the closed Anthropic Mythos model. It is:
- A **speculative reconstruction** based on public architecture descriptions
- Intended for **research and educational** purposes
- Missing the pretrained weights that make Mythos powerful

## Citation

```bibtex
@software{mythoscore,
  title = {MythosCore},
  author = {MythosCore Contributors},
  year = {2026},
  url = {https://github.com/anomalyco/mythoscore},
}
```

---

<p align="center">
  <strong>Interested in pushing the boundaries of reasoning models?</strong><br>
  Star us on GitHub and join the discussion!
</p>
