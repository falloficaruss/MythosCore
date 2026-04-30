# MythosCore

<p align="center">
  <strong>Recurrent-depth language models for iterative reasoning</strong><br>
  A research implementation of Mythos-style cognition
</p>

---

## What is MythosCore?

**MythosCore** is an open research implementation of a **recurrent-depth transformer architecture** designed to simulate *iterative reasoning inside the model itself*.

Instead of scaling width (parameters) or context length, MythosCore explores a different axis:

> **Depth as computation** — reasoning emerges from repeated internal refinement

This project is inspired by the closed-weight Mythos model, but focuses on **reconstructing the underlying ideas in an open, hackable form**.

---

## Why This Matters

Most modern LLMs are fundamentally **single-pass systems**:

* One forward pass → one answer
* “Reasoning” is implicit and brittle
* Scaling requires more parameters or more tokens

MythosCore introduces a different paradigm:

| Traditional LLMs        | MythosCore                    |
| ----------------------- | ----------------------------- |
| Single-pass inference   | Multi-step internal reasoning |
| Token-only state        | Semantic + reasoning states   |
| Fixed compute           | Adaptive depth                |
| Chain-of-thought (text) | Latent reasoning (internal)   |

---

## Key Idea: Recurrent Depth

Instead of deeper stacks, MythosCore **reuses the same layers multiple times**, refining its internal state at each step.

```
Input → Prelude → Recurrent Core (×N) → Coda → Output
              ↑
     Reasoning State ↔ Semantic State
```

Each iteration:

* Updates **semantic state** (token representations)
* Evolves **reasoning state** (latent thought vectors)
* Improves the model’s internal understanding

---

## Architecture Breakdown

### 1. Prelude (Encoder)

Standard transformer blocks that initialize semantic representations.

### 2. Recurrent Core (The Engine)

The defining innovation:

* Iterative refinement loop
* Shared weights across depth steps
* Dual-state system:

  * **Semantic state** → token-level meaning
  * **Reasoning state** → abstract reasoning workspace

Supports:

* **Fixed depth** → deterministic compute
* **Adaptive depth** → halting controller decides when to stop

---

### 3. Coda (Decoder)

Maps refined representations back to token logits.

---

## Key Innovations

| Feature                    | Why it matters                                           |
| -------------------------- | -------------------------------------------------------- |
| **Latent Reasoning Slots** | Moves reasoning out of text into structured latent space |
| **Recurrent Depth**        | Enables iterative thinking without parameter explosion   |
| **Adaptive Halting**       | Allocates compute dynamically per input                  |
| **Stable Recurrence**      | Prevents degradation across repeated passes              |
| **Optional MoE**           | Scales capacity without linear compute cost              |

---

## Quick Start

```bash
git clone https://github.com/falloficaruss/MythosCore.git
cd MythosCore
pip install -e ".[dev]"
```

```python
import torch
from mythos import MythosConfig, MythosCore

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

model = MythosCore(config)

input_ids = torch.randint(0, config.vocab_size, (1, 32))
output = model(input_ids, depth=4)
```

---

## Training Example

```python
from torch.optim import AdamW
from mythos.training.losses import next_token_loss

optimizer = AdamW(model.parameters(), lr=1e-4)

for step in range(1000):
    input_ids = torch.randint(0, config.vocab_size, (4, 32))

    optimizer.zero_grad()
    output = model(input_ids, depth=3)

    loss = next_token_loss(output.logits, input_ids)

    if output.aux_loss is not None:
        loss = loss + 0.01 * output.aux_loss

    loss.backward()
    optimizer.step()
```

---

## Configurations

Predefined configs in `configs/`:

* `tiny.yaml` → fast experiments
* `debug.yaml` → development
* `tiny_moe.yaml` → mixture-of-experts

---

## Research Lineage

MythosCore builds on ideas from:

* Universal Transformers
* Adaptive Computation Time
* Mixture of Experts
* Latent reasoning systems
* Chain-of-thought in latent space

---

## Important Disclaimer

This is **not** the original Mythos model.

It is:

* A **speculative reconstruction**
* Built from public research directions
* Intended for **experimentation and learning**

It does **not** include:

* Proprietary weights
* Internal training techniques
* Full-scale optimizations

---

## Where This Leads

This project explores a broader question:

> What if reasoning is not generated… but computed?

Future directions:

* Learned halting policies
* Hierarchical reasoning loops
* Integration with agents + tools
* Self-improving reasoning systems

---

## Contributing

We welcome:

* Architecture experiments
* Training improvements
* Benchmarks & evaluations
* New research ideas

---

## Support

If this project interests you, consider starring the repo and contributing.

---

