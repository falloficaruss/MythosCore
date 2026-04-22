import torch

from mythos import MythosConfig, MythosCore


def test_recurrent_norms_remain_finite():
    config = MythosConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        prelude_layers=1,
        recurrent_layers=1,
        coda_layers=1,
        max_depth=6,
        reasoning_slots=5,
    )
    model = MythosCore(config)
    output = model(torch.randint(0, config.vocab_size, (2, 13)))

    assert all(torch.isfinite(torch.tensor(output.diagnostics.semantic_norms)))
    assert all(torch.isfinite(torch.tensor(output.diagnostics.reasoning_norms)))
    assert max(output.diagnostics.semantic_norms) < 100.0
    assert max(output.diagnostics.reasoning_norms) < 100.0
