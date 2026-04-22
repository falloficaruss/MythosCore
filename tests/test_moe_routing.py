import torch

from mythos import MythosConfig, MythosCore


def test_moe_forward_emits_aux_loss_and_stats():
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
        max_depth=2,
        reasoning_slots=5,
        use_moe=True,
        n_experts=4,
        top_k=2,
    )
    model = MythosCore(config)
    output = model(torch.randint(0, config.vocab_size, (2, 9)))

    assert output.aux_loss is not None
    assert torch.isfinite(output.aux_loss)
    assert len(output.diagnostics.moe_stats) == config.max_depth
    assert output.diagnostics.moe_stats[0]["importance"].shape == (config.n_experts,)
