import torch

from mythos import MythosConfig, MythosCore


def test_forward_shapes_fixed_depth():
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
        max_depth=3,
        reasoning_slots=5,
    )
    model = MythosCore(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 11))
    output = model(input_ids)

    assert output.logits.shape == (2, 11, config.vocab_size)
    assert output.reasoning_state.shape == (2, config.reasoning_slots, config.d_model)
    assert len(output.diagnostics.semantic_norms) == config.max_depth
    assert len(output.diagnostics.reasoning_norms) == config.max_depth


def test_forward_uses_requested_depth():
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
        max_depth=4,
        reasoning_slots=5,
    )
    model = MythosCore(config)
    output = model(torch.randint(0, config.vocab_size, (1, 7)), depth=2)

    assert len(output.diagnostics.semantic_norms) == 2
    assert len(output.diagnostics.reasoning_norms) == 2
