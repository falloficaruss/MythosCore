import torch

from mythos import MythosConfig, MythosCore


def test_forward_adaptive_depth():
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
        use_moe=True,
        mode="adaptive",
        act_epsilon=0.01,
        act_max_halting_steps=2,
    )
    model = MythosCore(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 9))
    output = model(input_ids)

    assert output.aux_loss is not None
    assert torch.isfinite(output.aux_loss)
    assert config.max_depth <= output.diagnostics.final_depth <= (config.max_depth + config.act_max_halting_steps)
    # The sum of mean halting probabilities might not be exactly 1.0, but it should be close if it halted.
    assert sum(output.diagnostics.halting_probabilities) > 0.0
