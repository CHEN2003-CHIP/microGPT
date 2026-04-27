import torch


def test_gpt_compatibility_imports_match_model_package():
    from microchat.gpt import GPT as CompatGPT
    from microchat.gpt import GPTConfig as CompatGPTConfig
    from microchat.model import GPT, GPTConfig

    assert CompatGPT is GPT
    assert CompatGPTConfig is GPTConfig


def test_tiny_gpt_forward_loss_and_key_structure():
    from microchat.gpt import GPT, GPTConfig

    torch.manual_seed(1234)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
    )
    model = GPT(config)
    model.init_weights()
    model.eval()

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits = model(idx)
        loss = model(idx, targets)

    assert list(logits.shape) == [2, 8, 128]
    assert torch.isfinite(loss)
    state_keys = list(model.state_dict().keys())
    parameter_keys = list(dict(model.named_parameters()).keys())
    assert state_keys == parameter_keys
    assert state_keys[:4] == [
        "resid_lambdas",
        "x0_lambdas",
        "smear_lambda",
        "backout_lambda",
    ]
    assert "transformer.wte.weight" in state_keys
    assert "transformer.h.0.attn.c_q.weight" in state_keys
    assert "transformer.h.0.mlp.c_fc.weight" in state_keys
    assert "value_embeds.1.weight" in state_keys
    assert "lm_head.weight" in state_keys
