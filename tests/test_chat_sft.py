import importlib.util
import os
import sys

import torch

from microchat.tokenizer import RustBPETokenizer


def load_chat_sft_module():
    path = os.path.join(os.getcwd(), "scripts", "chat_sft.py")
    old_argv = sys.argv[:]
    try:
        sys.argv = ["chat_sft.py"]
        spec = importlib.util.spec_from_file_location("chat_sft_test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def test_chat_sft_lr_schedule():
    module = load_chat_sft_module()
    module.args.learning_rate = 2e-5
    module.args.min_lr = 2e-6
    module.args.warmup_iters = 2
    module.args.lr_decay_iters = 10

    assert abs(module.compute_learning_rate(1, 10) - 1e-5) < 1e-12
    assert abs(module.compute_learning_rate(2, 10) - 2e-5) < 1e-12
    assert module.compute_learning_rate(6, 10) < 2e-5
    assert abs(module.compute_learning_rate(10, 10) - 2e-6) < 1e-12


def test_chat_sft_resolve_identity_dataset_prefers_curated_when_local_missing():
    module = load_chat_sft_module()
    base_dir = os.path.join(os.getcwd(), ".microchat")
    resolved = module.resolve_identity_dataset(base_dir)
    assert (
        resolved.endswith("identity_conversations.anchor_en.jsonl")
        or resolved.endswith("identity_conversations.curated.jsonl")
        or resolved.endswith("identity_conversations.jsonl")
    )


def test_chat_sft_builds_phase2_local_mix():
    module = load_chat_sft_module()
    base_dir = os.path.join(os.getcwd(), ".microchat")
    train_tasks, val_tasks, label = module.build_local_sft_tasks(base_dir, repeat_count=2)

    assert len(train_tasks) >= 6
    assert len(val_tasks) >= 2
    assert "anchor_en" in label
    assert "phase2_mix_en" in label


def build_test_tokenizer():
    return RustBPETokenizer.train_from_iterator(
        [
            "System instruction user assistant banana answer hello world final short response",
            "turn one turn two turn three",
        ],
        vocab_size=300,
    )


def test_render_conversation_masks_only_assistant_tokens():
    tokenizer = build_test_tokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "banana"},
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=64)
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    assistant_start_index = ids.index(assistant_start)
    assistant_end_index = ids.index(assistant_end)

    assert all(value == 0 for value in mask[: assistant_start_index + 1])
    assert any(value == 1 for value in mask[assistant_start_index + 1 : assistant_end_index])
    assert mask[assistant_end_index] == 1


def test_render_conversation_formats_system_prompt_and_truncates_to_recent_turn():
    tokenizer = build_test_tokenizer()
    conversation = {
        "messages": [
            {"role": "system", "content": "answer short"},
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "turn two"},
            {"role": "assistant", "content": "final banana"},
        ]
    }

    full_ids, _ = tokenizer.render_conversation(conversation, max_tokens=128)
    full_text = tokenizer.decode(full_ids)
    assert "System instruction:" in full_text

    truncated_ids, truncated_mask = tokenizer.render_conversation(conversation, max_tokens=12)
    truncated_text = tokenizer.decode(truncated_ids)
    assert "final" in truncated_text or "banana" in truncated_text
    assert torch.tensor(truncated_mask).sum().item() > 0


def test_phase2_mix_dataset_loads_with_customjson():
    from tasks.customjson import CustomJSON

    dataset = CustomJSON(filepath=os.path.join(os.getcwd(), "scripts", "chat_phase2_mix_en.jsonl"))
    assert len(dataset) >= 20
    assert dataset[0]["messages"][0]["role"] == "user"
