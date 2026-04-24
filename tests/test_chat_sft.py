import importlib.util
import os
import sys


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
    assert resolved.endswith("identity_conversations.curated.jsonl") or resolved.endswith("identity_conversations.jsonl")
