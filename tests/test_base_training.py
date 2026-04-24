import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

import torch

from microchat.checkpoint_manager import load_optimizer_checkpoint, save_checkpoint
from microchat.gpt import GPT, GPTConfig


def load_base_train_module():
    path = os.path.join(os.getcwd(), "scripts", "base_train.py")
    old_argv = sys.argv[:]
    try:
        sys.argv = ["base_train.py"]
        spec = importlib.util.spec_from_file_location("base_train_test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


class BaseTrainingTests(unittest.TestCase):
    def test_cosine_lr_schedule(self):
        module = load_base_train_module()
        module.args.learning_rate = 1e-3
        module.args.min_lr = 1e-4
        module.args.warmup_iters = 2
        module.args.lr_decay_iters = 10

        self.assertAlmostEqual(module.compute_learning_rate(1, 10), 5e-4, places=8)
        self.assertAlmostEqual(module.compute_learning_rate(2, 10), 1e-3, places=8)
        self.assertLess(module.compute_learning_rate(6, 10), 1e-3)
        self.assertAlmostEqual(module.compute_learning_rate(10, 10), 1e-4, places=8)

    def test_checkpoint_saves_optimizer_state(self):
        tmpdir = os.path.join(os.getcwd(), ".test_checkpoint_tmp")
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)
        try:
            model_state = {"w": torch.ones(2)}
            optimizer_state = {"state": {0: {"step": torch.tensor(3)}}}
            save_checkpoint(
                tmpdir,
                7,
                model_state,
                {"step": 7, "model_config": {}, "training_config": {}},
                optimizer_state=optimizer_state,
            )
            restored = load_optimizer_checkpoint(tmpdir, 7, torch.device("cpu"))
            self.assertEqual(restored["state"][0]["step"].item(), 3)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)

    def test_standard_block_disables_experimental_paths(self):
        config = GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            standard_gpt_block=True,
        )
        model = GPT(config)
        self.assertEqual(len(model.value_embeds), 0)
        optimizer = model.setup_optimizer()
        self.assertGreaterEqual(len(optimizer.param_groups), 2)

    def test_sft_optimizer_full_adamw_groups(self):
        config = GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
        )
        model = GPT(config)
        optimizer = model.setup_sft_optimizer(optimizer_type="adamw_full", weight_decay=0.01)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr_scale"], 1.0)
        self.assertEqual(optimizer.param_groups[1]["lr_scale"], 1.0)

    def test_sft_optimizer_behavior_low_lr_groups(self):
        config = GPTConfig(
            sequence_len=32,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
        )
        model = GPT(config)
        optimizer = model.setup_sft_optimizer(
            optimizer_type="adamw_behavior_low_lr",
            weight_decay=0.01,
            behavior_lr_scale=0.2,
        )
        lr_scales = sorted(group["lr_scale"] for group in optimizer.param_groups)
        self.assertEqual(lr_scales, [0.2, 0.2, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
