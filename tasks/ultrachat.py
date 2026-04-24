"""
UltraChat 200k by HuggingFaceH4.
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
We use the SFT splits because they are already filtered for supervised fine-tuning.
"""

from microchat.common import configure_local_hf_cache

HF_CACHE_DIR = configure_local_hf_cache()

from datasets import load_dataset
from tasks.common import Task


class UltraChat(Task):
    """UltraChat 200k supervised fine-tuning split."""

    SPLIT_MAP = {
        "train": "train_sft",
        "test": "test_sft",
    }

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in self.SPLIT_MAP, "UltraChat split must be train|test"
        hf_split = self.SPLIT_MAP[split]
        self.ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=hf_split, cache_dir=HF_CACHE_DIR).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        assert len(messages) >= 2, "UltraChat messages must have at least 2 messages"

        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages

        assert len(rest_messages) >= 2, "UltraChat conversation must contain at least one user/assistant pair"
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"

        return {"messages": messages}
