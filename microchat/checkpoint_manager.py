"""Minimal checkpoint helpers for the microGPT teaching project."""

from __future__ import annotations

import glob
import json
import os

import torch

from microchat.common import get_base_dir
from microchat.gpt import GPT, GPTConfig
from microchat.tokenizer import get_tokenizer


def save_checkpoint(checkpoint_dir, step, model_state, metadata, optimizer_state=None):
    """Save a checkpoint consisting of the model state and associated metadata (e.g. config, training step)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    torch.save(model_state, model_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if optimizer_state is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{step:06d}.pt")
        torch.save(optimizer_state, optimizer_path)
    return model_path, meta_path


def load_checkpoint(checkpoint_dir, step, device):
    """Load a checkpoint by step number, returning the model state and associated metadata."""
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    model_state = torch.load(model_path, map_location=device)
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model_state, metadata


def load_optimizer_checkpoint(checkpoint_dir, step, device):
    """Load optimizer state for a training checkpoint, if it exists."""
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{step:06d}.pt")
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"No optimizer state found at {optimizer_path}")
    return torch.load(optimizer_path, map_location=device)


def find_last_step(checkpoint_dir):
    """Find the last training step in the checkpoint directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return int(max(os.path.basename(path).split("_")[-1].split(".")[0] for path in checkpoint_files))


def find_model_tag(checkpoints_dir, requested_tag=None):
    """Find the model tag to load. If a specific tag is requested, return it if it exists. Otherwise, find the tag with the highest depth (dNNN) or the most recently modified tag."""
    if requested_tag is not None:
        return requested_tag
    model_tags = [name for name in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, name))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    depth_tags = []
    for tag in model_tags:
        if tag.startswith("d") and tag[1:].isdigit():
            depth_tags.append((int(tag[1:]), tag))
    if depth_tags:
        depth_tags.sort(reverse=True)
        return depth_tags[0][1]
    model_tags.sort(key=lambda name: os.path.getmtime(os.path.join(checkpoints_dir, name)), reverse=True)
    return model_tags[0]


def build_model(checkpoint_dir, step, device, phase):
    """Build the model by loading the checkpoint and applying the model state to a new model instance initialized with the config from the metadata."""
    model_state, metadata = load_checkpoint(checkpoint_dir, step, device)
    config = GPTConfig(**metadata["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    clean_state = {key.removeprefix("_orig_mod."): value for key, value in model_state.items()}
    if device.type in {"cpu", "mps"}:
        clean_state = {
            key: value.float() if getattr(value, "dtype", None) == torch.bfloat16 else value
            for key, value in clean_state.items()
        }
    model.load_state_dict(clean_state, strict=True, assign=True)
    model.eval() if phase == "eval" else model.train()
    tokenizer = get_tokenizer()
    return model, tokenizer, metadata


def load_model(source, device, phase="eval", model_tag=None, step=None):
    """Load a model checkpoint from the specified source ("base" or "sft"), returning the model, tokenizer, and metadata. If model_tag is not specified, find the most recent or deepest tag. If step is not specified, find the last step in the checkpoint directory."""
    checkpoints_root = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
    }[source]
    checkpoints_dir = os.path.join(get_base_dir(), checkpoints_root)
    model_tag = find_model_tag(checkpoints_dir, model_tag)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    return build_model(checkpoint_dir, step, device, phase)
