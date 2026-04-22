# microGPT

microGPT is a teaching-first miniature GPT project that walks through the full local workflow: download pretraining shards, train a BPE tokenizer, pretrain a small base model, run chat-style supervised fine-tuning, and talk to the model from a CLI.

The goal is not to be the fastest or largest framework. The goal is to make the end-to-end pipeline understandable, hackable, and easy to run on a single machine.

## Highlights

- End-to-end workflow: dataset download, tokenizer training, base training, SFT, and CLI chat
- Teaching-oriented code: compact modules with a low amount of framework overhead
- Local-first outputs: caches, datasets, tokenizers, and checkpoints stay inside the repo by default
- Single-device simplicity: automatic `cuda` / `mps` / `cpu` detection
- Built-in chat data path: supports `SmolTalk` plus local JSONL identity/instruction conversations
- Basic test coverage for inference behavior, sampling, reproducibility, and KV cache logic

## Project Layout

```text
microGPT/
|-- microchat/             # Core model, tokenizer, engine, dataset, checkpoint code
|-- scripts/               # Training and inference entry points
|-- tasks/                 # SFT dataset wrappers
|-- tests/                 # Unit tests
|-- .microchat/            # Default local artifacts directory
|-- pyproject.toml         # Project metadata and dependencies
`-- README.md
```

## Requirements

- Python 3.10 or newer
- Recommended: [uv](https://docs.astral.sh/uv/) for environment management
- Runs on CPU, CUDA, or Apple Silicon `mps`

`torch` is configured in `pyproject.toml` with CPU and CUDA 12.8 sources.

Install for CPU:

```bash
uv sync --extra cpu
```

Install for GPU:

```bash
uv sync --extra gpu
```

Install development tools such as `pytest`:

```bash
uv sync --extra cpu --group dev
```

If you prefer another environment manager, install the dependencies from `pyproject.toml` manually.

## Quick Start

### 1. Download pretraining shards

```bash
uv run python -m microchat.dataset -n 2 -w 2
```

This downloads a small number of `climbmix` parquet shards plus one validation shard into:

```text
.microchat/base_data_climbmix/
```

### 2. Train the tokenizer

```bash
uv run python scripts/tok_train.py --max-chars 5000000 --doc-cap 4000 --vocab-size 8192
```

Outputs are written to:

```text
.microchat/tokenizer/
```

Expected files include:

- `tokenizer.pkl`
- `token_bytes.pt`

### 3. Train the base model

```bash
uv run python scripts/base_train.py --depth 4 --aspect-ratio 64 --head-dim 64 --max-seq-len 256 --num-iterations 20 --device-batch-size 1 --total-batch-size 512
```

By default, checkpoints are written to:

```text
.microchat/base_checkpoints/d4/
```

Use `--model-tag` if you want a custom checkpoint directory name.

### 4. Run chat SFT

```bash
uv run python scripts/chat_sft.py --model-tag d4 --num-iterations 20 --smoltalk-limit 20000 --local-repeat 50
```

SFT data behavior:

- Online: uses `SmolTalk` plus local identity/instruction JSONL data
- Offline fallback: uses the bundled `scripts/identity_conversations.sample.jsonl`

SFT checkpoints are written to:

```text
.microchat/chatsft_checkpoints/d4/
```

### 5. Chat with the model

```bash
uv run python scripts/chat_cli.py -i sft -g d4
```

Useful CLI flags:

- `-i, --source`: `base` or `sft`
- `-g, --model-tag`: checkpoint directory name
- `-s, --step`: specific checkpoint step
- `-p, --prompt`: single-prompt mode
- `-t, --temperature`: sampling temperature
- `-k, --top-k`: top-k sampling
- `-m, --max-tokens`: maximum generated tokens
- `--device-type`: force `cuda`, `cpu`, or `mps`

Single-prompt example:

```bash
uv run python scripts/chat_cli.py -i sft -g d4 -p "Introduce yourself."
```

## Local Data and Artifact Directories

By default, the project stores generated artifacts under:

```text
.microchat/
```

Typical subdirectories:

- `base_data_climbmix/`: downloaded parquet shards
- `hf_cache/`: Hugging Face cache
- `tokenizer/`: tokenizer files and token byte statistics
- `base_checkpoints/`: base model checkpoints
- `chatsft_checkpoints/`: SFT checkpoints

To move all local artifacts elsewhere, set:

```bash
MICROGPT_BASE_DIR=/your/path
```

You can also override compute dtype:

```bash
MICROGPT_DTYPE=float32
MICROGPT_DTYPE=float16
MICROGPT_DTYPE=bfloat16
```

If unset, the project prefers `bfloat16` on newer CUDA GPUs and otherwise falls back to `float32`.

## Custom SFT Data

The project looks for an optional local JSONL file at:

```text
.microchat/identity_conversations.jsonl
```

Each line should be a JSON array of alternating `user` and `assistant` messages, for example:

```json
[{"role":"user","content":"Who are you?"},{"role":"assistant","content":"I am a locally trained small model."}]
```

Rules:

- Each line must be a JSON array
- Messages must alternate as `user -> assistant -> user -> assistant`
- `content` is currently expected to be a string

If the file does not exist, SFT falls back to the bundled sample data so the pipeline still runs end to end.

## Tests

Run tests with:

```bash
uv run pytest
```

Current tests mainly cover:

- `Engine` sampling behavior
- determinism at `temperature=0`
- seed reproducibility
- multi-sample diversity
- `KVCache` basics

## What This Repo Is Good For

- Learning how a small GPT project fits together end to end
- Running a local tokenizer -> pretrain -> SFT pipeline without a huge framework
- Teaching, demos, experiments, and personal extensions
- Starting from a minimal codebase before adding more advanced features

## Known Boundaries

- The priority is clarity, not maximum throughput
- The current setup is single-machine oriented, not a full distributed training stack
- Default hyperparameters are tuned for quick iteration, not best possible model quality
- Final model quality depends heavily on data scale, training time, tokenizer size, and hardware

## Ideas for Extension

- Add richer experiment configuration and run tracking
- Add more evaluation scripts and benchmark reporting
- Support more chat data formats
- Add training dashboards and logging hooks
- Add cleaner export and serving paths

## Acknowledgment

This repository follows the spirit of small, readable GPT teaching projects: keep the core ideas visible, make the workflow runnable, and grow complexity only when needed.
