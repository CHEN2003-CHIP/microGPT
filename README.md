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

Useful advanced training flags:

- `--warmup-iters`, `--lr-decay-iters`, `--min-lr`: warmup + cosine decay schedule
- `--grad-clip`: clip gradient norm for stability
- `--save-every`: save periodic checkpoints during training
- `--resume`: continue from the latest checkpoint under the selected model tag
- `--max-train-tokens`: stop by token budget instead of only by step count
- `--standard-gpt-block`: disable experimental residual/value-embedding paths for A/B runs

Example for a stronger single-GPU recipe:

```bash
uv run python scripts/base_train.py --depth 6 --aspect-ratio 64 --head-dim 64 --max-seq-len 256 --num-iterations 2000 --total-batch-size 2048 --warmup-iters 100 --grad-clip 1.0 --save-every 200 --model-tag laptop_v2
```

Training metrics are appended to:

```text
.microchat/reports/<model-tag>_base_train_metrics.jsonl
```

A short Markdown summary is written to:

```text
.microchat/reports/<model-tag>_base_train_report.md
```

### 4. Run a base-model evaluation

You can run a short completion-style evaluation before SFT:

```bash
uv run python scripts/base_eval.py -g d4
```

Default behavior:

- Uses `.microchat/base_eval.jsonl` if present, otherwise falls back to `scripts/base_eval.sample.jsonl`
- Loads the latest base checkpoint for the selected model tag
- Writes Markdown and JSON reports to `.microchat/reports/base_eval/`

Each JSONL row looks like:

```json
{"id":"english_completion","prompt":"The capital of France is","contains_any":["paris"],"forbidden_contains":["london"]}
```

### 5. Run chat SFT

```bash
uv run python scripts/chat_sft.py --model-tag d4 --num-iterations 500 --max-seq-len 256 --total-batch-size 1024 --learning-rate 5e-6 --warmup-iters 50 --save-every 100 --smoltalk-limit 0 --ultrachat-limit 0 --local-repeat 80
```

SFT data behavior:

- Phase A (recommended first): uses a bundled English-first anchor dataset for identity/instruction tuning
- Phase B: mixes the anchor dataset with a light amount of `UltraChat 200k`
- Offline fallback: still works with the bundled anchor dataset only

Useful SFT flags:

- `--smoltalk-limit`: limit SmolTalk rows; `0` disables SmolTalk
- `--ultrachat-limit`: limit UltraChat 200k rows; `0` disables UltraChat
- `--local-repeat`: oversample local identity/instruction examples
- `--learning-rate`, `--min-lr`, `--warmup-iters`, `--lr-decay-iters`: stable SFT schedule
- `--grad-clip`: clip gradient norm to reduce collapse
- `--save-every`: save periodic SFT checkpoints
- `--resume`: continue the latest SFT checkpoint
- `--max-train-tokens`: stop by token budget instead of only by step count

Phase A, anchor-only English instruction tuning:

```bash
uv run python scripts/chat_sft.py --model-tag scale_12x768_anchor --num-iterations 500 --max-seq-len 256 --device-batch-size 1 --total-batch-size 1024 --learning-rate 5e-6 --warmup-iters 50 --save-every 100 --smoltalk-limit 0 --ultrachat-limit 0 --local-repeat 80
```

Phase B, light generalization mix after Phase A works:

```bash
uv run python scripts/chat_sft.py --model-tag scale_12x768_mix --num-iterations 800 --max-seq-len 256 --device-batch-size 1 --total-batch-size 1024 --learning-rate 3e-6 --warmup-iters 50 --save-every 100 --smoltalk-limit 0 --ultrachat-limit 5000 --local-repeat 50
```

Curated local identity/instruction data is resolved in this order:

- `.microchat/identity_conversations.jsonl` if you provide your own
- `scripts/identity_conversations.anchor_en.jsonl`
- `scripts/identity_conversations.curated.jsonl`
- `scripts/identity_conversations.sample.jsonl`

SFT metrics are appended to:

```text
.microchat/reports/<model-tag>_chat_sft_metrics.jsonl
```

A short Markdown summary is written to:

```text
.microchat/reports/<model-tag>_chat_sft_report.md
```

Periodic checkpoints are written to:

```text
.microchat/chatsft_checkpoints/<model-tag>/
```

SFT checkpoints are written to:

```text
.microchat/chatsft_checkpoints/d4/
```

### 6. Chat with the model

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

### 7. Run a chat evaluation benchmark

After SFT, you can run a small rule-based benchmark that prints pass/fail results,
computes an overall score, and writes Markdown plus JSON reports.

```bash
uv run python scripts/chat_eval.py -i sft -g d4
```

Default behavior:

- Loads the latest checkpoint for the selected model tag
- Uses `.microchat/chat_eval.jsonl` if present, otherwise falls back to `scripts/chat_eval.sample.jsonl`
- Runs deterministic generation with `temperature=0`
- Marks the model as `Qualified: YES/NO` using the overall pass threshold and required cases

Useful flags:

- `--eval-file`: custom benchmark JSONL path
- `--pass-threshold`: global qualification line, default `0.8`
- `--max-tokens`: cap response length during evaluation
- `--report-dir`: where to write the `.md` and `.json` reports

Example:

```bash
uv run python scripts/chat_eval.py -i sft -g d4 --pass-threshold 0.75 --max-tokens 96
```

Reports are written by default to:

```text
.microchat/reports/chat_eval/
```

Each JSONL benchmark row looks like this:

```json
{"id":"follow_instruction_cn","category":"instruction","required":true,"prompt":"只回复“苹果”，不要解释。","checks":{"exact":["苹果"]}}
```

Supported `checks` keys:

- `exact`: response must exactly match one of the candidate strings
- `contains_all`: every listed phrase must appear
- `contains_any`: at least one listed phrase must appear
- `regex_any`: at least one regex must match
- `forbidden_contains`: none of these phrases may appear
- `min_chars`: minimum response length

To compare multiple saved SFT checkpoints and pick the best step by chat score:

```bash
uv run python scripts/chat_eval_sweep.py -g scale_12x768_anchor --eval-file scripts/chat_eval.dev_en.jsonl
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
