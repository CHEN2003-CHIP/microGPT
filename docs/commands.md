# microGPT Command Guide

This is a quick command reference for the local teaching workflow. Run commands from the repository root.

## Environment

CPU:

```bash
uv sync --extra cpu --group dev
```

GPU:

```bash
uv sync --extra gpu --group dev
```

Conda users can activate the conda environment first, then install into it:

```bash
conda activate microgpt
pip install uv
uv sync --extra cpu --group dev --active
```

Use `--extra gpu` instead of `--extra cpu` for CUDA 12.8 PyTorch.

## Tests

Full test suite:

```bash
python -m pytest
```

Focused model and benchmark tests:

```bash
python -m pytest tests/test_model_refactor.py tests/test_experiments.py
```

## Dataset And Tokenizer

Download a small base pretraining dataset:

```bash
python -m microchat.dataset -n 2 -w 2
```

Important parameters:

- `-n / --num-files`: number of training parquet shards.
- `-w / --num-workers`: parallel download workers.

Train the default tokenizer:

```bash
python scripts/tok_train.py --max-chars 5000000 --doc-cap 4000 --vocab-size 8192
```

Important parameters:

- `--max-chars`: total training text budget.
- `--doc-cap`: max characters per document.
- `--vocab-size`: tokenizer vocabulary size. Keep this aligned with the model checkpoints you train.

Outputs:

```text
.microchat/tokenizer/tokenizer.pkl
.microchat/tokenizer/token_bytes.pt
```

## Base Training

CPU smoke test:

```bash
python scripts/base_train.py --device-type cpu --num-iterations 1 --eval-every 1 --eval-tokens 256 --device-batch-size 1 --total-batch-size 256 --max-seq-len 256
```

Small default local run:

```bash
python scripts/base_train.py --device-type cpu --depth 4 --aspect-ratio 64 --head-dim 64 --max-seq-len 256 --num-iterations 20 --device-batch-size 1 --total-batch-size 512 --eval-every 10
```

CUDA run:

```bash
python scripts/base_train.py --device-type cuda --depth 4 --aspect-ratio 64 --head-dim 64 --max-seq-len 256 --num-iterations 20 --device-batch-size 1 --total-batch-size 512 --eval-every 10
```

Important parameters:

- `--depth`: number of transformer blocks.
- `--aspect-ratio`: width is rounded from `depth * aspect_ratio`.
- `--head-dim`: attention head dimension.
- `--max-seq-len`: context length.
- `--window-pattern`: attention window pattern, such as `L` or `SL`.
- `--num-iterations`: optimizer steps.
- `--device-batch-size`: rows per micro-step.
- `--total-batch-size`: tokens per optimizer step; must divide by `device_batch_size * max_seq_len`.
- `--learning-rate`, `--min-lr`, `--warmup-iters`, `--lr-decay-iters`: LR schedule.
- `--eval-every`, `--eval-tokens`: validation cadence and budget.
- `--save-every`: periodic checkpoint interval; final checkpoint is always saved.
- `--model-tag`: checkpoint folder name.
- `--resume`: resume the latest checkpoint for the selected tag.
- `--standard-gpt-block`: disables experimental residual/value-embedding paths.

Outputs:

```text
.microchat/base_checkpoints/<model-tag>/
.microchat/reports/<model-tag>_base_train_metrics.jsonl
.microchat/reports/<model-tag>_base_train_report.md
```

## Base Evaluation

Evaluate a base checkpoint:

```bash
python scripts/base_eval.py --device-type cpu --model-tag d4 --max-tokens 48
```

Important parameters:

- `--model-tag`: checkpoint tag under `.microchat/base_checkpoints`.
- `--step`: specific checkpoint step; omitted means latest.
- `--eval-file`: custom JSONL eval file.
- `--temperature`, `--top-k`, `--max-tokens`: generation settings.
- `--report-dir`: custom report output directory.

## Chat SFT Training

CPU smoke SFT from a base checkpoint:

```bash
python scripts/chat_sft.py --device-type cpu --model-tag d4 --num-iterations 1 --eval-every 1 --eval-tokens 256 --device-batch-size 1 --total-batch-size 512 --max-seq-len 512 --smoltalk-limit 0 --ultrachat-limit 0
```

Local-data SFT run:

```bash
python scripts/chat_sft.py --device-type cpu --model-tag d4 --num-iterations 800 --device-batch-size 1 --total-batch-size 512 --max-seq-len 512 --eval-every 100 --save-every 100 --smoltalk-limit 0 --ultrachat-limit 0
```

CUDA SFT run:

```bash
python scripts/chat_sft.py --device-type cuda --model-tag d4 --num-iterations 800 --device-batch-size 1 --total-batch-size 512 --max-seq-len 512 --eval-every 100 --save-every 100
```

Important parameters:

- `--model-tag`: base checkpoint tag to load and SFT checkpoint tag to write.
- `--model-step`: specific base checkpoint step.
- `--max-seq-len`: SFT conversation context length.
- `--pack-conversations`: pack multiple short conversations into one row.
- `--sft-optimizer`: `adamw_full` or `adamw_behavior_low_lr`.
- `--behavior-lr-scale`: lower LR multiplier for behavior parameters in low-LR mode.
- `--smoltalk-limit`, `--ultrachat-limit`: optional remote dataset row limits.
- `--local-repeat`: oversampling count for bundled local JSONL data.
- `--resume`: resume latest SFT checkpoint for the selected tag.

Outputs:

```text
.microchat/chatsft_checkpoints/<model-tag>/
.microchat/reports/<model-tag>_chat_sft_metrics.jsonl
.microchat/reports/<model-tag>_chat_sft_report.md
```

## Chat CLI

Run a prompt against the latest SFT checkpoint:

```bash
python scripts/chat_cli.py --source sft --model-tag d4 --device-type cpu --prompt "hello" --max-tokens 128
```

Use a base checkpoint instead:

```bash
python scripts/chat_cli.py --source base --model-tag d4 --device-type cpu --prompt "Once upon a time" --max-tokens 128
```

Important parameters:

- `--source`: `base` or `sft`.
- `--model-tag`, `--step`: checkpoint selection.
- `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`: sampling controls.
- `--stop-on-user-start`: stop if the model begins a new user turn.

## Chat Evaluation

Evaluate one checkpoint:

```bash
python scripts/chat_eval.py --source sft --model-tag d4 --device-type cpu --max-tokens 128
```

Sweep saved SFT checkpoints:

```bash
python scripts/chat_eval_sweep.py --model-tag d4 --device-type cpu --max-tokens 128
```

Important parameters:

- `--eval-file`: custom benchmark JSONL.
- `--pass-threshold`: pass line from `0.0` to `1.0`.
- `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`: generation settings.

## MiniDeepSeek Experiments

GQA KV cache memory estimate:

```bash
python experiments/gqa_cache_memory.py --n-layer 2 --n-head 4 --n-embd 128 --seq-lens 128 256 --batch-sizes 1
```

KV cache generation benchmark:

```bash
python experiments/kv_cache_generation.py --device-type cpu --n-layer 2 --n-head 4 --n-kv-head 2 --n-embd 128 --max-seq-len 24 --prompt-len 16 --new-tokens 8 --warmup-runs 1 --runs 2
```

Important parameters:

- `--n-head`: query heads.
- `--n-kv-head`: KV heads for generation benchmark.
- `--n-kv-heads`: KV head list for memory estimate.
- `--max-seq-len`: cache/model capacity for generation benchmark.
- `--prompt-len`, `--new-tokens`: measured generation shape.
- `--report-dir`: custom report directory.

Reports:

```text
.microchat/reports/experiments/gqa_cache_memory.json
.microchat/reports/experiments/gqa_cache_memory.md
.microchat/reports/experiments/gqa_cache_memory.csv
.microchat/reports/experiments/kv_cache_generation.json
.microchat/reports/experiments/kv_cache_generation.md
.microchat/reports/experiments/kv_cache_generation.csv
```
