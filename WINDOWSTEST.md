# Windows CMD Test Guide

This guide shows how to test `microGPT` on Windows using `CMD` or `Anaconda Prompt`.

The instructions assume you are using the `pytorch` Conda environment shown in your screenshot, with Python 3.10.

## Recommended Environment

Expected Python version:

```cmd
Python 3.10.x
```

Expected interpreter path example:

```cmd
E:\anaconda3\envs\pytorch\python.exe
```

## 1. Open the Correct Shell

Use one of these:

- `Anaconda Prompt`
- `CMD` with the `pytorch` environment already activated

If needed, activate the environment first:

```cmd
conda activate pytorch
```

## 2. Enter the Project Directory

```cmd
cd /d "E:\Pycharm Project\MyGPT\microGPT"
```

## 3. Verify the Python Environment

Check that Windows is using the expected interpreter:

```cmd
python --version
where python
```

You should see Python 3.10 and a path similar to:

```cmd
E:\anaconda3\envs\pytorch\python.exe
```

## 4. Check Whether `torch` Is Installed

```cmd
python -c "import torch; print(torch.__version__)"
```

If this command fails, install the required dependencies first.

## 5. Install Project Dependencies

Install the project in editable mode:

```cmd
pip install -e .
```

Install test tools:

```cmd
pip install pytest
```

If `torch` is still missing, install it explicitly:

```cmd
pip install torch
```

## 6. Run Unit Tests

Run the core engine test first:

```cmd
python -m pytest tests\test_engine.py -v
```

Run all tests:

```cmd
python -m pytest
```

## 7. Run a Minimal End-to-End Smoke Test

This verifies that the project can actually run, not just import.

### Step 1: Download a Small Amount of Training Data

```cmd
python -m microchat.dataset -n 2 -w 2
```

### Step 2: Train the Tokenizer

```cmd
python scripts\tok_train.py --max-chars 5000000 --doc-cap 4000 --vocab-size 8192
```

### Step 3: Train a Small Base Model

```cmd
python scripts\base_train.py --depth 4 --aspect-ratio 64 --head-dim 64 --max-seq-len 256 --num-iterations 20 --device-batch-size 1 --total-batch-size 512
```

### Step 4: Run a Minimal SFT Pass

To reduce external dependency risk, this command uses local-only SFT data:

```cmd
python scripts\chat_sft.py --model-tag d4 --num-iterations 20 --smoltalk-limit 0 --local-repeat 20
```

### Step 5: Start the CLI Chat

```cmd
python scripts\chat_cli.py -i sft -g d4
```

These script commands now work directly from the repository root. If you prefer module-style execution, these are also valid:

```cmd
python -m scripts.tok_train --max-chars 5000000 --doc-cap 4000 --vocab-size 8192
python -m scripts.base_train --depth 4 --aspect-ratio 64 --head-dim 64 --max-seq-len 256 --num-iterations 20 --device-batch-size 1 --total-batch-size 512
python -m scripts.chat_sft --model-tag d4 --num-iterations 20 --smoltalk-limit 0 --local-repeat 20
python -m scripts.chat_cli -i sft -g d4
```

## 8. Default Output Directory

Generated files are stored under:

```cmd
E:\Pycharm Project\MyGPT\microGPT\.microchat
```

Common subdirectories:

```cmd
.microchat\base_data_climbmix
.microchat\tokenizer
.microchat\base_checkpoints
.microchat\chatsft_checkpoints
```

## 9. Common Problems

If `microchat` cannot be imported:

```cmd
pip install -e .
```

If `torch` is missing:

```cmd
pip install torch
```

If `pytest` is not recognized:

```cmd
python -m pytest
```

If dataset download fails, the most common cause is network access to Hugging Face.

## 10. Fastest Basic Validation Path

If you only want the shortest useful test sequence, run these commands:

```cmd
cd /d "E:\Pycharm Project\MyGPT\microGPT"
conda activate pytorch
python --version
python -c "import torch; print(torch.__version__)"
pip install -e .
pip install pytest
python -m pytest tests\test_engine.py -v
python -m pytest
```
