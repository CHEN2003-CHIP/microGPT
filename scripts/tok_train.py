"""Train the tokenizer used by the microGPT teaching project."""

import argparse
import os
import time


parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument("--max-chars", type=int, default=5_000_000, help="Maximum characters to train on")
parser.add_argument("--doc-cap", type=int, default=4_000, help="Maximum characters per document")
parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
args = parser.parse_args()

import torch

from microchat.common import get_base_dir
from microchat.dataset import list_parquet_files
from microchat.tokenizer import RustBPETokenizer


def text_iterator():
    import pyarrow.parquet as pq

    total_chars = 0
    parquet_paths = list_parquet_files()
    assert parquet_paths, "No dataset shards found. Run `python -m microchat.dataset` first."
    for parquet_path in parquet_paths[:-1]:
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_index in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_index)
            for document in row_group.column("text").to_pylist():
                document = document[: args.doc_cap]
                total_chars += len(document)
                yield document
                if total_chars >= args.max_chars:
                    return


print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
tokenizer.save(tokenizer_dir)

token_bytes = []
special_tokens = set(tokenizer.get_special_tokens())
for token_id in range(tokenizer.get_vocab_size()):
    token_text = tokenizer.decode([token_id])
    token_bytes.append(0 if token_text in special_tokens else len(token_text.encode("utf-8")))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "wb") as handle:
    torch.save(token_bytes, handle)
print("Saved tokenizer and token_bytes.pt")
