"""Simple single-device dataloaders for teaching pretraining."""

from __future__ import annotations

import torch

from microchat.dataset import list_parquet_files


def _iter_document_text(split):
    """Iterate over document text from the parquet files for the given split."""
    parquet_paths = list_parquet_files()
    assert parquet_paths, "No parquet files found. Run `python -m microchat.dataset` first."
    #取最后一个文件作为验证集，其他作为训练集
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    import pyarrow.parquet as pq

    while True:
        #循环读取 parquet 文件，提取 "text" 列的内容，作为文档文本
        for parquet_path in parquet_paths:
            parquet_file = pq.ParquetFile(parquet_path)
            for row_group_index in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_index)
                for text in row_group.column("text").to_pylist():
                    yield text

# 构建一个生成器函数，返回训练批次，每个批次包含输入 tokens 和目标 tokens
def token_batch_loader(tokenizer, batch_size, seq_len, split, device, tokenizer_threads=4, tokenizer_batch_size=128):
    """
    Build full-utilization next-token training batches.
    Each row begins with BOS and documents are packed greedily until the row is full.
    """
    row_capacity = seq_len + 1
    # 获取 tokenizer 的 BOS token ID，作为每个文档的起始标记
    bos_token = tokenizer.get_bos_token_id()
    # 构建一个生成器函数，返回文档文本
    documents = _iter_document_text(split)
    # 维护一个缓冲区，存储已经 tokenized 的文档 tokens，等待被打包成训练批次
    buffer = []
    use_cuda = device.type == "cuda"


    def refill_buffer(min_size=512):
        """Refill the buffer with tokenized documents until it reaches the minimum size."""
        while len(buffer) < min_size:
            text_batch = [next(documents) for _ in range(tokenizer_batch_size)]
            token_lists = tokenizer.encode(text_batch, prepend=bos_token, num_threads=tokenizer_threads)
            buffer.extend(token_lists)

    while True:
        refill_buffer()
        rows = []
        for _ in range(batch_size):
            row = []
            while len(row) < row_capacity:
                remaining = row_capacity - len(row)
                candidate_index = -1
                candidate_len = 0
                for index, document_tokens in enumerate(buffer):
                    doc_len = len(document_tokens)
                    if doc_len <= remaining and doc_len > candidate_len:
                        candidate_index = index
                        candidate_len = doc_len
                if candidate_index >= 0:
                    row.extend(buffer.pop(candidate_index))
                    continue
                shortest_index = min(range(len(buffer)), key=lambda index: len(buffer[index]))
                row.extend(buffer.pop(shortest_index)[:remaining])
            rows.append(row[:row_capacity])

        batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch[:, :-1].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        targets = batch[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        yield inputs, targets
