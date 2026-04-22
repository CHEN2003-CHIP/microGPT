"""Dataset helpers for remote parquet-based pretraining data."""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

from microchat.common import get_base_dir


BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
DATA_DIR = os.path.join(get_base_dir(), "base_data_climbmix")


def index_to_filename(index):
    return f"shard_{index:05d}.parquet"


def list_parquet_files(data_dir=None):
    data_dir = DATA_DIR if data_dir is None else data_dir
    if not os.path.exists(data_dir):
        return []
    filenames = [name for name in os.listdir(data_dir) if name.endswith(".parquet") and not name.endswith(".tmp")]
    filenames.sort()
    return [os.path.join(data_dir, name) for name in filenames]


def download_single_file(index):
    import requests

    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")
    for attempt in range(1, 6):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(temp_path, filepath)
            return True
        except Exception as exc:
            print(f"Attempt {attempt}/5 failed for {filename}: {exc}")
            for path in (filepath, filepath + ".tmp"):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < 5:
                time.sleep(2 ** attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=2, help="Number of training shards to download")
    parser.add_argument("-w", "--num-workers", type=int, default=2, help="Number of parallel download workers")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    num_train_shards = min(args.num_files, MAX_SHARD)
    shard_ids = list(range(num_train_shards))
    shard_ids.append(MAX_SHARD)  # always keep one validation shard

    print(f"Target directory: {DATA_DIR}")
    print(f"Downloading {len(shard_ids)} shards with {args.num_workers} workers...")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, shard_ids)
    print(f"Downloaded {sum(bool(result) for result in results)}/{len(shard_ids)} shards")


if __name__ == "__main__":
    main()
