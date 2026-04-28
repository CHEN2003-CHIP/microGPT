import json
import subprocess
import sys


def test_gqa_cache_memory_script_writes_reports(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/gqa_cache_memory.py",
            "--n-layer",
            "2",
            "--n-head",
            "4",
            "--n-embd",
            "128",
            "--seq-lens",
            "128",
            "256",
            "--batch-sizes",
            "1",
            "--report-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    json_path = tmp_path / "gqa_cache_memory.json"
    md_path = tmp_path / "gqa_cache_memory.md"
    csv_path = tmp_path / "gqa_cache_memory.csv"
    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["model_config"]["n_head"] == 4
    assert payload["metadata"]["torch_version"]
    assert payload["rows"]
    assert "JSON report:" in result.stdout
    assert "CSV report:" in result.stdout


def test_kv_cache_generation_script_writes_reports(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/kv_cache_generation.py",
            "--device-type",
            "cpu",
            "--n-layer",
            "2",
            "--n-head",
            "4",
            "--n-kv-head",
            "2",
            "--n-embd",
            "128",
            "--max-seq-len",
            "24",
            "--prompt-len",
            "16",
            "--new-tokens",
            "8",
            "--warmup-runs",
            "0",
            "--runs",
            "1",
            "--report-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    json_path = tmp_path / "kv_cache_generation.json"
    md_path = tmp_path / "kv_cache_generation.md"
    csv_path = tmp_path / "kv_cache_generation.csv"
    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["model_config"]["n_kv_head"] == 2
    assert payload["metadata"]["model_config"]["sequence_len"] == 24
    assert payload["metadata"]["torch_version"]
    assert payload["results"]["speedup"] > 0
    assert "speedup:" in result.stdout
    assert "CSV report:" in result.stdout
