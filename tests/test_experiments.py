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


def test_moe_router_stats_script_writes_reports(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/moe_router_stats.py",
            "--device-type",
            "cpu",
            "--n-layer",
            "2",
            "--n-head",
            "2",
            "--n-kv-head",
            "2",
            "--n-embd",
            "64",
            "--vocab-size",
            "128",
            "--seq-len",
            "8",
            "--batch-size",
            "2",
            "--num-experts",
            "4",
            "--moe-top-k",
            "2",
            "--report-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    json_path = tmp_path / "moe_router_stats.json"
    md_path = tmp_path / "moe_router_stats.md"
    csv_path = tmp_path / "moe_router_stats.csv"
    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["model_config"]["ffn_type"] == "moe"
    assert payload["metadata"]["model_config"]["num_experts"] == 4
    assert payload["results"]["total_assignments"] == payload["results"]["expected_total_assignments"]
    assert "assignments:" in result.stdout
    assert "CSV report:" in result.stdout


def test_moe_dense_compare_script_writes_reports(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "experiments/moe_dense_compare.py",
            "--device-type",
            "cpu",
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--steps",
            "1",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--n-kv-head",
            "1",
            "--n-embd",
            "64",
            "--vocab-size",
            "128",
            "--num-experts",
            "4",
            "--moe-top-k",
            "2",
            "--report-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    json_path = tmp_path / "moe_dense_compare.json"
    md_path = tmp_path / "moe_dense_compare.md"
    csv_path = tmp_path / "moe_dense_compare.csv"
    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = {row["mode"]: row for row in payload["rows"]}
    assert set(rows) == {"dense", "moe"}
    for row in rows.values():
        assert row["total_params"] > 0
        assert row["trainable_params"] > 0
        assert row["forward_time_sec"] > 0
        assert row["tokens_per_sec"] > 0
        assert row["ce_loss"] > 0
        assert row["total_loss"] > 0
    assert rows["dense"]["moe_aux_loss"] == 0.0
    assert rows["moe"]["moe_aux_loss"] > 0
    assert len(rows["moe"]["expert_usage_ratios"]) == 4
    assert "dense:" in result.stdout
    assert "moe:" in result.stdout
    assert "CSV report:" in result.stdout
